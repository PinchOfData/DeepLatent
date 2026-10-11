#!/usr/bin/env Rscript
# Canonical Gentzkow–Shapiro–Taddy left/right classifier.
#
# There is one production specification: source texts are aggregated to the
# substantive political-document cells below before both fitting and scoring.
# There is no raw-document, tagged-output, alternate-vocabulary, or sampled
# branch.
#
# Scope: all sources for the selected country are pooled into a single fit.
#   us = congress + tweet + tv_ad + website
#   fr = parliament + professions_de_foi
#
#   congress:            politician x election that produced the term
#   tweet:               politician x election that produced the term
#   tv_ad:               politician x election x primary/general x race/seat
#   website:             politician x election x primary/general
#   parliament:          politician x election that produced the term
#   professions_de_foi:  one standard (falc == 0) source document per cell
#
# This is a CPU-only HPC workload. Use the canonical SLURM wrapper documented
# in Codes/python/scripts/HPC_WORKFLOW.md.
#
# Usage:
#   Rscript Codes/R/02_party_classifier_gst.R --country us
#   Rscript Codes/R/02_party_classifier_gst.R --country fr
#
# Outputs:
#   Codes/python/ckpt/party_classifier/{us,fr}/gst_R/{fit.rds,
#                                                     vocabulary.rds,
#                                                     covariates.rds}
#   Temp/party_classifier/train_predictions_{us,fr}_gst_R.csv
#   Temp/party_classifier/gst_phrase_contributors_{country}_{source}.csv
#   Tables/party_classifier_word_loadings_{us,fr}_gst_R.csv
#   Tables/party_classifier_{us,fr}_gst_R.tex

suppressPackageStartupMessages({
  required <- c("distrom", "gamlr", "Matrix", "data.table", "here", "parallel")
  missing  <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing) > 0) {
    stop(sprintf(
      "Missing R packages: %s. Install with install.packages(c(%s)).",
      paste(missing, collapse = ", "),
      paste(shQuote(missing), collapse = ", ")
    ))
  }
  library(distrom)
  library(Matrix)
  library(data.table)
  library(here)
  library(parallel)
})

# ============================================================================
# Canonical analytical configuration and resource-only tuning
# ============================================================================

# Vectorizer settings.
#   token_pattern = r"(?u)\b\w{3,}\b"   → tokens of length >= 3
# The sole vocabulary is the manually curated production (1,2)-gram list. Greedy
# matching counts a retained bigram once rather than also counting its two
# component unigrams.
TOKEN_MIN_LEN <- 3L
SEED          <- 42L

# These environment variables change only parallelism/memory use, never the
# sample, cell definition, vocabulary, or estimand.
AGG_CHUNK_DOCS <- as.integer(Sys.getenv("GST_R_AGG_CHUNK", unset = "100000"))

# dmr() arguments — verbatim from compute.r
LAMBDA_MIN_RATIO <- 1e-5
NLAMBDA          <- 100L
FIXEDCOST        <- 1e-5
N_CORES          <- as.integer(Sys.getenv("GST_R_NCORES", unset = "2"))
# π computation batch size: held N×V dense blocks per batch are ~6× B × V × 8B.
# B=200, V=50k → ~480 MB per batch; safe on a laptop. Bump on HPC if desired.
PI_BATCH_SIZE    <- as.integer(Sys.getenv("GST_R_PI_BATCH", unset = "200"))
CELL_IDENTITY_VERSION <- "2026-08-16_contextual_us_identity_v3"
GST_NUISANCE_DESIGN_VERSION <-
  "2026-08-16_source_x_stage_plus_election_year_v1"

if (is.na(AGG_CHUNK_DOCS) || AGG_CHUNK_DOCS < 1L)
  stop("GST_R_AGG_CHUNK must be a positive integer.")
if (is.na(N_CORES) || N_CORES < 1L)
  stop("GST_R_NCORES must be a positive integer.")
if (is.na(PI_BATCH_SIZE) || PI_BATCH_SIZE < 1L)
  stop("GST_R_PI_BATCH must be a positive integer.")

PROJECT_ROOT <- here::here()
DATA_DIR     <- file.path(PROJECT_ROOT, "Data")
CKPT_ROOT    <- file.path(PROJECT_ROOT, "Codes", "python", "ckpt",
                          "party_classifier")
TABLES_DIR   <- file.path(PROJECT_ROOT, "Tables")
TEMP_DIR     <- file.path(PROJECT_ROOT, "Temp", "party_classifier")
WEBSITE_IDENTITY_OVERRIDES_RELATIVE <- file.path(
  "Data", "crosswalks", "us_website_identity_overrides.csv")

website_identity_overrides_path <- function() {
  file.path(PROJECT_ROOT, WEBSITE_IDENTITY_OVERRIDES_RELATIVE)
}

checkpoint_dir <- function(country) {
  file.path(CKPT_ROOT, country, "gst_R")
}

log_info <- function(...) {
  cat(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), "[INFO]", ..., "\n", sep = " ")
}

log_mem <- function(tag = "") {
  g <- gc(verbose = FALSE, reset = FALSE, full = TRUE)
  used_mb <- sum(g[, 2])
  log_info(sprintf("RSS ≈ %.0f MB after gc()%s",
                   used_mb,
                   if (nzchar(tag)) paste0(" (", tag, ")") else ""))
}

# ============================================================================
# Sole curated GST vocabulary
# ============================================================================

VOCAB_DIR <- file.path(PROJECT_ROOT, "Codes", "python", "utils")

frozen_vocab_relative_path <- function(country) {
  file.path("Codes", "python", "utils", sprintf("vocab_%s.txt", country))
}

frozen_vocab_path <- function(country) {
  file.path(PROJECT_ROOT, frozen_vocab_relative_path(country))
}

load_frozen_vocab <- function(country) {
  path <- frozen_vocab_path(country)
  if (!file.exists(path)) stop("Missing canonical curated vocabulary: ", path)
  v <- readLines(path, encoding = "UTF-8", warn = FALSE)
  if (any(!nzchar(v)) || any(v != trimws(v)) || any(v != tolower(v)) ||
      anyDuplicated(v) || any(lengths(strsplit(v, " ", fixed = TRUE)) > 2L)) {
    stop("Canonical vocabulary must contain unique, lowercase, trimmed uni-/bigrams: ",
         path)
  }
  v
}


# ============================================================================
# French parliamentary group → left/right mapping
# (transcribed from the archived Codes/python/scripts/old/02_party_classifier.py)
# ============================================================================

# Manifesto sources kept for France (Data/professions_de_foi/professions_de_foi.csv
# `data_source`): legislative elections only.
FR_LEGISLATIVE_MANIFESTO_SOURCES <- c("v1", "v2_legislative", "v3_legislative")

FR_GROUP_TO_LABEL <- c(
  # Left (0)
  "Socialiste" = 0L,
  "Socialiste, républicain et citoyen" = 0L,
  "Socialiste, radical et citoyen et divers gauche" = 0L,
  "Socialiste, radical et citoyen" = 0L,
  "Socialistes et apparentés" = 0L,
  "Socialiste, écologiste et républicain" = 0L,
  "Socialiste écologiste et républicain" = 0L,
  "Apparenté socialiste" = 0L,
  "Apparenté socialiste, radical et citoyen et divers gauche" = 0L,
  "Parti socialiste et radicaux de gauche" = 0L,
  "Communiste" = 0L,
  "Communiste -1" = 0L,
  "Communistes et républicains" = 0L,
  "Apparenté Communiste" = 0L,
  "Apparenté communiste" = 0L,
  "Gauche démocrate et républicaine" = 0L,
  "Gauche démocrate et républicaine - NUPES" = 0L,
  "La France insoumise" = 0L,
  "La France insoumise - Nouvelle Union Populaire écologique et sociale" = 0L,
  "Nouvelle Gauche" = 0L,
  "Écologiste" = 0L,
  "Écologiste - NUPES" = 0L,
  "Fédération de la gauche démocrate et socialiste" = 0L,
  "Fédération de la gauche démocrate et socialiste - Non inscrit" = 0L,
  "Radical, citoyen et verts" = 0L,
  "Radical, républicain, démocrate et progressiste" = 0L,
  "Radical, républicain et démocrate progressiste" = 0L,
  "Radicaux de gauche" = 0L,
  "Rassemblement démocratique" = 0L,
  # Right (1) — Gaullist family
  "Rassemblement pour la République" = 1L,
  "Apparenté au Rassemblement pour la République" = 1L,
  "Rassemblement pour la République - Non inscrit" = 1L,
  "Union pour un mouvement populaire" = 1L,
  "Union pour un Mouvement Populaire" = 1L,
  "Apparenté à l'Union pour un mouvement populaire" = 1L,
  "Apparenté Union pour un Mouvement Populaire" = 1L,
  "Union pour la majorité présidentielle" = 1L,
  "Apparenté Union pour la majorité présidentielle" = 1L,
  "Apparentée Union pour la majorité présidentielle" = 1L,
  "Les Républicains" = 1L,
  "Union des démocrates pour la République" = 1L,
  "Union des démocrates pour la  République" = 1L,
  "Union des démocrates pour la République - Non inscrit" = 1L,
  "Union pour la nouvelle République-UDT" = 1L,
  "Union pour la nouvelle République" = 1L,
  "Union démocratique pour la V° République" = 1L,
  "Unité de la République" = 1L,
  "Républicains indépendants" = 1L,
  "Républicains indépendants - Réélu le 21 novembre 1976" = 1L,
  "Démocratie libérale" = 1L,
  "Indépendants et paysans d'action sociale" = 1L,
  # Right (1) — Far-right
  "Front national" = 1L,
  "Front national -6" = 1L,
  "Rassemblement National" = 1L
)

# ============================================================================
# Canonical cell loaders
# ============================================================================

read_clean <- function(rel_path, cols = c("raw_idx", "doc_clean")) {
  path <- file.path(DATA_DIR, "processed", rel_path)
  if (!file.exists(path)) stop("Missing cleaned CSV: ", path)
  dt <- fread(path, select = cols, encoding = "UTF-8")
  dt <- dt[!is.na(doc_clean) & nchar(trimws(doc_clean)) > 0]
  dt
}

key_part <- function(x) {
  x <- as.character(x)
  x[is.na(x) | !nzchar(trimws(x))] <- "<NA>"
  gsub("\u241f", " ", x, fixed = TRUE)
}

make_cell_key <- function(...) {
  parts <- lapply(list(...), key_part)
  do.call(paste, c(parts, sep = "\u241f"))
}

website_candidate_pair_key <- function(website_context, a, b) {
  a <- as.character(a); b <- as.character(b)
  paste(website_context, pmin(a, b), pmax(a, b), sep = "\u241e")
}

load_website_identity_overrides <- function() {
  path <- website_identity_overrides_path()
  if (!file.exists(path))
    stop("Missing reviewed website identity overrides: ", path)
  x <- fread(path, encoding = "UTF-8")
  required <- c(
    "state", "district", "year", "party", "candidate_general",
    "candidate_primary", "decision", "evidence", "review_status"
  )
  missing <- setdiff(required, names(x))
  if (length(missing) > 0L)
    stop("Website identity overrides are missing columns: ",
         paste(missing, collapse = ", "))
  x[, `:=`(
    state = tolower(trimws(as.character(state))),
    district = as.integer(district),
    year = as.integer(year),
    party = tolower(trimws(as.character(party))),
    candidate_general = trimws(as.character(candidate_general)),
    candidate_primary = trimws(as.character(candidate_primary)),
    decision = tolower(trimws(as.character(decision))),
    evidence = trimws(as.character(evidence)),
    review_status = trimws(as.character(review_status))
  )]
  invalid <- x[
    is.na(state) | !nzchar(state) | is.na(district) | is.na(year) |
      is.na(party) | !nzchar(party) |
      is.na(candidate_general) | !nzchar(candidate_general) |
      is.na(candidate_primary) | !nzchar(candidate_primary) |
      !decision %chin% c("merge", "keep") |
      is.na(evidence) | !nzchar(evidence) |
      is.na(review_status) | !nzchar(review_status)
  ]
  if (nrow(invalid) > 0L)
    stop("Website identity overrides contain ", nrow(invalid),
         " incomplete or invalid rows.")
  x[, website_context := make_cell_key(state, district, year, party)]
  x[, override_key := website_candidate_pair_key(
    website_context, candidate_general, candidate_primary
  )]
  if (anyDuplicated(x$override_key))
    stop("Website identity overrides contain duplicate campaign/name pairs.")
  x
}

election_year_from_calendar <- function(year) {
  year <- as.integer(year)
  year - ifelse(year %% 2L == 1L, 1L, 2L)
}

normalize_candidate_key <- function(x) {
  x <- tolower(trimws(as.character(x)))
  x <- iconv(x, from = "UTF-8", to = "ASCII//TRANSLIT", sub = "")
  x <- gsub("[^a-z0-9]+", "_", x)
  gsub("^_+|_+$", "", x)
}

candidate_first_two <- function(x) {
  vapply(strsplit(x, "_", fixed = TRUE), function(z) {
    z <- z[nzchar(z)]
    paste(head(z, 2L), collapse = "_")
  }, character(1))
}

# Candidate labels in the two website collections use different conventions:
# primary slugs are surname-first (``akin_todd``), while Wayback labels are
# usually ``Surname, Given Names``.  The first-pass alias table below resolves
# exact and prefix cases.  These helpers provide a second, context-bounded
# reconciliation for initials, middle names, common nicknames, and compound
# surnames.  A match is considered only within the exact
# state/district/year/party campaign, so a shared name in another seat can
# never join two politicians.
WEBSITE_NAME_SUFFIXES <- c("jr", "sr", "ii", "iii", "iv", "v")
WEBSITE_NICKNAME_GROUPS <- strsplit(
  c(
    "william bill billy will liam", "robert bob bobby rob",
    "charles charlie chuck", "joseph joe joey", "michael mike mick",
    "steven stephen steve", "gregory greg", "zachary zach zack",
    "richard rick dick", "thomas tom tommy", "edward ed eddie ted",
    "theodore ted teddy", "james jim jimmy", "john jack johnny",
    "christopher chris", "patrick pat", "daniel dan danny",
    "donald don", "ronald ron", "kenneth ken kenny", "raymond ray",
    "gerald gerry jerry", "lawrence larry", "francis frank",
    "frederick fred", "anthony tony", "nicholas nick", "timothy tim",
    "benjamin ben", "samuel sam", "alexander alex", "jeffrey jeff",
    "jonathan jon", "mathew matthew matt", "andrew andy",
    "douglas doug", "philip phillip phil", "leonard len",
    "catherine katherine kathryn kathy kate cathy",
    "elizabeth liz beth betty", "margaret maggie meg peggy",
    "susan sue susie", "rebecca becky", "deborah debbie",
    "jennifer jen jenny", "jacqueline jackie", "barbara barb"
  ),
  " ", fixed = TRUE
)

website_name_tokens <- function(x) {
  z <- strsplit(normalize_candidate_key(x), "_", fixed = TRUE)[[1L]]
  z <- z[nzchar(z)]
  # Some slugs attach Jr/Sr to the given name (e.g. brown_henryjr).
  z <- vapply(z, function(token) {
    stripped <- sub("(jr|sr)$", "", token)
    if (nchar(stripped) >= 2L) stripped else token
  }, character(1))
  z[!z %chin% WEBSITE_NAME_SUFFIXES & nzchar(z)]
}

website_name_profiles <- function(x) {
  raw <- trimws(as.character(x))
  if (!nzchar(raw)) return(list())
  if (grepl(",", raw, fixed = TRUE)) {
    pieces <- strsplit(raw, ",", fixed = TRUE)[[1L]]
    surname <- website_name_tokens(pieces[1L])
    given <- website_name_tokens(paste(pieces[-1L], collapse = " "))
    if (length(surname) == 0L) return(list())
    return(list(list(surname = paste0(surname, collapse = ""), given = given)))
  }
  tokens <- website_name_tokens(raw)
  if (length(tokens) == 0L) return(list())
  if (length(tokens) == 1L)
    return(list(list(surname = tokens[1L], given = character())))
  # Slugs do not mark the boundary of compound surnames.  Try one to three
  # leading tokens, always leaving at least one token as a given name.
  max_surname <- min(3L, length(tokens) - 1L)
  lapply(seq_len(max_surname), function(k) {
    list(surname = paste0(tokens[seq_len(k)], collapse = ""),
         given = tokens[(k + 1L):length(tokens)])
  })
}

website_given_match_score <- function(a, b) {
  if (length(a) == 0L || length(b) == 0L) return(0L)
  if (nzchar(paste0(a, collapse = "")) &&
      identical(paste0(a, collapse = ""), paste0(b, collapse = "")))
    return(6L)
  full_a <- a[nchar(a) > 1L]
  full_b <- b[nchar(b) > 1L]
  best <- 0L
  for (x in full_a) for (y in full_b) {
    if (identical(x, y)) {
      best <- max(best, 6L)
      next
    }
    nickname <- any(vapply(
      WEBSITE_NICKNAME_GROUPS,
      function(group) x %chin% group && y %chin% group,
      logical(1)
    ))
    if (nickname) {
      best <- max(best, 5L)
    } else if (nchar(x) >= 3L && nchar(y) >= 3L &&
               (startsWith(x, y) || startsWith(y, x))) {
      best <- max(best, 4L)
    } else if (nchar(x) >= 4L && nchar(y) >= 4L && adist(x, y)[1L] <= 1L) {
      best <- max(best, 4L)
    }
  }
  if (best > 0L || (length(full_a) > 0L && length(full_b) > 0L)) return(best)
  if (any(outer(
    substr(a, 1L, 1L), substr(b, 1L, 1L), FUN = "=="
  ))) best <- 3L
  best
}

website_candidate_match_score <- function(a, b) {
  na <- normalize_candidate_key(a)
  nb <- normalize_candidate_key(b)
  if (!nzchar(na) || !nzchar(nb)) return(0L)
  if (identical(na, nb)) return(100L)
  if (identical(gsub("_", "", na, fixed = TRUE),
                gsub("_", "", nb, fixed = TRUE))) return(95L)
  pa <- website_name_profiles(a)
  pb <- website_name_profiles(b)
  best <- 0L
  for (xa in pa) for (xb in pb) {
    if (!identical(xa$surname, xb$surname)) next
    given <- website_given_match_score(xa$given, xb$given)
    if (given > 0L) best <- max(best, 80L + given)
  }
  best
}

website_profile_surnames <- function(x) {
  profiles <- website_name_profiles(x)
  if (length(profiles) == 0L) return(character())
  unique(vapply(profiles, function(profile) profile$surname, character(1)))
}

website_union_components <- function(n_nodes, left, right) {
  parent <- seq_len(n_nodes)
  find_root <- function(x) {
    root <- x
    while (parent[root] != root) root <- parent[root]
    while (parent[x] != x) {
      next_x <- parent[x]
      parent[x] <<- root
      x <- next_x
    }
    root
  }
  if (length(left) > 0L) {
    for (i in seq_along(left)) {
      a <- find_root(left[i]); b <- find_root(right[i])
      if (a != b) parent[max(a, b)] <- min(a, b)
    }
  }
  vapply(seq_len(n_nodes), find_root, integer(1))
}

unique_value_count <- function(x) uniqueN(x, na.rm = FALSE)

finish_cell_rows <- function(dt, labeled_only, left = "D", right = "R") {
  source_name <- unique(dt$source)
  if (length(source_name) != 1L) stop("Cell loader must contain exactly one source.")
  required <- c("raw_idx", "source", "election_year", "election_stage",
                "family", "label", "entity", "entity_scope", "cell_key",
                "doc_clean")
  missing <- setdiff(required, names(dt))
  if (length(missing) > 0L)
    stop("Cell loader is missing columns: ", paste(missing, collapse = ", "))
  if (anyNA(dt$election_year) || anyNA(dt$entity) ||
      any(!nzchar(trimws(dt$entity))))
    stop(source_name, " retained rows without a politician/election key.")

  if (labeled_only) {
    before <- nrow(dt)
    dt <- dt[!is.na(label)]
    log_info(sprintf("  %s label filter: retained %s/%s rows",
                     source_name, format(nrow(dt), big.mark = ","),
                     format(before, big.mark = ",")))
    mixed <- dt[, .(n_label = uniqueN(label)), by = cell_key][n_label > 1L]
    if (nrow(mixed) > 0L) {
      n_drop <- dt[cell_key %in% mixed$cell_key, .N]
      log_info(sprintf(paste0("  %s: excluded %s rows in %s cells whose party/",
                              "bloc changes within the same election"),
                       source_name, format(n_drop, big.mark = ","),
                       format(nrow(mixed), big.mark = ",")))
      dt <- dt[!cell_key %in% mixed$cell_key]
    }
    log_info(sprintf("  %s training rows: %s (%s=%s, %s=%s)",
                     source_name, format(nrow(dt), big.mark = ","),
                     left, format(sum(dt$label == 0L), big.mark = ","),
                     right, format(sum(dt$label == 1L), big.mark = ",")))
  } else {
    log_info(sprintf("  %s prediction rows: %s", source_name,
                     format(nrow(dt), big.mark = ",")))
  }
  if (nrow(dt) == 0L) stop("No rows remain for source ", source_name)
  dt
}

load_us_congress_cells <- function(labeled_only = TRUE) {
  log_info("Loading U.S. congressional politician-election cells...")
  members_path <- file.path(DATA_DIR, "us_congress", "HSall_members.csv")
  members <- fread(members_path,
                   select = c("bioguide_id", "party_code", "congress", "chamber"))
  members[, `:=`(bioguide_id = trimws(as.character(bioguide_id)),
                 congress = as.integer(congress),
                 chamber = tolower(trimws(as.character(chamber))))]
  members <- members[!is.na(bioguide_id) & nzchar(bioguide_id)]
  members <- members[, .(
    party_code = if (uniqueN(party_code) == 1L) party_code[1L] else NA_integer_,
    family = paste(sort(unique(as.character(party_code))), collapse = "|")
  ), by = .(bioguide_id, congress, chamber)]
  ambiguous <- members[is.na(party_code), .N]
  if (ambiguous > 0L)
    log_info(sprintf("  Congress roster: %s politician-Congress-chamber keys have multiple parties",
                     format(ambiguous, big.mark = ",")))

  clean <- read_clean("us_congress_clean.csv",
                      cols = c("raw_idx", "date", "congress", "chamber",
                               "speaker_bioguide", "doc_clean"))
  clean[, `:=`(speaker_bioguide = trimws(as.character(speaker_bioguide)),
               congress = as.integer(congress),
               chamber = tolower(trimws(as.character(chamber))))]
  clean[, election_year := 1786L + 2L * congress]
  speech_year <- as.integer(substr(clean$date, 1L, 4L))
  outside_term <- !is.na(speech_year) & !is.na(clean$election_year) &
    (speech_year < clean$election_year + 1L |
       speech_year > clean$election_year + 3L)
  if (any(outside_term))
    stop("Congress number/date fall outside the corresponding term in ",
         sum(outside_term), " rows.")
  clean <- clean[!is.na(election_year)]
  before_key_filter <- nrow(clean)
  clean <- clean[!is.na(speaker_bioguide) & nzchar(trimws(speaker_bioguide))]
  log_info(sprintf(paste0("  congress politician filter: excluded %s cleaned rows ",
                          "without a bioguide"),
                   format(before_key_filter - nrow(clean), big.mark = ",")))
  dt <- merge(clean, members,
              by.x = c("speaker_bioguide", "congress", "chamber"),
              by.y = c("bioguide_id", "congress", "chamber"), all.x = TRUE)
  dt[, `:=`(
    source = "congress",
    label = fifelse(party_code == 200L, 1L,
                    fifelse(party_code == 100L, 0L, NA_integer_)),
    entity = trimws(as.character(speaker_bioguide)),
    entity_scope = "us_legislator",
    election_stage = "office"
  )]
  dt[, cell_key := make_cell_key(source, entity, election_year)]
  dt <- dt[, .(raw_idx, source, election_year, election_stage, family, label,
               entity, entity_scope,
               cell_key, doc_clean)]
  finish_cell_rows(dt, labeled_only)
}

load_us_tweet_cells <- function(labeled_only = TRUE) {
  log_info("Loading U.S. tweet politician-election cells...")
  raw_path <- file.path(DATA_DIR, "tweets", "us_congress_tweets.csv")
  linked_path <- file.path(DATA_DIR, "tweets", "us_congress_tweets_linked.csv")
  raw <- fread(raw_path, select = c("id", "party"))
  linked <- fread(linked_path, select = c("id", "time", "bioguide"))
  if (nrow(raw) != nrow(linked)) {
    stop(sprintf("Tweet positional linkage row mismatch: raw=%d linked=%d",
                 nrow(raw), nrow(linked)))
  }
  set.seed(SEED)
  spot <- sort(sample.int(nrow(raw), min(100L, nrow(raw))))
  raw_id <- as.character(raw$id[spot])
  linked_id <- as.character(linked$id[spot])
  same <- (is.na(raw_id) & is.na(linked_id)) |
          (!is.na(raw_id) & !is.na(linked_id) & raw_id == linked_id)
  if (!all(same))
    stop("Tweet positional linkage failed sampled id-column verification.")
  log_info(sprintf("  tweet linkage verified: equal %s-row files and %d sampled ids",
                   format(nrow(raw), big.mark = ","), length(spot)))

  meta <- data.table(
    raw_idx = seq_len(nrow(raw)) - 1L,
    party = raw$party,
    time = linked$time,
    bioguide = trimws(as.character(linked$bioguide))
  )
  meta[, label := fifelse(party == "R", 1L,
                          fifelse(party == "D", 0L, NA_integer_))]
  meta[, election_year := election_year_from_calendar(
    as.integer(substr(as.character(time), 1L, 4L)))]

  clean <- read_clean("us_tweets_clean.csv",
                      cols = c("raw_idx", "congress", "doc_clean"))
  dt <- merge(clean, meta[, .(raw_idx, family = party, label, election_year,
                              bioguide)], by = "raw_idx")
  dt[, term_election_year := 1786L + 2L * as.integer(congress)]
  mismatch <- !is.na(dt$election_year) & !is.na(dt$term_election_year) &
    dt$election_year != dt$term_election_year
  if (any(mismatch)) {
    # Calendar-year arithmetic differs during the few January days before a
    # newly elected Congress begins. The cleaned file's exact session mapping
    # is authoritative for the office term that produced each tweet.
    log_info(sprintf(paste0("  tweet term boundary: corrected %s rows whose ",
                            "calendar-year shortcut selected the next election"),
                     format(sum(mismatch), big.mark = ",")))
  }
  dt[, election_year := term_election_year]
  dt[, term_election_year := NULL]
  before_key_filter <- nrow(dt)
  dt <- dt[!is.na(bioguide) & nzchar(trimws(bioguide))]
  log_info(sprintf(paste0("  tweet politician filter: excluded %s cleaned tweets ",
                          "from committee/party/caucus accounts without bioguides"),
                   format(before_key_filter - nrow(dt), big.mark = ",")))
  dt[, `:=`(source = "tweet", entity = bioguide,
            entity_scope = "us_legislator", election_stage = "office")]
  dt[, cell_key := make_cell_key(source, entity, election_year)]
  dt <- dt[, .(raw_idx, source, election_year, election_stage, family, label,
               entity, entity_scope,
               cell_key, doc_clean)]
  finish_cell_rows(dt, labeled_only)
}

derive_website_aliases <- function(raw) {
  u <- unique(raw[!is.na(candidate) & nzchar(trimws(candidate)),
                  .(candidate, state, district, year, party, data_source)])
  u[, norm := normalize_candidate_key(candidate)]
  u[, compact := gsub("_", "", norm, fixed = TRUE)]
  u[, first_two := candidate_first_two(norm)]
  primary <- u[data_source == "primary"]
  general <- u[data_source == "general_wayback"]

  # Globally unique exact normalized labels (e.g. turnham_joe ↔
  # "Turnham, Joe") are the highest-coverage, zero-fuzz linkage.  Candidate
  # labels are not globally person-unique, however; the website loader below
  # disambiguates reused labels with election-seat context before making cells.
  p_norm <- primary[, .(n_p = uniqueN(candidate),
                        p_candidate = if (uniqueN(candidate) == 1L) candidate[1L] else NA_character_),
                    by = norm][n_p == 1L & nzchar(norm)]
  exact <- merge(unique(general[, .(candidate_g = candidate, norm)]),
                 p_norm[, .(norm, candidate_p = p_candidate)], by = "norm")
  exact[, score := 4L]
  exact <- exact[, .(candidate_g, candidate_p, score)]

  # Candidate fields can differ by middle initials/suffixes.  Only consider
  # pairs observed in the same seat-year-party context, then require a unique
  # best match for each general-wayback name.
  ctx <- c("state", "district", "year", "party")
  gp <- merge(
    unique(general[, c(ctx, "candidate", "norm", "compact", "first_two"), with = FALSE]),
    unique(primary[, c(ctx, "candidate", "norm", "compact", "first_two"), with = FALSE]),
    by = ctx, allow.cartesian = TRUE, suffixes = c("_g", "_p")
  )
  if (nrow(gp) > 0L) {
    gp[, score := fifelse(norm_g == norm_p, 4L,
                   fifelse(compact_g == compact_p & nzchar(compact_g), 3L,
                   fifelse(startsWith(norm_g, paste0(norm_p, "_")) |
                           startsWith(norm_p, paste0(norm_g, "_")), 2L,
                   fifelse(first_two_g == first_two_p & nzchar(first_two_g), 1L, 0L))))]
    contextual <- gp[score > 0L,
                     .(candidate_g = candidate_g, candidate_p = candidate_p, score)]
  } else {
    contextual <- data.table(candidate_g = character(), candidate_p = character(),
                             score = integer())
  }

  # Reuse the precise irregular aliases produced by the candidate-measures
  # linkage workflow when they are present in both collections.
  xwalk_path <- file.path(DATA_DIR, "processed", "us_name_slug_xwalk.csv")
  if (file.exists(xwalk_path)) {
    xw <- fread(xwalk_path, select = c("name", "slug"))
    xw <- xw[name != slug & name %in% general$candidate & slug %in% primary$candidate]
    precise <- unique(xw[, .(candidate_g = name, candidate_p = slug, score = 5L)])
  } else {
    precise <- data.table(candidate_g = character(), candidate_p = character(),
                          score = integer())
  }

  candidates <- unique(rbindlist(list(precise, exact, contextual), use.names = TRUE))
  if (nrow(candidates) == 0L)
    return(data.table(candidate_g = character(), candidate_p = character(), score = integer()))
  best <- candidates[, {
    mx <- max(score)
    z <- unique(candidate_p[score == mx])
    .(candidate_p = if (length(z) == 1L) z else NA_character_,
      score = mx, ambiguous = length(z) != 1L)
  }, by = candidate_g]
  best[!ambiguous & !is.na(candidate_p), .(candidate_g, candidate_p, score)]
}

reconcile_website_candidate_components <- function(raw) {
  valid <- !is.na(raw$candidate) & nzchar(trimws(raw$candidate))
  if (!any(valid)) return(raw)

  # The campaign context is deliberately stricter than the eventual model
  # cell: party is included here to prevent two primary opponents with a shared
  # name from being reconciled across labels, and stage is excluded so the same
  # identity is used in both primary/general text periods.
  raw[, website_context := make_cell_key(state, district, year, party)]
  node_map <- unique(raw[valid, .(
    website_context, candidate, candidate_canonical, data_source
  )])
  source_meta <- node_map[, .(
    candidate_canonical = sort(unique(candidate_canonical))[1L],
    has_primary = any(data_source == "primary"),
    has_general = any(data_source == "general_wayback"),
    data_sources = paste(sort(unique(data_source)), collapse = "+"),
    source_prefix = if (any(data_source == "primary")) "primary" else
      sort(unique(data_source))[1L]
  ), by = .(website_context, candidate)]
  old_ambiguous <- node_map[, uniqueN(candidate_canonical),
                            by = .(website_context, candidate)][V1 > 1L]
  if (nrow(old_ambiguous) > 0L)
    stop("Website candidate/context labels map to multiple first-pass identities in ",
         nrow(old_ambiguous), " cases.")

  nodes <- source_meta
  setorder(nodes, website_context, candidate)
  nodes[, node_id := .I]
  pairs <- merge(
    nodes[, .(website_context, left = node_id, candidate_left = candidate,
              canonical_left = candidate_canonical,
              has_primary_left = has_primary,
              has_general_left = has_general,
              data_sources_left = data_sources)],
    nodes[, .(website_context, right = node_id, candidate_right = candidate,
              canonical_right = candidate_canonical,
              has_primary_right = has_primary,
              has_general_right = has_general,
              data_sources_right = data_sources)],
    by = "website_context", allow.cartesian = TRUE
  )[left < right]
  if (nrow(pairs) > 0L) {
    pairs[, algorithm_score := fifelse(
      canonical_left == canonical_right,
      1000L,
      mapply(website_candidate_match_score, candidate_left, candidate_right,
             USE.NAMES = FALSE)
    )]
    pairs[, score := algorithm_score]

    overrides <- load_website_identity_overrides()
    pairs[, override_key := website_candidate_pair_key(
      website_context, candidate_left, candidate_right
    )]
    matched <- match(overrides$override_key, pairs$override_key)
    if (anyNA(matched)) {
      stale <- overrides[is.na(matched), .(
        state, district, year, party, candidate_general, candidate_primary
      )]
      stop("Reviewed website identity overrides no longer match the input in ",
           nrow(stale), " cases.")
    }
    pairs[, `:=`(
      override_decision = NA_character_,
      override_evidence = NA_character_,
      override_review_status = NA_character_
    )]
    pairs[matched, `:=`(
      override_decision = overrides$decision,
      override_evidence = overrides$evidence,
      override_review_status = overrides$review_status
    )]
    pairs[override_decision == "merge", score := 2000L]
    pairs[override_decision == "keep", score := 0L]

    pairs[, cross_source :=
      (has_primary_left & has_general_right) |
      (has_general_left & has_primary_right)]
    pairs[, shares_surname := mapply(
      function(a, b) {
        length(intersect(website_profile_surnames(a),
                         website_profile_surnames(b))) > 0L
      },
      candidate_left, candidate_right,
      USE.NAMES = FALSE
    )]
    unresolved <- pairs[
      score == 0L & cross_source & shares_surname &
        is.na(override_decision)
    ]
    if (nrow(unresolved) > 0L)
      stop("Website identity audit found ", nrow(unresolved),
           " unreviewed cross-source same-surname pairs. Add explicit merge/keep ",
           "decisions to ", WEBSITE_IDENTITY_OVERRIDES_RELATIVE, ".")
    log_info(sprintf(paste0("  website reviewed identity overrides: %s merge, ",
                            "%s keep; no unreviewed same-surname pairs"),
                     format(overrides[decision == "merge", .N], big.mark = ","),
                     format(overrides[decision == "keep", .N], big.mark = ",")))
    edges <- pairs[score > 0L]
  } else {
    edges <- data.table(left = integer(), right = integer(), score = integer())
  }
  nodes[, component := website_union_components(
    .N, edges$left, edges$right
  )]

  # Prefer a native primary slug, then the shortest normalized candidate key.
  # This choice affects only the human-readable canonical label; every native
  # alias is retained in the prediction metadata.
  nodes[, normalized := normalize_candidate_key(candidate)]
  representatives <- nodes[
    order(component, !has_primary, nchar(normalized), normalized, candidate),
    .SD[1L], by = component
  ][, .(
    component,
    candidate_canonical_new = paste0(source_prefix, "::", key_part(candidate))
  )]
  nodes[representatives, on = "component",
        candidate_canonical_new := i.candidate_canonical_new]
  if (anyNA(nodes$candidate_canonical_new))
    stop("Website component reconciliation left candidate identities unresolved.")

  component_audit <- nodes[, .(
    n_native_labels = .N,
    n_first_pass = uniqueN(candidate_canonical),
    n_primary_labels = sum(has_primary)
  ), by = .(website_context, component)]
  bad_context <- nodes[, uniqueN(website_context), by = component][V1 != 1L]
  if (nrow(bad_context) > 0L)
    stop("Website identity components crossed campaign contexts.")
  max_component <- max(component_audit$n_native_labels)
  log_info(sprintf(paste0("  website contextual identity reconciliation: ",
                          "%s multi-label components; %s additional first-pass ",
                          "identities merged; largest component=%s"),
                   format(component_audit[n_native_labels > 1L, .N], big.mark = ","),
                   format(sum(pmax(component_audit$n_first_pass - 1L, 0L)),
                          big.mark = ","),
                   format(max_component, big.mark = ",")))

  # Optional exhaustive preflight artefacts.  The identity-smoke entry point
  # enables these files; production fitting and prediction do not.  They make
  # both sides of the linkage decision inspectable: every accepted edge and
  # every rejected cross-source pair that nevertheless shares a possible
  # surname under the compound-surname parser.
  audit_dir <- Sys.getenv("GST_R_IDENTITY_AUDIT_DIR", unset = "")
  if (nzchar(audit_dir)) {
    dir.create(audit_dir, recursive = TRUE, showWarnings = FALSE)
    context_meta <- unique(raw[valid, .(
      website_context, state, district, year, party
    )])
    context_conflict <- context_meta[, .N, by = website_context][N != 1L]
    if (nrow(context_conflict) > 0L)
      stop("Website audit contexts do not map one-to-one to campaign metadata.")

    edge_audit <- merge(copy(edges), context_meta,
                        by = "website_context", all.x = TRUE)
    edge_audit[, first_pass_same_identity :=
                 canonical_left == canonical_right]
    setorder(edge_audit, state, district, year, party,
             candidate_left, candidate_right)
    fwrite(edge_audit, file.path(
      audit_dir, "identity_audit_us_website_accepted_edges.csv"))

    component_nodes <- merge(copy(nodes), context_meta,
                             by = "website_context", all.x = TRUE)
    setorder(component_nodes, state, district, year, party,
             component, candidate)
    fwrite(component_nodes, file.path(
      audit_dir, "identity_audit_us_website_components.csv"))

    rejected <- copy(pairs[score == 0L])
    if (nrow(rejected) > 0L) {
      rejected <- rejected[cross_source & shares_surname]
    }
    rejected <- merge(rejected, context_meta,
                      by = "website_context", all.x = TRUE)
    setorder(rejected, state, district, year, party,
             candidate_left, candidate_right)
    fwrite(rejected, file.path(
      audit_dir, "identity_audit_us_website_residual_same_surname.csv"))
    log_info(sprintf(paste0("  website exhaustive identity audit: wrote %s ",
                            "accepted edges, %s component nodes, and %s ",
                            "residual cross-source same-surname pairs"),
                     format(nrow(edge_audit), big.mark = ","),
                     format(nrow(component_nodes), big.mark = ","),
                     format(nrow(rejected), big.mark = ",")))
  }

  lookup <- nodes[, .(website_context, candidate, candidate_canonical_new)]
  raw[lookup, on = .(website_context, candidate),
      candidate_canonical := i.candidate_canonical_new]
  if (any(valid & (is.na(raw$candidate_canonical) |
                   !nzchar(raw$candidate_canonical))))
    stop("Website contextual identity reconciliation produced missing keys.")
  raw[, website_context := NULL]
  raw
}

canonicalize_website_candidates <- function(raw) {
  aliases <- derive_website_aliases(raw)
  alias_lut <- setNames(aliases$candidate_p, aliases$candidate_g)
  raw[, linked_primary := unname(alias_lut[as.character(candidate)])]
  raw[, candidate_canonical := fifelse(
    data_source == "primary",
    paste0("primary::", key_part(candidate)),
    fifelse(data_source == "general_wayback" & !is.na(linked_primary),
            paste0("primary::", linked_primary),
            paste0(key_part(data_source), "::", key_part(candidate))))]
  missing_candidate <- is.na(raw$candidate) | !nzchar(trimws(raw$candidate))
  if (any(missing_candidate)) {
    raw[missing_candidate, candidate_canonical :=
          paste0(key_part(data_source), "::raw_", raw_idx)]
  }
  raw <- reconcile_website_candidate_components(raw)

  ex <- unique(raw[tolower(candidate) %in% c("turnham_joe", "turnham, joe"),
                   .(candidate, candidate_canonical)])
  if (nrow(ex) >= 2L && uniqueN(ex$candidate_canonical) != 1L)
    stop("Website canonicalization failed required turnham_joe ↔ Turnham, Joe linkage.")

  native_n <- uniqueN(raw, by = c("candidate", "state", "district", "year", "stage"))
  canon_n <- uniqueN(raw, by = c("candidate_canonical", "state", "district", "year", "stage"))
  both_full <- raw[, .(n_data_sources = uniqueN(data_source)),
                   by = .(candidate_canonical, state, district, year, stage)][n_data_sources > 1L, .N]
  both_simple <- raw[, .(n_data_sources = uniqueN(data_source)),
                     by = .(candidate_canonical, year, stage)][n_data_sources > 1L, .N]
  log_info(sprintf(paste0("  website canonicalization (full classified corpus): ",
                          "%s aliases; native cells=%s, canonical cells=%s; ",
                          "both-source cells=%s (candidate-year-stage=%s)"),
                   format(nrow(aliases), big.mark = ","),
                   format(native_n, big.mark = ","),
                   format(canon_n, big.mark = ","),
                   format(both_full, big.mark = ","),
                   format(both_simple, big.mark = ",")))
  raw[, linked_primary := NULL]
  raw
}

load_us_website_cells <- function(labeled_only = TRUE) {
  log_info("Loading U.S. website politician-election-stage cells...")
  raw_path <- file.path(DATA_DIR, "processed", "us_websites_classified.csv")
  raw <- fread(raw_path, select = c("candidate", "state", "district", "year",
                                    "data_source", "stage", "party"))
  raw[, raw_idx := .I - 1L]
  # Source-data correction approved 2026-08-13: Congress.gov and the House
  # historian identify Bill Sali (ID-1) as Republican.  The primary collection
  # has exactly 16 `sali_bill` rows mislabeled Democrat, while the same
  # billsali.com candidate is Republican in general_wayback.
  sali <- raw[
    candidate == "sali_bill" & state == "idaho" & district == 1L &
      year == 2006L & data_source == "primary"
  ]
  if (nrow(sali) != 16L || any(sali$party != "democrat") ||
      uniqueN(sali$stage) != 1L || sali$stage[1L] != 1L) {
    stop("Unexpected Bill Sali source rows; refusing the approved label override.")
  }
  raw[
    candidate == "sali_bill" & state == "idaho" & district == 1L &
      year == 2006L & data_source == "primary",
    party := "republican"
  ]
  log_info("  corrected 16 primary-collection Bill Sali rows: Democrat → Republican")
  raw <- canonicalize_website_candidates(raw)
  party_map <- c("democrat" = 0L, "republican" = 1L)
  raw[, label := party_map[party]]
  raw[, family := as.character(party)]

  clean <- read_clean("us_websites_clean.csv",
                      cols = c("raw_idx", "year", "doc_clean"))
  setnames(clean, "year", "clean_year")
  dt <- merge(clean, raw, by = "raw_idx")
  mismatch <- dt[!is.na(clean_year) & !is.na(year) & as.integer(clean_year) != as.integer(year), .N]
  if (mismatch > 0L)
    stop(sprintf("Website clean/classified year mismatch in %s rows.",
                 format(mismatch, big.mark = ",")))
  dt[, election_year := as.integer(year)]
  dt[, election_stage := fifelse(stage == 1L, "primary",
                                 fifelse(stage == 2L, "general", NA_character_))]
  if (anyNA(dt$election_year) || anyNA(dt$election_stage))
    stop("Website aggregation metadata contain missing year/stage values.")
  dt[, source := "website"]

  # Native slugs and display names are not globally person-unique.  In the
  # current corpus, for example, Jeff Miller labels occur for FL-1 and NC-11
  # in 2010, and Dave Thomas labels occur for CO-7 and UT-1 in 2002.  The
  # contextual canonicalizer above reconciles aliases only inside a campaign;
  # retain that seat/party context in the model entity unconditionally so a
  # future same-name candidate can never collide with it.
  dt[, seat_key := make_cell_key(state, district)]
  identity_audit <- dt[, .(n_seat = uniqueN(seat_key)),
                       by = .(candidate_canonical, election_year)][n_seat > 1L]
  dt[, entity := make_cell_key(candidate_canonical, state, district, party)]
  log_info(sprintf(paste0("  website identity audit: %s candidate-election labels ",
                          "are reused across seats and remain context-separated"),
                   format(nrow(identity_audit), big.mark = ",")))
  dt[, entity_scope := "us_website_candidate"]
  dt[, cell_key := make_cell_key(source, entity, election_year, election_stage)]

  # Every resulting politician-election-stage cell must now contain exactly
  # one seat.  Keep this guard so a future source revision cannot silently pool
  # two different candidates.
  seat_audit <- dt[, .(n_seat = uniqueN(seat_key)),
                   by = cell_key][n_seat > 1L]
  if (nrow(seat_audit) > 0L)
    stop("Website politician keys collide across seats in ", nrow(seat_audit),
         " election-stage cells.")
  dt[, seat_key := NULL]

  model_native <- uniqueN(dt, by = c("candidate", "election_year", "election_stage"))
  model_canon <- uniqueN(dt, by = c("entity", "election_year",
                                    "election_stage"))
  model_both <- dt[, .(n_data_sources = uniqueN(data_source)),
                   by = .(entity, election_year,
                          election_stage)][n_data_sources > 1L, .N]
  log_info(sprintf(paste0("  website canonicalization (labeled clean universe): ",
                          "native cells=%s, canonical cells=%s, both-source cells=%s"),
                   format(model_native, big.mark = ","),
                   format(model_canon, big.mark = ","),
                   format(model_both, big.mark = ",")))
  dt <- dt[, .(raw_idx, source, election_year, election_stage, family, label,
               entity, entity_scope, cell_key, candidate, candidate_canonical, state,
               district, stage, data_source, doc_clean)]
  finish_cell_rows(dt, labeled_only)
}

load_us_tv_ad_cells <- function(labeled_only = TRUE) {
  log_info("Loading U.S. TV-ad politician-election-stage cells...")
  clean <- read_clean("us_tv_ads_clean.csv",
                      cols = c("raw_idx", "id", "race", "party", "state2",
                               "district2", "year", "primary",
                               "sponsor_or_positive_id",
                               "sponsor_or_positive_name", "doc_clean"))
  clean[, party := tolower(trimws(as.character(party)))]
  party_map <- c("democrat" = 0L, "republican" = 1L)
  clean[, label := party_map[party]]
  clean[, family := as.character(party)]
  clean <- clean[!is.na(year)]
  input_universe <- nrow(clean)
  clean[, candidate_id := trimws(as.character(sponsor_or_positive_id))]
  clean[, candidate_name_key := normalize_candidate_key(sponsor_or_positive_name)]
  valid_id <- !is.na(clean$candidate_id) & nzchar(clean$candidate_id)
  valid_name <- !is.na(clean$candidate_name_key) & nzchar(clean$candidate_name_key)
  attributable <- valid_id | valid_name
  log_info(sprintf(paste0("  tv_ad attribution filter: excluded %s/%s rows ",
                          "without a sponsoring/benefiting politician"),
                   format(sum(!attributable), big.mark = ","),
                   format(input_universe, big.mark = ",")))
  clean <- clean[attributable]

  clean[, election_year := as.integer(year)]
  primary_flag <- tolower(trimws(as.character(clean$primary)))
  if (any(!primary_flag %in% c("true", "false", "1", "0")))
    stop("TV-ad primary field contains values other than true/false.")
  clean[, election_stage := fifelse(primary_flag %in% c("true", "1"),
                                    "primary", "general")]

  # Candidate IDs are not globally name-stable: six IDs use harmless name
  # variants in different elections (for example, Bill versus William).  They
  # are unambiguous within an election, which is also the scope of a GST cell.
  id_names <- clean[
    !is.na(candidate_id) & nzchar(candidate_id) &
      !is.na(candidate_name_key) & nzchar(candidate_name_key),
    .(n_name = uniqueN(candidate_name_key),
      candidate_name_from_id = candidate_name_key[1L]),
    by = .(candidate_id, election_year)
  ]
  ambiguous_ids <- id_names[n_name > 1L]
  if (nrow(ambiguous_ids) > 0L)
    stop("TV-ad candidate ID-year keys map to multiple politician names in ",
         nrow(ambiguous_ids), " cases.")

  # Fill a missing display name only from the same candidate ID and election.
  # Rows without a same-election name retain their source ID as the entity.
  name_lut <- id_names[, .(candidate_id, election_year,
                           candidate_name_from_id)]
  clean[name_lut, on = .(candidate_id, election_year),
        candidate_name_from_id := i.candidate_name_from_id]
  missing_name <- is.na(clean$candidate_name_key) |
    !nzchar(clean$candidate_name_key)
  can_impute_name <- missing_name & !is.na(clean$candidate_name_from_id) &
    nzchar(clean$candidate_name_from_id)
  clean[can_impute_name,
        candidate_name_key := candidate_name_from_id]
  log_info(sprintf("  tv_ad name fallback: resolved %s/%s missing-name rows within election",
                   format(sum(can_impute_name), big.mark = ","),
                   format(sum(missing_name), big.mark = ",")))
  clean[, candidate_name_from_id := NULL]

  # The source has 71 name-year-stage keys with two internal candidate IDs.
  # A local audit found that every pair has the exact same displayed name,
  # party, office, state, district, and stage.  Enforce those conditions before
  # coalescing IDs by name so a same-name candidate in another race cannot be
  # silently merged in a future input revision.
  clean[, `:=`(
    candidate_name_raw_key = tolower(trimws(as.character(
      sponsor_or_positive_name))),
    race_key = tolower(trimws(as.character(race))),
    state_key = trimws(as.character(state2)),
    district_key = trimws(as.character(district2))
  )]
  multi_id_names <- clean[
    !is.na(candidate_id) & nzchar(candidate_id) &
      !is.na(candidate_name_key) & nzchar(candidate_name_key),
    .(
      n_candidate = uniqueN(candidate_id),
      n_raw_name = uniqueN(candidate_name_raw_key[
        !is.na(candidate_name_raw_key) & nzchar(candidate_name_raw_key)]),
      n_family = uniqueN(family),
      n_race = uniqueN(race_key),
      n_state = uniqueN(state_key),
      n_district = uniqueN(district_key),
      incomplete_context = any(
        is.na(family) | !nzchar(family) |
          is.na(race_key) | !nzchar(race_key) |
          is.na(state_key) | !nzchar(state_key) |
          is.na(district_key) | !nzchar(district_key)
      )
    ),
    by = .(candidate_name_key, election_year, election_stage)
  ][n_candidate > 1L]
  unsafe_name_merges <- multi_id_names[
    n_raw_name != 1L | n_family != 1L | n_race != 1L | n_state != 1L |
      n_district != 1L | incomplete_context
  ]
  if (nrow(unsafe_name_merges) > 0L)
    stop("TV-ad normalized name-year-stage keys would merge candidate IDs ",
         "across different raw names or campaign contexts in ",
         nrow(unsafe_name_merges), " cases.")
  log_info(sprintf(paste0("  tv_ad identity audit: safely coalescing %s ",
                          "name-year-stage keys with multiple source IDs"),
                   format(nrow(multi_id_names), big.mark = ",")))

  clean[, entity_base := fifelse(
    !is.na(candidate_name_key) & nzchar(candidate_name_key),
    paste0("name::", candidate_name_key), paste0("id::", candidate_id)
  )]
  reused_context <- clean[, .(
    n_context = uniqueN(make_cell_key(race_key, state_key, district_key, family))
  ), by = .(entity_base, election_year, election_stage)][n_context > 1L]
  log_info(sprintf(paste0("  tv_ad campaign-context identity: disambiguating %s ",
                          "name/ID-year-stage keys reused across offices or seats"),
                   format(nrow(reused_context), big.mark = ",")))
  clean[, entity := make_cell_key(
    entity_base, race_key, state_key, district_key, family
  )]

  # Multiple source rows can be the same creative carried in multiple market
  # records.  They have identical text/candidate/party/year metadata and must
  # not multiply that creative's word mass inside the politician-election-stage
  # cell.
  creative_audit <- clean[, .(
    n_candidate = uniqueN(candidate_id), n_election = uniqueN(election_year),
    n_stage = uniqueN(election_stage), n_family = uniqueN(family),
    n_text = uniqueN(doc_clean)
  ), by = id]
  bad_creative <- creative_audit[
    n_candidate != 1L | n_election != 1L | n_stage != 1L |
      n_family != 1L | n_text != 1L]
  if (nrow(bad_creative) > 0L)
    stop(sprintf("TV-ad creative IDs have %s mixed-metadata records.",
                 format(nrow(bad_creative), big.mark = ",")))
  before_dedup <- nrow(clean)
  setorder(clean, raw_idx)
  clean <- clean[!duplicated(id)]
  log_info(sprintf("  tv_ad creative deduplication: %s rows -> %s unique ads",
                   format(before_dedup, big.mark = ","),
                   format(nrow(clean), big.mark = ",")))
  clean[, `:=`(
    source = "tv_ad",
    entity_scope = "us_tv_ad_candidate"
  )]
  clean[, cell_key := make_cell_key(source, entity, election_year,
                                    election_stage)]
  tv_context_audit <- clean[, .(
    n_race = uniqueN(race_key), n_state = uniqueN(state_key),
    n_district = uniqueN(district_key)
  ), by = cell_key][n_race != 1L | n_state != 1L | n_district != 1L]
  if (nrow(tv_context_audit) > 0L)
    stop("TV-ad politician-election-stage cells span multiple campaign contexts in ",
         nrow(tv_context_audit), " cases.")
  dt <- clean[, .(raw_idx, source, election_year, election_stage, family,
                  label, entity, entity_scope, cell_key,
                  candidate_id, candidate_name = sponsor_or_positive_name,
                  race = race_key, state2 = state_key, district2 = district_key,
                  doc_clean)]
  finish_cell_rows(dt, labeled_only)
}

normalize_fr_candidate_id <- function(x) {
  raw <- trimws(as.character(x))
  numeric_id <- suppressWarnings(as.numeric(raw))
  normalized <- ifelse(!is.na(numeric_id) & numeric_id == floor(numeric_id),
                       format(numeric_id, scientific = FALSE, trim = TRUE), raw)
  normalized[is.na(x) | !nzchar(raw)] <- NA_character_
  normalized
}

load_fr_term_bridge <- function() {
  keys <- c("name_MP_db", "start_term_MP_db", "end_term_MP_db")
  bridge_col_classes <- c(
    name_MP_db = "character",
    start_term_MP_db = "character",
    end_term_MP_db = "character",
    id_unique_cand_elec_db = "character"
  )
  paths <- file.path(DATA_DIR, "french_parliament",
                     c("1958_2002_with_id.csv", "2002_2024_with_id.csv"))
  parts <- lapply(paths, function(path) {
    if (!file.exists(path)) stop("Missing French MP-term bridge: ", path)
    fread(path, select = c(keys, "id_unique_cand_elec_db"),
          colClasses = bridge_col_classes)
  })
  bridge <- unique(rbindlist(parts, use.names = TRUE))
  bridge[, id_unique_cand := normalize_fr_candidate_id(id_unique_cand_elec_db)]
  bridge <- bridge[!is.na(id_unique_cand)]
  ambiguity <- bridge[, .(n_id = uniqueN(id_unique_cand)), by = keys][n_id > 1L]
  if (nrow(ambiguity) > 0L)
    stop("French MP-term bridge maps ", nrow(ambiguity),
         " terms to multiple candidacies.")
  bridge <- unique(bridge[, c(keys, "id_unique_cand"), with = FALSE], by = keys)

  professions_path <- file.path(DATA_DIR, "professions_de_foi",
                                "professions_de_foi.csv")
  elections <- fread(professions_path, select = c("id_unique_cand", "year"),
                     colClasses = c(id_unique_cand = "character"))
  elections[, id_unique_cand := normalize_fr_candidate_id(id_unique_cand)]
  elections[, election_year := as.integer(year)]
  election_ambiguity <- elections[!is.na(id_unique_cand) & !is.na(election_year),
                                  .(n_year = uniqueN(election_year)),
                                  by = id_unique_cand][n_year > 1L]
  if (nrow(election_ambiguity) > 0L)
    stop("French candidacy identifiers map to multiple election years.")
  elections <- unique(elections[!is.na(id_unique_cand) & !is.na(election_year),
                                .(id_unique_cand, election_year)],
                      by = "id_unique_cand")
  bridge <- merge(bridge, elections, by = "id_unique_cand", all.x = TRUE)
  log_info(sprintf("  French bridge: %s MP terms linked to candidacies",
                   format(nrow(bridge), big.mark = ",")))
  bridge
}

load_fr_parliament_cells <- function(labeled_only = TRUE) {
  log_info("Loading French parliamentary politician-election cells...")
  raw_path <- file.path(DATA_DIR, "french_parliament", "collapsed_1958_2024.csv")
  header <- names(fread(raw_path, nrows = 0L))
  needed <- c("group_MP_db", "name_MP_db", "start_term_MP_db", "end_term_MP_db")
  if (!all(needed %in% header)) {
    stop("collapsed_1958_2024.csv lacks MP-term columns; keyed *_with_id alignment is required.")
  }
  select <- needed
  if ("year_start_term_MP_db" %in% header) select <- c(select, "year_start_term_MP_db")
  raw <- fread(
    raw_path,
    select = select,
    colClasses = c(
      name_MP_db = "character",
      start_term_MP_db = "character",
      end_term_MP_db = "character"
    )
  )
  raw[, raw_idx := .I - 1L]
  raw[, label := FR_GROUP_TO_LABEL[group_MP_db]]
  raw[, family := as.character(group_MP_db)]
  # `year_start_term_MP_db` exists but is only partially populated; derive the
  # required election year directly from the authoritative start-term date.
  raw[, term_start_year := as.integer(substr(as.character(start_term_MP_db), 1L, 4L))]
  if ("year_start_term_MP_db" %in% names(raw)) {
    supplied_start_year <- suppressWarnings(as.integer(raw$year_start_term_MP_db))
    mismatch <- !is.na(supplied_start_year) & !is.na(raw$term_start_year) &
      supplied_start_year != raw$term_start_year
    if (any(mismatch))
      stop(sprintf("FR parliament supplied/derived start-term year mismatch in %s rows.",
                   format(sum(mismatch), big.mark = ",")))
  }

  clean <- read_clean("fr_parliament_clean.csv",
                      cols = c("raw_idx", "date", "doc_clean"))
  clean[, speech_year := as.integer(substr(date, 1L, 4L))]
  clean <- clean[!is.na(speech_year)]
  clean[, c("date", "speech_year") := NULL]
  dt <- merge(clean, raw, by = "raw_idx")
  bridge <- load_fr_term_bridge()
  term_keys <- c("name_MP_db", "start_term_MP_db", "end_term_MP_db")
  dt <- merge(dt, bridge, by = term_keys, all.x = TRUE, sort = FALSE)
  dt[, election_year := fifelse(!is.na(election_year), election_year,
                                term_start_year)]
  has_name <- !is.na(dt$name_MP_db) & nzchar(trimws(dt$name_MP_db))
  has_start <- !is.na(dt$start_term_MP_db) &
    nzchar(trimws(dt$start_term_MP_db))
  has_end <- !is.na(dt$end_term_MP_db) & nzchar(trimws(dt$end_term_MP_db))
  has_election <- !is.na(dt$election_year)
  valid_meta <- has_name & has_start & has_end & has_election
  if (any(!valid_meta)) {
    log_info(sprintf(paste0("  parliament politician filter: excluded %s cleaned rows ",
                            "with incomplete MP-term metadata ",
                            "(name=%s, start=%s, end=%s, election=%s)"),
                     format(sum(!valid_meta), big.mark = ","),
                     format(sum(!has_name), big.mark = ","),
                     format(sum(!has_start), big.mark = ","),
                     format(sum(!has_end), big.mark = ","),
                     format(sum(!has_election), big.mark = ",")))
    dt <- dt[valid_meta]
  }
  if (nrow(dt) == 0L)
    stop("No French parliamentary rows have complete MP-term metadata.")
  dt[, term_key := make_cell_key(name_MP_db, start_term_MP_db, end_term_MP_db)]
  dt[, entity := fifelse(!is.na(id_unique_cand), id_unique_cand, term_key)]
  dt[, `:=`(source = "parliament",
            entity_scope = fifelse(!is.na(id_unique_cand), "fr_candidate",
                                   "fr_parliamentarian"),
            election_stage = "office")]
  dt[, cell_key := make_cell_key(source, entity, election_year)]
  dt <- dt[, .(raw_idx, source, election_year, election_stage, family, label,
               entity, entity_scope, cell_key, name_MP_db, start_term_MP_db,
               end_term_MP_db, id_unique_cand, doc_clean)]
  finish_cell_rows(dt, labeled_only, "L", "R")
}

load_fr_profession_cells <- function(labeled_only = TRUE) {
  log_info("Loading French professions de foi (one manifesto per cell)...")
  raw_path <- file.path(DATA_DIR, "professions_de_foi", "professions_de_foi.csv")
  raw <- fread(raw_path, select = c("id_unique_cand", "year", "tour", "nuance",
                                    "orientation", "falc", "data_source"),
               colClasses = c(id_unique_cand = "character"))
  raw[, raw_idx := .I - 1L]
  raw[, falc := as.integer(falc)]
  if (anyNA(raw$falc) || !all(raw$falc %in% c(0L, 1L)))
    stop("Profession de foi falc must be coded 0/1 on every source row.")
  n_falc <- raw[falc == 1L, .N]
  raw <- raw[falc == 0L]
  log_info(sprintf("  professions: excluded %s FALC alternative versions",
                   format(n_falc, big.mark = ",")))
  # Legislative manifestos only.  The source file also holds cantonal
  # (1979-2011) and departementale (2015, 2021) manifestos -- data_source
  # v2_cantonale / v2_departementale / v3_departementale -- which are excluded
  # from every French analysis and from the training corpus (decision
  # 2026-08-26; same rule as Codes/R/00_fr_party_families.R).
  n_local <- raw[!data_source %in% FR_LEGISLATIVE_MANIFESTO_SOURCES, .N]
  raw <- raw[data_source %in% FR_LEGISLATIVE_MANIFESTO_SOURCES]
  log_info(sprintf("  professions: excluded %s cantonal/departementale manifestos (legislative sources: %s)",
                   format(n_local, big.mark = ","),
                   paste(FR_LEGISLATIVE_MANIFESTO_SOURCES, collapse = ", ")))
  nuance_map <- c("left" = 0L, "right" = 1L)
  raw[, label := nuance_map[nuance]]
  raw[, family := as.character(orientation)]
  clean <- read_clean("fr_professions_clean.csv",
                      cols = c("raw_idx", "year", "doc_clean"))
  setnames(clean, "year", "clean_year")
  dt <- merge(clean, raw, by = "raw_idx")
  mismatch <- dt[!is.na(clean_year) & !is.na(year) &
                   as.integer(clean_year) != as.integer(year), .N]
  if (mismatch > 0L)
    stop("Profession clean/raw election-year mismatch in ", mismatch, " rows.")
  dt[, election_year := as.integer(year)]
  dt[, election_stage := fifelse(tour == 1L, "round1",
                                 fifelse(tour == 2L, "round2", NA_character_))]
  if (anyNA(dt$election_stage))
    stop("Profession de foi contains a tour other than 1 or 2.")
  dt[, candidate_id := normalize_fr_candidate_id(id_unique_cand)]
  dt[is.na(candidate_id) | !nzchar(candidate_id),
     candidate_id := paste0("raw_", raw_idx)]
  dt[, `:=`(source = "professions_de_foi", entity = candidate_id,
            entity_scope = "fr_candidate")]
  dt[, cell_key := make_cell_key(source, raw_idx)]
  dt <- dt[, .(raw_idx, source, election_year, election_stage, family, label,
               entity, entity_scope, cell_key, id_unique_cand = candidate_id,
               tour = as.integer(tour), doc_clean)]
  finish_cell_rows(dt, labeled_only, "L", "R")
}

aggregate_source_counts <- function(dt, vocab, labeled_only) {
  sources <- unique(dt$source)
  if (length(sources) != 1L) stop("aggregate_source_counts expects one source.")
  src <- sources[1L]
  universe <- nrow(dt)

  audit <- dt[, .(
    n_source = unique_value_count(source),
    n_election = unique_value_count(election_year),
    n_stage = unique_value_count(election_stage),
    n_label = uniqueN(label[!is.na(label)]),
    n_entity = unique_value_count(entity),
    n_scope = unique_value_count(entity_scope)
  ), by = cell_key]
  bad <- audit[n_source != 1L | n_election != 1L | n_stage != 1L |
               n_entity != 1L | n_scope != 1L |
               (labeled_only & n_label != 1L)]
  if (nrow(bad) > 0L) {
    stop(sprintf("%s has %s aggregation cells with mixed metadata/labels.",
                 src, format(nrow(bad), big.mark = ",")))
  }
  rm(audit, bad)

  setorder(dt, cell_key, raw_idx)
  meta <- dt[, .(
    source = source[1L], election_year = election_year[1L],
    election_stage = election_stage[1L],
    family = {
      z <- sort(unique(family[!is.na(family) & nzchar(trimws(family))]))
      if (length(z) == 0L) NA_character_ else paste(z, collapse = "|")
    },
    label = if (all(is.na(label))) NA_integer_ else label[!is.na(label)][1L],
    entity = entity[1L], entity_scope = entity_scope[1L],
    first_raw_idx = min(raw_idx), n_docs_raw = .N
  ), by = cell_key]
  if (src == "website") {
    extra <- dt[, .(
      candidate_canonical = candidate_canonical[1L], state = state[1L],
      district = district[1L], stage = stage[1L],
      candidate_primary_aliases = {
        z <- sort(unique(candidate[data_source == "primary" &
                                     !is.na(candidate) & nzchar(trimws(candidate))]))
        if (length(z) == 0L) NA_character_ else paste(z, collapse = "\u241e")
      },
      candidate_general_aliases = {
        z <- sort(unique(candidate[data_source == "general_wayback" &
                                     !is.na(candidate) & nzchar(trimws(candidate))]))
        if (length(z) == 0L) NA_character_ else paste(z, collapse = "\u241e")
      },
      n_data_sources_raw = uniqueN(data_source)
    ), by = cell_key]
    meta <- merge(meta, extra, by = "cell_key", sort = FALSE)
    setorder(meta, cell_key)
  } else if (src == "tv_ad") {
    extra <- dt[, .(
      candidate_name = {
        z <- sort(unique(candidate_name[!is.na(candidate_name) &
                                            nzchar(trimws(candidate_name))]))
        if (length(z) == 0L) NA_character_ else paste(z, collapse = "|")
      },
      race = race[1L], state2 = state2[1L], district2 = district2[1L]
    ), by = cell_key]
    meta <- merge(meta, extra, by = "cell_key", sort = FALSE)
    setorder(meta, cell_key)
  } else if (src == "parliament") {
    extra <- dt[, .(name_MP_db = name_MP_db[1L],
                    start_term_MP_db = start_term_MP_db[1L],
                    end_term_MP_db = end_term_MP_db[1L],
                    id_unique_cand = {
                      z <- unique(id_unique_cand[!is.na(id_unique_cand)])
                      if (length(z) == 0L) NA_character_ else z[1L]
                    }), by = cell_key]
    meta <- merge(meta, extra, by = "cell_key", sort = FALSE)
    setorder(meta, cell_key)
  } else if (src == "professions_de_foi") {
    extra <- dt[, .(id_unique_cand = id_unique_cand[1L], tour = tour[1L]),
                by = cell_key]
    meta <- merge(meta, extra, by = "cell_key", sort = FALSE)
    setorder(meta, cell_key)
  }
  meta[, doc_id := sprintf("%s_cell_%06d", src, seq_len(.N))]
  meta[, `:=`(
    start_row = cumsum(shift(n_docs_raw, fill = 0L)) + 1L,
    end_row = cumsum(n_docs_raw)
  )]
  meta[, batch := (start_row - 1L) %/% AGG_CHUNK_DOCS]

  C_parts <- vector("list", uniqueN(meta$batch))
  n_docs <- integer(nrow(meta))
  n_data_sources_positive <- integer(nrow(meta))
  batches <- sort(unique(meta$batch))
  log_info(sprintf("Aggregating %s: %s docs → %s raw cells in %d no-sampling batches",
                   src, format(universe, big.mark = ","),
                   format(nrow(meta), big.mark = ","), length(batches)))
  for (bb in seq_along(batches)) {
    b <- batches[bb]
    mi <- which(meta$batch == b)
    lo <- min(meta$start_row[mi]); hi <- max(meta$end_row[mi])
    sub <- dt[lo:hi]
    vt <- build_counts(sub$doc_clean, vocab = vocab)
    C_doc <- vt$C
    positive <- rowSums(C_doc) > 0
    local_cell <- match(sub$cell_key, meta$cell_key[mi])
    A <- sparseMatrix(i = local_cell, j = seq_len(nrow(sub)), x = 1,
                      dims = c(length(mi), nrow(sub)))
    C_part <- A %*% C_doc
    rownames(C_part) <- meta$doc_id[mi]
    colnames(C_part) <- vocab
    C_parts[[bb]] <- C_part
    n_docs[mi] <- tabulate(local_cell[positive], nbins = length(mi))
    if (src == "website" && any(positive)) {
      ds <- unique(data.table(local_cell = local_cell[positive],
                              data_source = sub$data_source[positive]))
      n_data_sources_positive[mi] <- tabulate(ds$local_cell, nbins = length(mi))
    }
    rm(sub, vt, C_doc, A, C_part); gc(verbose = FALSE)
    log_info(sprintf("  %s batch %d/%d: rows %s-%s, cells %s",
                     src, bb, length(batches), format(lo, big.mark = ","),
                     format(hi, big.mark = ","), format(length(mi), big.mark = ",")))
  }
  C <- do.call(rbind, C_parts)
  rm(C_parts, dt); gc(verbose = FALSE, full = TRUE)
  if (!identical(rownames(C), meta$doc_id))
    stop(src, " aggregated count rows are misaligned with cell metadata.")

  meta[, n_docs := n_docs]
  if (src == "website") meta[, n_data_sources := n_data_sources_positive]
  tokens <- as.numeric(rowSums(C))
  meta[, n_tokens := tokens]
  zero_docs <- sum(meta$n_docs_raw - meta$n_docs)
  zero_cells <- sum(tokens == 0)
  keep <- tokens > 0
  kept_docs <- sum(meta$n_docs[keep])
  if (universe != zero_docs + kept_docs) {
    stop(sprintf("%s document conservation failed: %d != %d + %d",
                 src, universe, zero_docs, kept_docs))
  }
  log_info(sprintf(paste0("CONSERVATION %s: loaded documents=%s = ",
                          "zero-vocabulary documents %s + documents in kept cells %s"),
                   src, format(universe, big.mark = ","),
                   format(zero_docs, big.mark = ","),
                   format(kept_docs, big.mark = ",")))
  log_info(sprintf("CELL FILTER %s: raw=%s; zero-token=%s; kept=%s",
                   src, format(nrow(meta), big.mark = ","),
                   format(zero_cells, big.mark = ","),
                   format(sum(keep), big.mark = ",")))
  if (!any(keep))
    stop(src, " has no cells with any canonical-vocabulary phrase counts.")

  C <- C[keep, , drop = FALSE]
  meta <- meta[keep]
  # GST's multinomial-Poisson model uses each cell's actual phrase count as
  # its verbosity/exposure: mu_i = log(m_i), m_i = sum_j c_ij.  Retain the
  # integer row-summed counts here; a common-mass rescaling would change the
  # penalized likelihood and is not part of the reference implementation.
  rownames(C) <- meta$doc_id
  colnames(C) <- vocab
  if (any(rowSums(C) <= 0)) stop(src, " retained a zero-vocabulary cell.")
  if (src == "website") {
    both_raw <- meta[n_data_sources_raw > 1L, .N]
    both_positive <- meta[n_data_sources > 1L, .N]
    log_info(sprintf(paste0("WEBSITE KEPT CELLS: %s total; %s contain both data sources ",
                            "by raw snapshots (%s by positive-vocab snapshots)"),
                     format(nrow(meta), big.mark = ","),
                     format(both_raw, big.mark = ","),
                     format(both_positive, big.mark = ",")))
  }
  meta[, c("start_row", "end_row", "batch") := NULL]
  summary <- data.table(
    source = src, loaded_documents = universe, raw_cells = length(tokens),
    zero_vocab_docs = zero_docs, zero_token_cells = zero_cells,
    kept_cells = nrow(meta), kept_docs = kept_docs
  )
  list(C = C, meta = meta, summary = summary)
}

aggregate_country_counts <- function(country, vocab, labeled_only = TRUE) {
  loaders <- if (country == "us") {
    list(load_us_congress_cells, load_us_tweet_cells,
         load_us_tv_ad_cells, load_us_website_cells)
  } else if (country == "fr") {
    list(load_fr_parliament_cells, load_fr_profession_cells)
  } else {
    stop("Unknown country: ", country)
  }
  packs <- vector("list", length(loaders))
  for (i in seq_along(loaders)) {
    dt <- loaders[[i]](labeled_only = labeled_only)
    log_mem(sprintf("after %s aggregation load", unique(dt$source)))
    packs[[i]] <- aggregate_source_counts(dt, vocab, labeled_only)
    rm(dt); gc(verbose = FALSE, full = TRUE)
    log_mem(sprintf("after source aggregation %d/%d", i, length(loaders)))
  }
  C <- do.call(rbind, lapply(packs, `[[`, "C"))
  meta <- rbindlist(lapply(packs, `[[`, "meta"), use.names = TRUE, fill = TRUE)
  summary <- rbindlist(lapply(packs, `[[`, "summary"), use.names = TRUE)
  if (!identical(rownames(C), meta$doc_id))
    stop("Country-level aggregated counts are misaligned with metadata.")
  log_info("Final post-filter cell counts by source:")
  for (i in seq_len(nrow(summary))) {
    log_info(sprintf("  %s: %s", summary$source[i],
                     format(summary$kept_cells[i], big.mark = ",")))
  }
  list(C = C, meta = meta, summary = summary)
}

# ============================================================================
# Build sparse phrase-count matrix from doc_clean
# ============================================================================

build_counts <- function(docs, vocab, token_min_len = TOKEN_MIN_LEN) {
  # Tokenize like sklearn CountVectorizer(token_pattern=r"(?u)\b\w{3,}\b"):
  # lowercase, split on non-word runs, keep tokens >= token_min_len chars.
  # (*UCP) makes \w Unicode-aware so accented French words stay whole
  # ("égalité"), matching Python's (?u)\w; without it PCRE \w is ASCII-only and
  # shreds accented tokens.
  tok_list <- strsplit(tolower(docs), "(*UCP)\\W+", perl = TRUE)
  tok_list <- lapply(tok_list, function(toks) toks[nchar(toks) >= token_min_len])

  # Curated vocabulary (1,2)-gram, greedy NON-OVERLAPPING matching.
  # A vocab BIGRAM consumes its two component tokens: walking left to right, if
  # the current adjacent pair is a vocab bigram, emit the bigram and advance by
  # two — so "national security" counts once as the bigram and is NOT also
  # tallied as the unigrams "national" + "security".
  # NO stopword removal: the frozen vocab already excludes stopword unigrams,
  # and its bigrams were built from ADJACENT raw tokens, so the stream here must
  # preserve that adjacency (do NOT delete tokens between).
  # --------------------------------------------------------------------------
  is_bi    <- grepl(" ", vocab, fixed = TRUE)
  bi_vocab <- vocab[is_bi]
  bi_idx   <- setNames(seq_along(bi_vocab), bi_vocab)   # hashed membership test
  term_to_col <- setNames(seq_along(vocab), vocab)
  V <- length(vocab); n_docs <- length(tok_list)

  emit_ngrams <- function(toks) {
    n <- length(toks)
    if (n == 0L) return(character(0))
    bgs   <- if (n >= 2L) paste(toks[-n], toks[-1L]) else character(0)
    in_bi <- if (n >= 2L) !is.na(bi_idx[bgs]) else logical(0)
    out <- character(n); k <- 0L; i <- 1L
    while (i <= n) {
      if (i < n && in_bi[i]) {           # adjacent pair is a vocab bigram
        k <- k + 1L; out[k] <- bgs[i]; i <- i + 2L
      } else {                            # single token (filtered by vocab below)
        k <- k + 1L; out[k] <- toks[i]; i <- i + 1L
      }
    }
    out[seq_len(k)]
  }

  log_info(sprintf("Frozen vocab: %s tokens (unigrams=%s, bigrams=%s)",
                   format(length(vocab),         big.mark = ","),
                   format(sum(!is_bi),           big.mark = ","),
                   format(sum(is_bi),            big.mark = ",")))
  # Per-document column indices via greedy n-gram emission. The tokenizer is the
  # predict-time bottleneck (scoring is just a matmul), so we PARALLELIZE it — but
  # over a PSOCK cluster, NOT parallel::mclapply. On Linux mclapply FORKS, and R's
  # GC mark phase writes to every object header, defeating copy-on-write: each
  # fork duplicates the parent heap, so N forks ≈ N× RSS and the box OOMs (this
  # took down WSL at 6 cores). PSOCK workers are independent processes that
  # receive ONLY their doc block over a socket — bounded, predictable memory.
  # Each worker returns per-doc in-vocab column-index vectors; duplicate
  # (doc,col) pairs are counted ONCE at the end via data.table — avoiding a
  # per-doc length-V tabulate(). Output C is identical to the serial path.
  ncores <- max(1L, min(N_CORES, n_docs))
  emit_one_serial <- function(toks) {                 # parent-side (pristine env)
    cc <- term_to_col[emit_ngrams(toks)]; cc[!is.na(cc)]
  }
  if (ncores > 1L) {
    # Re-scope a copy of emit_ngrams + the block fn to globalenv so PSOCK
    # serialization ships only the small lookups (the fn, bi_idx, term_to_col)
    # to each worker's .GlobalEnv — NEVER this frame's (huge) tok_list.
    eng <- emit_ngrams; environment(eng) <- globalenv()
    blk <- function(tok_sub) lapply(tok_sub, function(toks) {
      cc <- term_to_col[emit_ngrams(toks)]; cc[!is.na(cc)]
    })
    environment(blk) <- globalenv()
    cl <- parallel::makeCluster(ncores, type = "PSOCK")
    on.exit(parallel::stopCluster(cl), add = TRUE)
    exp <- new.env(parent = emptyenv())
    exp$emit_ngrams <- eng; exp$bi_idx <- bi_idx; exp$term_to_col <- term_to_col
    parallel::clusterExport(cl, c("emit_ngrams", "bi_idx", "term_to_col"), envir = exp)
    blocks     <- split(seq_len(n_docs), (seq_len(n_docs) - 1L) %% ncores)
    tok_blocks <- lapply(blocks, function(ix) tok_list[ix])
    res <- tryCatch(parallel::parLapply(cl, tok_blocks, blk),
                    error = function(e) { log_info(sprintf(
                      "PSOCK tokenize failed (%s); serial fallback", conditionMessage(e))); NULL })
    if (is.null(res)) {                                # cluster failure -> serial
      jj    <- lapply(tok_list, emit_one_serial)
      i_vec <- rep.int(seq_len(n_docs), lengths(jj)); j_vec <- unlist(jj, use.names = FALSE)
    } else {
      i_vec <- unlist(Map(function(ix, jl) rep.int(ix, lengths(jl)), blocks, res),
                      use.names = FALSE)
      j_vec <- unlist(res, use.names = FALSE)
    }
  } else {
    jj    <- lapply(tok_list, emit_one_serial)
    i_vec <- rep.int(seq_len(n_docs), lengths(jj)); j_vec <- unlist(jj, use.names = FALSE)
  }
  if (length(i_vec) == 0L) {
    C <- sparseMatrix(i = integer(0), j = integer(0), x = numeric(0),
                      dims = c(n_docs, V), dimnames = list(NULL, vocab))
    return(list(C = C, vocab = vocab))
  }
  agg <- data.table::data.table(i = i_vec, j = j_vec)[, .(x = .N), by = .(i, j)]
  C <- sparseMatrix(i = agg$i, j = agg$j, x = agg$x,
                    dims = c(n_docs, V), dimnames = list(NULL, vocab))
  list(C = C, vocab = vocab)
}

# Write the largest positive and negative terms in every scored politician cell.
# Each reported value is an exact additive component of the stored projection:
#   contribution_nb = count_nb * phi_b / sum_b count_nb
# The files are source-specific so rows remain sorted by cell_key and the local
# inspection server can retrieve one cell with byte-level binary search.
write_gst_phrase_contributors <- function(C, meta, vocab, phi, country,
                                          top_n = 25L, batch_size = 1000L) {
  if (nrow(C) != nrow(meta) || ncol(C) != length(vocab) ||
      length(phi) != length(vocab))
    stop("Contributor export received misaligned counts, metadata, vocab, or phi.")
  if (any(rowSums(C) != meta$n_tokens))
    stop("Contributor export token denominators differ from prediction metadata.")

  dir.create(TEMP_DIR, recursive = TRUE, showWarnings = FALSE)
  for (src in unique(meta$source)) {
    source_rows <- which(meta$source == src)
    source_keys <- meta$cell_key[source_rows]
    # meta was ordered by setorder(), i.e. in the C locale (byte order), which is
    # also the code-point order the inspection server's binary search assumes.
    # is.unsorted() compares in the session's collation locale, where glibc
    # ignores the U+241F separator at the first level and puts "1002<sep>2002"
    # before "10020<sep>2012" (predict failed on the 2026-08-25 parliament
    # cells).  chorder() reproduces setorder()'s C-locale order and, unlike
    # sort(method = "radix"), accepts strings with an unknown encoding mark.
    if (!identical(source_keys, source_keys[chorder(source_keys)]))
      stop("Contributor source cells are not sorted by cell_key: ", src)

    out_path <- file.path(
      TEMP_DIR,
      sprintf("gst_phrase_contributors_%s_%s.csv", country, src)
    )
    temporary <- paste0(out_path, ".tmp")
    if (file.exists(temporary) && !file.remove(temporary))
      stop("Could not remove stale contributor temporary file: ", temporary)
    first_write <- TRUE

    blocks <- split(source_rows,
                    (seq_along(source_rows) - 1L) %/% batch_size)
    for (block_rows in blocks) {
      block <- as(C[block_rows, , drop = FALSE], "RsparseMatrix")
      rows <- vector("list", 2L * length(block_rows))
      out_i <- 0L
      for (local_i in seq_along(block_rows)) {
        lo <- block@p[local_i] + 1L
        hi <- block@p[local_i + 1L]
        if (hi < lo) next
        cols <- block@j[lo:hi] + 1L
        counts <- block@x[lo:hi]
        global_i <- block_rows[local_i]
        contributions <- counts * phi[cols] / meta$n_tokens[global_i]

        selections <- list(
          positive = head(which(contributions > 0)[
            order(contributions[contributions > 0], decreasing = TRUE)
          ], top_n),
          negative = head(which(contributions < 0)[
            order(contributions[contributions < 0], decreasing = FALSE)
          ], top_n)
        )
        for (direction in names(selections)) {
          keep <- selections[[direction]]
          if (length(keep) == 0L) next
          out_i <- out_i + 1L
          rows[[out_i]] <- data.table(
            cell_key = meta$cell_key[global_i],
            direction = direction,
            rank = seq_along(keep),
            term = vocab[cols[keep]],
            count = as.integer(round(counts[keep])),
            phi = phi[cols[keep]],
            contribution = contributions[keep]
          )
        }
      }
      chunk <- if (out_i > 0L) {
        rbindlist(rows[seq_len(out_i)], use.names = TRUE)
      } else {
        data.table()
      }
      if (nrow(chunk) > 0L) {
        fwrite(chunk, temporary, append = !first_write,
               col.names = first_write)
        first_write <- FALSE
      }
      rm(block, rows, chunk); gc(verbose = FALSE)
    }
    if (first_write) {
      fwrite(data.table(
        cell_key = character(), direction = character(), rank = integer(),
        term = character(), count = integer(), phi = numeric(),
        contribution = numeric()
      ), temporary)
    }
    if (file.exists(out_path) && !file.remove(out_path))
      stop("Could not replace contributor output: ", out_path)
    if (!file.rename(temporary, out_path))
      stop("Could not install contributor output: ", out_path)
    log_info(sprintf("Wrote top-%d positive/negative contributors for %s to %s",
                     top_n, src, out_path))
  }
}

# ============================================================================
# Metrics + LaTeX table
# (mirrors _metrics_from_arrays / write_metrics_latex from the Python file)
# ============================================================================

SOURCE_LABELS <- c(
  All                  = "All",
  congress             = "Congress",
  tweet                = "Tweets",
  tv_ad                = "TV Ads",
  website              = "Websites",
  parliament           = "Parliament",
  professions_de_foi   = "Prof.\\ de foi"
)

auc_score <- function(y, p) {
  if (length(unique(y)) < 2L) return(NA_real_)
  o     <- order(p)
  y_ord <- y[o]
  n1    <- sum(y == 1L); n0 <- sum(y == 0L)
  if (n1 == 0L || n0 == 0L) return(NA_real_)
  rks   <- rank(p, ties.method = "average")
  (sum(rks[y == 1L]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}

metrics <- function(y, p, threshold = 0.5) {
  y    <- as.integer(y)
  pred <- as.integer(p > threshold)
  tp <- sum(pred == 1L & y == 1L)
  fp <- sum(pred == 1L & y == 0L)
  fn <- sum(pred == 0L & y == 1L)
  prec <- if (tp + fp == 0L) 0 else tp / (tp + fp)
  rec  <- if (tp + fn == 0L) 0 else tp / (tp + fn)
  f1   <- if (prec + rec == 0) 0 else 2 * prec * rec / (prec + rec)
  list(n = length(y),
       accuracy  = mean(pred == y),
       precision = prec,
       recall    = rec,
       f1        = f1,
       auc       = auc_score(y, p))
}

write_metrics_latex <- function(results, country, tex_path) {
  dir.create(dirname(tex_path), recursive = TRUE, showWarnings = FALSE)
  con <- file(tex_path, open = "w", encoding = "UTF-8")
  on.exit(close(con))
  country_label <- if (country == "fr") "French" else "US"
  writeLines("\\begin{table}[htbp]", con)
  writeLines("\\centering", con)
  writeLines(sprintf("\\caption{In-sample Party Classifier Fit (%s, GST in R)}",
                     country_label), con)
  writeLines(sprintf("\\label{tab:party_classifier_%s_gst_R}", country), con)
  writeLines("\\begin{tabular}{lrrrrrr}", con)
  writeLines("\\toprule", con)
  writeLines("Source & $N$ & Accuracy & Precision & Recall & F1 & AUC \\\\", con)
  writeLines("\\midrule", con)
  for (r in results) {
    if (r$source == "All") writeLines("\\midrule", con)
    name <- if (!is.null(SOURCE_LABELS[[r$source]])) SOURCE_LABELS[[r$source]] else r$source
    auc_str <- if (is.na(r$auc)) "--" else sprintf("%.3f", r$auc)
    writeLines(sprintf("%s & %s & %.3f & %.3f & %.3f & %.3f & %s \\\\",
                       name, format(r$n, big.mark = ","),
                       r$accuracy, r$precision, r$recall, r$f1, auc_str),
               con)
  }
  writeLines("\\bottomrule", con)
  writeLines("\\end{tabular}", con)
  writeLines("\\end{table}", con)
  log_info(sprintf("Saved LaTeX table to %s", tex_path))
}

# ============================================================================
# GST estimation block — verbatim translation of compute.r
# ============================================================================

GST_ALLOWED_STAGES <- list(
  congress = "office",
  tweet = "office",
  tv_ad = c("primary", "general"),
  website = c("primary", "general"),
  parliament = "office",
  professions_de_foi = c("round1", "round2")
)

gst_source_stage_keys <- function(meta) {
  required <- c("source", "election_stage", "election_year", "doc_id")
  missing <- setdiff(required, names(meta))
  if (length(missing) > 0L)
    stop("GST nuisance design metadata missing: ", paste(missing, collapse = ", "))

  source <- trimws(as.character(meta$source))
  stage <- trimws(as.character(meta$election_stage))
  election <- trimws(as.character(meta$election_year))
  doc_id <- trimws(as.character(meta$doc_id))
  if (anyNA(source) || anyNA(stage) || anyNA(election) || anyNA(doc_id) ||
      any(source == "") || any(stage == "") || any(election == "") ||
      any(doc_id == ""))
    stop("GST nuisance design requires non-missing source, stage, year, and doc_id.")

  unknown_sources <- setdiff(unique(source), names(GST_ALLOWED_STAGES))
  if (length(unknown_sources) > 0L)
    stop("GST nuisance design has unknown sources: ",
         paste(sort(unknown_sources), collapse = ", "))

  for (src in sort(unique(source))) {
    observed <- sort(unique(stage[source == src]))
    unexpected <- setdiff(observed, GST_ALLOWED_STAGES[[src]])
    if (length(unexpected) > 0L)
      stop(sprintf("GST nuisance design has invalid stage(s) for %s: %s",
                   src, paste(unexpected, collapse = ", ")))
  }
  paste(source, stage, sep = "::")
}

# Construct the exact same nuisance design at fit and score time. Source is
# interacted with campaign/election stage, so observed U.S. primary/general
# material and French round-1/round-2 manifestos have distinct intercepts. The
# current TV-ad corpus happens to contain only general-election ads. Election-
# year fixed effects remain additive. The party loading in R is still common.
build_gst_nuisance_matrix <- function(meta, source_stage_levels = NULL,
                                      election_levels = NULL,
                                      train_columns = NULL) {
  training <- is.null(source_stage_levels) && is.null(election_levels) &&
    is.null(train_columns)
  if (!training && (is.null(source_stage_levels) || is.null(election_levels) ||
                    is.null(train_columns)))
    stop("Scoring GST nuisance design requires training levels and columns.")

  source_stage <- gst_source_stage_keys(meta)
  election <- as.character(meta$election_year)
  if (training) {
    source_stage_levels <- sort(unique(source_stage))
    election_levels <- sort(unique(election))
    unseen <- rep(FALSE, nrow(meta))
  } else {
    source_stage_levels <- as.character(source_stage_levels)
    election_levels <- as.character(election_levels)
    unseen_source_stage <- !(source_stage %in% source_stage_levels)
    unseen_election <- !(election %in% election_levels)
    unseen <- unseen_source_stage | unseen_election
    if (any(unseen)) {
      log_info(sprintf(
        paste0("GST nuisance design: %d / %d rows have unseen source-stage ",
               "or election year; their nuisance block is set to zero."),
        sum(unseen), nrow(meta)))
      source_stage[unseen_source_stage] <- source_stage_levels[1L]
      election[unseen_election] <- election_levels[1L]
    }
  }

  design_data <- data.frame(
    source_stage_f = factor(source_stage, levels = source_stage_levels),
    election_f = factor(election, levels = election_levels),
    stringsAsFactors = FALSE
  )
  X <- sparse.model.matrix(
    ~ 0 + source_stage_f + election_f, data = design_data)
  if (training) {
    qx <- qr(as.matrix(X))
    X <- X[, qx$pivot[seq_len(qx$rank)], drop = FALSE]
  } else {
    missing_columns <- setdiff(train_columns, colnames(X))
    if (length(missing_columns) > 0L) {
      add <- Matrix(
        0, nrow = nrow(X), ncol = length(missing_columns), sparse = TRUE,
        dimnames = list(NULL, missing_columns))
      X <- cbind(X, add)
    }
    X <- X[, train_columns, drop = FALSE]
    if (any(unseen)) X[unseen, ] <- 0
  }
  rownames(X) <- as.character(meta$doc_id)
  list(
    X = X,
    source_stage_levels = source_stage_levels,
    election_levels = election_levels,
    unseen = unseen
  )
}

fit_gst <- function(C, meta) {
  design <- build_gst_nuisance_matrix(meta)
  X <- design$X
  X <- X[rownames(C), , drop = FALSE]
  log_info(sprintf("GST nuisance design version: %s",
                   GST_NUISANCE_DESIGN_VERSION))
  log_info(sprintf("GST source x stage levels: %s",
                   paste(design$source_stage_levels, collapse = ", ")))

  # Single global session (T = 1) → R is a single-column intercept block.
  R <- sparseMatrix(i = seq_len(nrow(meta)), j = rep(1L, nrow(meta)),
                    x = as.numeric(meta$republican),
                    dims = c(nrow(meta), 1L),
                    dimnames = list(meta$doc_id, "republican"))
  R <- R[rownames(C), , drop = FALSE]

  mu <- log(rowSums(C))

  # Estimate penalized MLE — identical call to compute.r:26–39
  # Aggregated US fitting has a much wider response (32k+ phrase models) after
  # loading millions of source documents.  FORK workers inherit that large R
  # heap and then dirty/duplicate it while dmr() distributes column chunks;
  # this exhausted WSL RAM + swap in the production US run.  Start clean PSOCK
  # workers. Forking this large parent heap duplicates memory during GC.
  cl <- makeCluster(N_CORES, type = "PSOCK")
  on.exit(stopCluster(cl), add = TRUE)
  # dmr() itself sends count chunks and arguments. Export only the small helper;
  # copying C/X into every worker would defeat the bounded-memory design.
  worker_ok <- clusterEvalQ(cl, {
    suppressPackageStartupMessages({
      library(Matrix)
      library(gamlr)
      library(distrom)
    })
    TRUE
  })
  worker_exports <- new.env(parent = emptyenv())
  worker_exports$onerun <- get("onerun", envir = asNamespace("distrom"))
  clusterExport(cl, "onerun", envir = worker_exports)
  if (!all(unlist(worker_ok))) stop("PSOCK dmr worker initialization failed.")
  log_info(sprintf(paste0("Fitting dmr() via %s cluster (%d workers) with ",
                          "N=%s docs, V=%s phrases, P=%s covars"),
                   "PSOCK", N_CORES,
                   format(nrow(C), big.mark = ","),
                   format(ncol(C), big.mark = ","),
                   format(ncol(X) + ncol(R), big.mark = ",")))
  fit <- dmr(
    cl              = cl,
    covars          = cbind(X, R),
    counts          = C,
    mu              = mu,
    free            = seq_len(ncol(X)),
    fixedcost       = FIXEDCOST,
    lambda.start    = Inf,
    lambda.min.ratio = LAMBDA_MIN_RATIO,
    nlambda         = NLAMBDA,
    standardize     = FALSE
  )
  list(
    fit = fit, X = X, R = R,
    source_stage_levels = design$source_stage_levels,
    election_levels = design$election_levels
  )
}

# Crash-proof replacement for distrom::coef.dmr. coef.dmr builds its sparse
# coefficient matrix with dims = c(nrow(B[[1]]), length(B)); if the FIRST phrase
# has a degenerate (NULL) gamlr fit it is padded to a 1×1 zero, nrow(B[[1]])
# collapses to 1, and assembling the other columns throws
#   "'dims' must contain all (i,j) pairs".
# We instead size the matrix from a KNOWN-GOOD column and pad every degenerate
# column with zeros, so extraction never crashes regardless of fit order.
# Result is identical to coef.dmr for all non-degenerate columns.
safe_dmr_coef <- function(fit, k) {
  B  <- lapply(fit, function(m)
    tryCatch(coef(m, k = k, corrected = FALSE), error = function(e) NULL))
  ok <- vapply(B, function(b) !is.null(b) && nrow(b) > 1L, logical(1))
  good <- which(ok)
  if (length(good) == 0L)
    stop("safe_dmr_coef: every phrase fit is degenerate")
  P  <- nrow(B[[good[1]]]); rn <- rownames(B[[good[1]]])
  ndeg <- sum(!ok)
  if (ndeg > 0L)
    log_info(sprintf("safe_dmr_coef: %s degenerate phrase columns padded to zero",
                     format(ndeg, big.mark = ",")))
  ii <- jj <- xx <- vector("list", length(good))
  for (n in seq_along(good)) {
    j <- good[n]; b <- as.numeric(B[[j]])
    if (length(b) != P) next
    nz <- which(b != 0)
    ii[[n]] <- nz; jj[[n]] <- rep.int(j, length(nz)); xx[[n]] <- b[nz]
  }
  Matrix::sparseMatrix(
    i = unlist(ii, use.names = FALSE),
    j = unlist(jj, use.names = FALSE),
    x = unlist(xx, use.names = FALSE),
    dims = c(P, length(B)), dimnames = list(rn, names(fit)))
}

compute_pi <- function(fit, C, X, R, republican,
                       batch_size = PI_BATCH_SIZE, k = NULL) {
  # Per-document posterior P(Republican | text, covariates), computed via
  # the closed-form Bayes posterior under the fitted multinomial-choice
  # model and a uniform party prior:
  #
  #     log P(c | R, X) - log P(c | D, X)
  #         = sum_v c_v * phi_v  -  N * (log Z_R(X) - log Z_D(X))
  #     pi  = sigmoid(...)
  #
  # where phi_v = coefs_R[1, v] is the trained Republican log-rate-ratio
  # for phrase v (T=1 → one global "session"), U_D = intercept + X %*% coefs_X,
  # U_R = U_D + phi, and Z_p = sum_v exp(U_p[v]).
  #
  # We deliberately depart from compute.r:42–80 here: that block computes
  # the paper's session-averaged π statistic, which is a function of X only
  # and gives identical π to every speaker sharing X — useless for per-doc
  # classification. The Bayes posterior above actually uses the document's
  # word counts c_v and yields a meaningful per-doc score.
  # k is the BIC penalty for selecting one model off the regularization path.
  # Default log(nrow(X)) selects coefficients using the estimation-sample BIC
  # penalty. Predict mode passes the same k = log(N_train), so coefficients are a
  # property of the trained model, not of the set being scored.
  if (is.null(k)) k <- log(nrow(X))
  coefs   <- as.matrix(safe_dmr_coef(fit, k = k))
  coefs_X <- coefs[colnames(X), , drop = FALSE]
  coefs_R <- coefs[colnames(R), , drop = FALSE]
  intercept_row <- coefs["intercept", , drop = FALSE]
  phi_global    <- as.numeric(coefs_R)   # length V (T=1 → 1 × V collapsed)
  stopifnot(length(phi_global) == ncol(C))
  intercept_vec <- as.numeric(intercept_row)

  N <- nrow(C)
  pi_vec <- numeric(N)
  starts <- seq.int(1L, N, by = batch_size)

  log_sum_exp_rows <- function(M) {
    m <- apply(M, 1, max)
    m + log(rowSums(exp(M - m)))
  }

  for (start in starts) {
    end <- min(start + batch_size - 1L, N)
    idx <- start:end

    X_b <- X[idx, , drop = FALSE]
    C_b <- C[idx, , drop = FALSE]
    N_i <- as.numeric(rowSums(C_b))           # length B (total words)

    # U_D (B × V) = intercept + X %*% coefs_X
    ud_b <- as.matrix(X_b %*% coefs_X)
    ud_b <- sweep(ud_b, 2, intercept_vec, FUN = "+")

    # U_R = U_D + phi (broadcast phi across rows)
    ur_b <- sweep(ud_b, 2, phi_global, FUN = "+")

    logZ_D <- log_sum_exp_rows(ud_b)
    logZ_R <- log_sum_exp_rows(ur_b)

    # Sum over v of c_v * phi_v, batched as sparse-matrix-vector product.
    score_text <- as.numeric(C_b %*% phi_global)   # length B
    log_odds   <- score_text - N_i * (logZ_R - logZ_D)
    pi_vec[idx] <- plogis(log_odds)                # numerically stable sigmoid
  }

  list(pi = pi_vec, coefs = coefs, coefs_X = coefs_X, coefs_R = coefs_R,
       phi_global = phi_global)
}

# Score a new (val / new-data) split with the same trained fit.
# Aligns covariates to the training-time factor levels so column names match.
score_split <- function(fit_obj_pack, C_new, meta_new, k = NULL) {
  fit <- fit_obj_pack$fit
  design <- build_gst_nuisance_matrix(
    meta_new,
    source_stage_levels = fit_obj_pack$source_stage_levels,
    election_levels = fit_obj_pack$election_levels,
    train_columns = colnames(fit_obj_pack$X)
  )
  X_new <- design$X
  X_new <- X_new[rownames(C_new), , drop = FALSE]

  R_new <- sparseMatrix(i = seq_len(nrow(meta_new)),
                        j = rep(1L, nrow(meta_new)),
                        x = as.numeric(meta_new$republican),
                        dims = c(nrow(meta_new), 1L),
                        dimnames = list(meta_new$doc_id, "republican"))
  R_new <- R_new[rownames(C_new), , drop = FALSE]

  # Align republican vector to row order of C_new (which equals meta$doc_id).
  rep_vec <- as.numeric(meta_new$republican)
  names(rep_vec) <- meta_new$doc_id
  rep_vec <- rep_vec[rownames(C_new)]

  compute_pi(fit, C_new, X_new, R_new, republican = rep_vec, k = k)
}

# ============================================================================
# Main
# ============================================================================

parse_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  country <- NULL
  predict <- FALSE
  identity_smoke <- FALSE
  design_smoke <- FALSE
  i <- 1L
  while (i <= length(args)) {
    a <- args[i]
    if (a == "--country") {
      country <- args[i + 1L]; i <- i + 2L
    } else if (a == "--predict") {
      predict <- TRUE; i <- i + 1L
    } else if (a == "--identity-smoke") {
      identity_smoke <- TRUE; i <- i + 1L
    } else if (a == "--design-smoke") {
      design_smoke <- TRUE; i <- i + 1L
    } else {
      stop("Unknown argument: ", a)
    }
  }
  if (is.null(country) || !country %in% c("us", "fr"))
    stop(paste0("Usage: Rscript 02_party_classifier_gst.R --country {us|fr} ",
                "[--predict|--identity-smoke|--design-smoke]"))
  if (sum(c(predict, identity_smoke, design_smoke)) > 1L)
    stop("--predict, --identity-smoke, and --design-smoke are mutually exclusive.")
  if (identity_smoke && country != "us")
    stop("--identity-smoke currently applies only to the U.S. candidate sources.")
  list(country = country, predict = predict, identity_smoke = identity_smoke,
       design_smoke = design_smoke)
}

design_smoke_main <- function(country) {
  if (country == "us") {
    meta <- data.frame(
      doc_id = paste0("us_design_", seq_len(6L)),
      source = c("congress", "tweet", "tv_ad", "tv_ad", "website", "website"),
      election_stage = c("office", "office", "primary", "general",
                         "primary", "general"),
      election_year = c(2000L, 2002L, 2004L, 2004L, 2006L, 2006L),
      stringsAsFactors = FALSE
    )
    expected <- c("congress::office", "tweet::office", "tv_ad::primary",
                  "tv_ad::general", "website::primary", "website::general")
  } else {
    meta <- data.frame(
      doc_id = paste0("fr_design_", seq_len(3L)),
      source = c("parliament", "professions_de_foi", "professions_de_foi"),
      election_stage = c("office", "round1", "round2"),
      election_year = c(2002L, 2007L, 2007L),
      stringsAsFactors = FALSE
    )
    expected <- c("parliament::office", "professions_de_foi::round1",
                  "professions_de_foi::round2")
  }

  training <- build_gst_nuisance_matrix(meta)
  if (!setequal(training$source_stage_levels, expected))
    stop("GST design smoke did not preserve the expected source-stage levels.")
  replay_meta <- meta[rev(seq_len(nrow(meta))), , drop = FALSE]
  replay <- build_gst_nuisance_matrix(
    replay_meta,
    source_stage_levels = training$source_stage_levels,
    election_levels = training$election_levels,
    train_columns = colnames(training$X)
  )
  if (!identical(colnames(replay$X), colnames(training$X)))
    stop("GST design smoke failed fit/score column alignment.")

  unseen_meta <- meta[1L, , drop = FALSE]
  unseen_meta$doc_id <- paste0(country, "_unseen_year")
  unseen_meta$election_year <- 2099L
  unseen <- build_gst_nuisance_matrix(
    unseen_meta,
    source_stage_levels = training$source_stage_levels,
    election_levels = training$election_levels,
    train_columns = colnames(training$X)
  )
  if (any(as.numeric(unseen$X) != 0))
    stop("GST design smoke failed unseen-level zeroing.")

  log_info(sprintf("GST nuisance-design smoke passed | country=%s | version=%s",
                   country, GST_NUISANCE_DESIGN_VERSION))
  log_info(sprintf("  source x stage levels: %s",
                   paste(training$source_stage_levels, collapse = ", ")))
  invisible(NULL)
}

identity_smoke_main <- function() {
  log_info("U.S. contextual candidate-identity smoke test")
  dir.create(TEMP_DIR, recursive = TRUE, showWarnings = FALSE)
  tv <- load_us_tv_ad_cells(labeled_only = FALSE)
  tv_audit <- tv[, .(
    n_race = uniqueN(race), n_state = uniqueN(state2),
    n_district = uniqueN(district2)
  ), by = cell_key][n_race != 1L | n_state != 1L | n_district != 1L]
  if (nrow(tv_audit) > 0L)
    stop("TV identity smoke test found mixed campaign contexts.")
  log_info(sprintf("  TV smoke: %s rows in %s context-pure cells",
                   format(nrow(tv), big.mark = ","),
                   format(uniqueN(tv$cell_key), big.mark = ",")))
  tv_n_rows <- nrow(tv)
  tv_n_cells <- uniqueN(tv$cell_key)
  tv_cells <- tv[, .(
    n_ads = .N,
    n_candidate_ids = uniqueN(candidate_id),
    candidate_ids = paste(sort(unique(candidate_id)), collapse = "\u241e"),
    candidate_names = paste(sort(unique(candidate_name)), collapse = "\u241e")
  ), by = .(cell_key, election_year, election_stage, family,
            race, state2, district2)]
  setorder(tv_cells, election_year, race, state2, district2,
           election_stage, cell_key)
  fwrite(tv_cells, file.path(TEMP_DIR, "identity_audit_us_tv_cells.csv"))
  tv_multi_id_cells <- tv_cells[n_candidate_ids > 1L, .N]
  tv_context_splits <- tv_cells[, .(
    n_contexts = uniqueN(make_cell_key(race, state2, district2, family))
  ), by = .(
    entity_base = vapply(strsplit(cell_key, "\u241f", fixed = TRUE),
                         function(parts) parts[2L], character(1)),
    election_year, election_stage
  )][n_contexts > 1L, .N]
  rm(tv, tv_audit); gc(verbose = FALSE, full = TRUE)

  raw <- fread(
    file.path(DATA_DIR, "processed", "us_websites_classified.csv"),
    select = c("candidate", "state", "district", "year", "data_source",
               "stage", "party")
  )
  raw[, raw_idx := .I - 1L]
  native <- uniqueN(raw, by = c("candidate", "state", "district", "year", "stage"))
  old_audit_dir <- Sys.getenv("GST_R_IDENTITY_AUDIT_DIR", unset = NA_character_)
  Sys.setenv(GST_R_IDENTITY_AUDIT_DIR = TEMP_DIR)
  on.exit({
    if (is.na(old_audit_dir)) {
      Sys.unsetenv("GST_R_IDENTITY_AUDIT_DIR")
    } else {
      Sys.setenv(GST_R_IDENTITY_AUDIT_DIR = old_audit_dir)
    }
  }, add = TRUE)
  raw <- canonicalize_website_candidates(raw)
  canonical <- uniqueN(
    raw, by = c("candidate_canonical", "state", "district", "year", "party", "stage")
  )
  both_source <- raw[, .(n_data_sources = uniqueN(data_source)),
                     by = .(candidate_canonical, state, district, year,
                            party, stage)][n_data_sources > 1L, .N]
  akin <- raw[
    state == "missouri" & district == 2L & year == 2006L &
      tolower(candidate) %chin% c("akin, w. todd", "akin_todd"),
    unique(candidate_canonical)
  ]
  brown <- raw[
    state == "florida" & district == 3L & year == 2004L &
      candidate %chin% c("brown_corrine", "brown_prince"),
    unique(candidate_canonical)
  ]
  hough <- raw[
    state == "hawaii" & district == 1L & year == 2006L &
      candidate %chin% c("Hough, Richard A.", "hough_noah"),
    unique(candidate_canonical)
  ]
  andrews <- raw[
    state == "new jersey" & district == 1L & year == 2008L &
      candidate %chin% c("Andrews, Robert E.", "andrews_camille"),
    unique(candidate_canonical)
  ]
  smith <- raw[
    state == "new mexico" & district == 2L & year == 2002L &
      candidate %chin% c("Smith, John Arthur", "smith_ruben"),
    unique(candidate_canonical)
  ]
  if (length(akin) != 1L || length(brown) != 2L ||
      length(hough) != 1L || length(andrews) != 2L || length(smith) != 2L)
    stop("Website identity smoke test failed positive/negative controls.")
  log_info(sprintf(paste0("  Website smoke: native cells=%s; canonical cells=%s; ",
                          "Akin/Hough aliases merged; Brown/Andrews/Smith ",
                          "opponents separated"),
                   format(native, big.mark = ","),
                   format(canonical, big.mark = ",")))
  identity_summary <- data.table(
    metric = c(
      "tv_attributable_unique_ads", "tv_canonical_cells",
      "tv_multi_id_cells", "tv_reused_identity_context_splits",
      "website_raw_snapshots", "website_native_cells",
      "website_canonical_cells", "website_both_source_cells",
      "website_reviewed_override_merges", "website_reviewed_override_keeps"
    ),
    value = c(
      tv_n_rows, tv_n_cells, tv_multi_id_cells, tv_context_splits,
      nrow(raw), native, canonical, both_source,
      load_website_identity_overrides()[decision == "merge", .N],
      load_website_identity_overrides()[decision == "keep", .N]
    )
  )
  fwrite(identity_summary, file.path(
    TEMP_DIR, "identity_audit_us_summary.csv"))
  invisible(NULL)
}

# ============================================================================
# Predict mode — aggregate the full corpus to the exact same substantive cells
# used for fitting, then project each cell on the trained GST sufficient
# reduction. Third-party/centrist families are retained and scored.
# ============================================================================

predict_main <- function(country) {
  log_info(sprintf("PREDICT mode | country=%s | full aggregated corpus", country))

  ckpt_dir <- checkpoint_dir(country)
  fit   <- readRDS(file.path(ckpt_dir, "fit.rds"))
  vocab <- readRDS(file.path(ckpt_dir, "vocabulary.rds"))
  cov   <- readRDS(file.path(ckpt_dir, "covariates.rds"))
  if (!identical(cov$vocab_source, frozen_vocab_relative_path(country)))
    stop("Checkpoint was not fitted with the canonical curated vocabulary: ",
         frozen_vocab_relative_path(country))
  current_vocab_md5 <- unname(tools::md5sum(frozen_vocab_path(country)))
  if (is.null(cov$vocab_md5) || !identical(cov$vocab_md5, current_vocab_md5))
    stop("Canonical vocabulary differs from the file used to fit this checkpoint.")
  if (!identical(cov$aggregation, "source_specific_politician_election_cells"))
    stop("Checkpoint does not use the canonical GST cell specification.")
  if (!identical(cov$gst_nuisance_design_version,
                 GST_NUISANCE_DESIGN_VERSION))
    stop("Checkpoint predates the source-by-stage GST nuisance design; refit required.")
  if (country == "us" &&
      !identical(cov$cell_identity_version, CELL_IDENTITY_VERSION))
    stop("U.S. checkpoint predates the corrected contextual candidate identity; refit required.")
  if (country == "us") {
    current_overrides_md5 <- unname(tools::md5sum(
      website_identity_overrides_path()))
    if (is.null(cov$website_identity_overrides_source) ||
        !identical(cov$website_identity_overrides_source,
                   WEBSITE_IDENTITY_OVERRIDES_RELATIVE) ||
        is.null(cov$website_identity_overrides_md5) ||
        !identical(cov$website_identity_overrides_md5,
                   current_overrides_md5))
      stop("Reviewed website identity overrides differ from the file used to fit ",
           "this U.S. checkpoint; refit required.")
  }
  n_train <- tryCatch(fit[[1]]$nobs, error = function(e) NA_integer_)
  log_info(sprintf("Loaded fit (N_train=%s) / vocab (%s phrases) / covariates",
                   format(n_train, big.mark = ","),
                   format(length(vocab), big.mark = ",")))

  agg <- aggregate_country_counts(country, vocab, labeled_only = FALSE)
  C <- agg$C
  meta <- agg$meta
  diagnostics_path <- file.path(
    TEMP_DIR, sprintf("prediction_aggregation_diagnostics_%s_gst_R.csv", country))
  fwrite(agg$summary, diagnostics_path)
  rm(agg); gc(verbose = FALSE, full = TRUE)
  log_info(sprintf("Prediction corpus: %s cells", format(nrow(meta), big.mark = ",")))
  log_mem("after prediction aggregation")

  # Out-of-sample scoring = the sufficient-reduction PROJECTION (Taddy MNIR;
  # Gentzkow-Shapiro-Taddy 2019b; Eq. 1 of Widmer et al. "Media Slant is
  # Contagious"):  slant_n = sum_b f_bn * phi_b,  f_bn = c_bn / sum_b c_bn.
  # phi_b = the per-phrase Republican/right loading = the 'republican' coef row,
  # extracted ONCE off the regularization path at the training BIC penalty
  # k = log(N_train). No partition function and no covariates at score time, so
  # scoring is one sparse matmul per chunk — vs the old per-doc Bayes posterior
  # that recomputed a dense V-wide log-partition sum for EVERY document (the
  # ~45,000x-redundant cost behind the slow/OOM overnight runs).
  k_train <- if (!is.na(n_train)) log(n_train) else log(nrow(meta))
  coefs <- as.matrix(safe_dmr_coef(fit, k = k_train))
  if (!("republican" %in% rownames(coefs)))
    stop("predict: coefficient matrix has no 'republican' loading row")
  phi <- as.numeric(coefs["republican", ])
  if (length(phi) != length(vocab))
    stop(sprintf("predict: phi length (%d) != vocab length (%d) — column misalignment",
                 length(phi), length(vocab)))
  rm(coefs); gc(verbose = FALSE)
  log_info(sprintf("Extracted phi (Republican loadings) for %s phrases at k=log(N_train)=%.3f",
                   format(length(phi), big.mark = ","), k_train))

  dir.create(TEMP_DIR, recursive = TRUE, showWarnings = FALSE)
  meta[, slant := as.numeric(C %*% phi) / rowSums(C)]
  meta[, rep := as.integer(slant > 0)]
  write_gst_phrase_contributors(C, meta, vocab, phi, country)
  meta[, label := NULL]

  preferred <- c(
    "doc_id", "cell_key", "source", "entity", "entity_scope",
    "election_year", "election_stage", "family", "first_raw_idx",
    "n_docs", "n_docs_raw", "n_tokens", "candidate_canonical", "state",
    "district", "stage", "candidate_primary_aliases", "candidate_general_aliases",
    "candidate_name", "race", "state2", "district2", "name_MP_db",
    "start_term_MP_db", "end_term_MP_db", "id_unique_cand", "tour",
    "n_data_sources", "n_data_sources_raw", "slant", "rep"
  )
  keep_cols <- intersect(preferred, names(meta))
  setcolorder(meta, keep_cols)
  out_path <- file.path(TEMP_DIR, sprintf("predictions_%s_gst_R.csv", country))
  temporary <- paste0(out_path, ".tmp")
  fwrite(meta[, ..keep_cols], temporary)
  if (!file.rename(temporary, out_path))
    stop("Could not atomically replace prediction output: ", out_path)
  log_info(sprintf("Wrote %s cell predictions to %s",
                   format(nrow(meta), big.mark = ","), out_path))
  invisible(NULL)
}

main <- function() {
  args   <- parse_args()
  country <- args$country
  log_info(sprintf("Country: %s | N_CORES=%d | seed=%d",
                   country, N_CORES, SEED))
  log_info("Canonical cell aggregation | no sampling | curated vocabulary")

  ckpt_dir <- checkpoint_dir(country)
  dir.create(ckpt_dir, recursive = TRUE, showWarnings = FALSE)
  dir.create(TABLES_DIR, recursive = TRUE, showWarnings = FALSE)
  dir.create(TEMP_DIR,   recursive = TRUE, showWarnings = FALSE)

  vocab <- load_frozen_vocab(country)
  log_info(sprintf("Loaded canonical curated vocabulary: %s terms from %s",
                   format(length(vocab), big.mark = ","),
                   frozen_vocab_path(country)))
  agg <- aggregate_country_counts(country, vocab, labeled_only = TRUE)
  C_train <- agg$C
  train <- agg$meta
  summary_path <- file.path(
    TEMP_DIR, sprintf("aggregation_diagnostics_%s_gst_R.csv", country))
  fwrite(agg$summary, summary_path)
  cells_path <- file.path(
    TEMP_DIR, sprintf("aggregation_cells_%s_gst_R.csv", country))
  fwrite(train, cells_path)
  log_info(sprintf("Saved aggregation diagnostics to %s and %s",
                   summary_path, cells_path))
  rm(agg); gc(verbose = FALSE, full = TRUE)
  log_info(sprintf("Estimation sample: all %s labeled cells (no holdout)",
                   format(nrow(train), big.mark = ",")))
  train[, republican := as.integer(label == 1L)]

  # A vocabulary term absent from the full estimation sample has no estimable
  # coefficient.
  nz_cols <- which(Matrix::colSums(C_train) > 0)
  if (length(nz_cols) < ncol(C_train)) {
    log_info(sprintf(
      "Dropping %s zero-count vocabulary columns; %s remain",
      format(ncol(C_train) - length(nz_cols), big.mark = ","),
      format(length(nz_cols), big.mark = ",")))
    C_train <- C_train[, nz_cols, drop = FALSE]
    vocab <- vocab[nz_cols]
  }
  log_mem("after full-sample aggregation")

  # ---- Fit ------------------------------------------------------------------
  fit_pack <- fit_gst(C_train, train)
  log_info("dmr() fit complete.")
  log_mem("after dmr fit")

  # ---- Persist the expensive fit IMMEDIATELY, before scoring ----------------
  # dmr() is by far the costliest step; a downstream scoring/coef error (or a
  # machine crash) must never force a refit. Save fit + vocab + covariates now.
  saveRDS(fit_pack$fit, file = file.path(ckpt_dir, "fit.rds"))
  saveRDS(vocab,        file = file.path(ckpt_dir, "vocabulary.rds"))
  covariates_out <- list(
    sources = sort(unique(train$source)),
    election_years = sort(unique(train$election_year)),
    source_stage_levels = fit_pack$source_stage_levels,
    train_X_cols = colnames(fit_pack$X),
    train_R_cols = colnames(fit_pack$R),
    aggregation = "source_specific_politician_election_cells",
    cell_identity_version = CELL_IDENTITY_VERSION,
    gst_nuisance_design_version = GST_NUISANCE_DESIGN_VERSION,
    gst_nuisance_design =
      "source_by_election_stage_fixed_effects_plus_election_year_fixed_effects",
    cell_specification = c(
      congress = "politician_x_election",
      tweet = "politician_x_election",
      tv_ad = "politician_x_election_x_primary_general_x_campaign_context",
      website = "politician_x_election_x_primary_general",
      parliament = "politician_x_legislative_election",
      professions_de_foi = "one_standard_source_document_falc_0"
    ),
    zero_vocabulary_cells_dropped = TRUE,
    count_weighting = "raw_aggregated_phrase_counts",
    estimation_sample = "all_labeled_cells_no_holdout",
    reported_fit_diagnostics = "in_sample_bayes_posterior",
    vocab_source = frozen_vocab_relative_path(country),
    vocab_md5 = unname(tools::md5sum(frozen_vocab_path(country)))
  )
  if (country == "us") {
    covariates_out$website_identity_overrides_source <-
      WEBSITE_IDENTITY_OVERRIDES_RELATIVE
    covariates_out$website_identity_overrides_md5 <- unname(tools::md5sum(
      website_identity_overrides_path()))
  }
  saveRDS(covariates_out, file = file.path(ckpt_dir, "covariates.rds"))
  log_info(sprintf("Saved fit / vocab / covariates under %s (pre-scoring)",
                   ckpt_dir))

  log_info("Computing in-sample π fit diagnostics on all estimation cells...")

  # ---- In-sample π & φ ------------------------------------------------------
  res_train <- score_split(fit_pack, C_train, train, k = log(nrow(train)))
  pi_train  <- as.numeric(res_train$pi)

  # ---- In-sample fit metrics ------------------------------------------------
  overall <- metrics(train$label, pi_train)
  log_info(sprintf("TRAIN (in-sample)  Acc=%.4f  F1=%.4f  AUC=%.4f  N=%s",
                   overall$accuracy, overall$f1, overall$auc,
                   format(overall$n, big.mark = ",")))
  log_info(paste0("NOTE: these are descriptive in-sample MLE fit diagnostics, ",
                  "not held-out prediction estimates."))
  results <- list(c(list(source = "All"), overall))
  for (src in sort(unique(train$source))) {
    mask <- train$source == src
    if (!any(mask)) next
    m <- metrics(train$label[mask], pi_train[mask])
    results[[length(results) + 1L]] <- c(list(source = src), m)
    log_info(sprintf("  %s: n=%s acc=%.4f f1=%.4f auc=%.4f",
                     src, format(m$n, big.mark = ","),
                     m$accuracy, m$f1, m$auc))
  }
  # ---- Persist remaining artefacts -----------------------------------------
  # (fit / vocab / covariates were already saved above, pre-scoring.)

  # In-sample training-fit predictions CSV
  train_out <- copy(train)
  train_out[, prob := pi_train]
  train_out[, pred := as.integer(prob > 0.5)]
  train_pred_path <- file.path(
    TEMP_DIR, sprintf("train_predictions_%s_gst_R.csv", country))
  keep_cols <- intersect(
    c("doc_id", "cell_key", "entity", "source", "election_year",
      "election_stage", "label", "first_raw_idx", "n_docs", "n_docs_raw",
      "n_tokens", "candidate_canonical", "state", "district", "stage",
      "candidate_primary_aliases", "candidate_general_aliases", "candidate_name",
      "race", "state2", "district2", "name_MP_db", "start_term_MP_db",
      "end_term_MP_db",
      "id_unique_cand", "tour", "n_data_sources", "n_data_sources_raw",
      "prob", "pred"),
    names(train_out))
  fwrite(train_out[, ..keep_cols], train_pred_path)
  log_info(sprintf("Saved in-sample fit predictions to %s", train_pred_path))

  # Phrase × φ loadings — T=1 collapses coefs_R to a single row of V loadings.
  phi_phrase <- as.numeric(res_train$coefs_R)
  if (length(phi_phrase) != length(vocab)) {
    stop(sprintf("coefs_R has %d values, expected %d (vocab size)",
                 length(phi_phrase), length(vocab)))
  }
  loadings <- data.table(term = vocab, phi = phi_phrase)
  setorder(loadings, -phi)
  loadings_path <- file.path(TABLES_DIR,
                             sprintf("party_classifier_word_loadings_%s_gst_R.csv",
                                     country))
  fwrite(loadings, loadings_path)
  log_info(sprintf("Saved %s phrase loadings to %s",
                   format(nrow(loadings), big.mark = ","), loadings_path))

  top_n <- min(30L, nrow(loadings))
  log_info(sprintf("Top right phrases: %s",
                   paste(head(loadings$term, top_n), collapse = ", ")))
  log_info(sprintf("Top left  phrases: %s",
                   paste(tail(loadings$term, top_n), collapse = ", ")))

  # LaTeX table
  tex_path <- file.path(TABLES_DIR,
                        sprintf("party_classifier_%s_gst_R.tex", country))
  write_metrics_latex(results, country, tex_path)

  invisible(NULL)
}

# Skip execution when sourced for testing (set GST_R_SOURCE_ONLY=1).
if (Sys.getenv("GST_R_SOURCE_ONLY", unset = "") == "") {
  .args <- parse_args()
  if (isTRUE(.args$design_smoke)) {
    design_smoke_main(.args$country)
  } else if (isTRUE(.args$identity_smoke)) {
    identity_smoke_main()
  } else if (isTRUE(.args$predict)) {
    predict_main(.args$country)
  } else {
    main()
  }
}
