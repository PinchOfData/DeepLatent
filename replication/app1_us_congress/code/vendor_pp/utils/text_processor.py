#!/usr/bin/env python
# -*- encoding: utf-8 -*-
#
# Local copy of deeplatent.utils.text_processor with marginal modifications:
#   1. spacy.load(...) no longer disables 'ner' (so entity labels are populated).
#   2. New `drop_ent_labels` kwarg: tokens whose ent_type_ matches are filtered.
#   3. Reload-model-every blocks use the same disable= list so NER stays on.
# Everything else is byte-identical to the original.

from typing import List, Optional
from tqdm import tqdm
import pandas as pd
import csv
import spacy
from spacy.cli import download as spacy_download
import re


class text_processor_new:
    """
    Clean a corpus with SpaCy.
    """

    def __init__(
        self,
        spacy_model: str,
        remove_punctuation: bool = True,
        remove_digits: bool = True,
        stop_words: List[str] = [],
        lowercase: bool = True,
        lemmatize: bool = True,
        pos_tags_to_keep: Optional[List[str]] = None,
        remove_chars: Optional[List[str]] = None,
        drop_ent_labels: Optional[List[str]] = None,
    ):
        """
        Args:
            spacy_model : string, name of the SpaCy model to use
            remove_punctuation : boolean, whether to remove punctuation or not
            remove_digits : boolean, whether to remove digits or not
            stop_words : list of strings, list of stop words to remove
            lowercase : boolean, whether to lowercase or not
            lemmatize : boolean, whether to lemmatize or not
            pos_tags_to_keep : list of strings, list of Part of Speech tags to keep
            remove_chars : list of strings, list of characters to remove
            drop_ent_labels : list of strings, NER labels (e.g. ["PERSON","GPE","LOC"])
                whose tokens should be dropped from output. None disables filtering.
        """

        if not spacy.util.is_package(spacy_model):
            spacy_download(spacy_model)

        self.spacy_model = spacy_model
        self._nlp = spacy.load(spacy_model, disable=["parser"])
        self._nlp.add_pipe("sentencizer")
        self.remove_punctuation = remove_punctuation
        self.remove_digits = remove_digits
        self.stop_words = stop_words
        self.lowercase = lowercase
        self.lemmatize = lemmatize
        self.pos_tags_to_keep = pos_tags_to_keep
        self.remove_chars = remove_chars
        self.drop_ent_labels = drop_ent_labels

    def clean_text(self, s) -> List[str]:
        """
        Clean a SpaCy doc.
        """

        if self.remove_punctuation:
            s = [t for t in s if t.is_punct == False]

        if self.remove_digits:
            #s = [t for t in s if t.is_digit == False]
            s = [t for t in s if not re.search(r'\d', t.text)]

        if self.drop_ent_labels is not None:
            s = [t for t in s if t.ent_type_ not in self.drop_ent_labels]

        if self.pos_tags_to_keep is not None:
            temp = []
            for t in s:
                if t.pos_ in self.pos_tags_to_keep:
                    temp.append(t)
            s = temp

        if self.lowercase and not self.lemmatize:
            s = [t.lower_ for t in s]

        if self.lowercase and self.lemmatize:
            s = [t.lemma_.lower() for t in s]

        if not self.lowercase and not self.lemmatize:
            s = [t.text for t in s]

        if self.remove_chars is not None:
            for char in self.remove_chars:
                s = [t.replace(char, " ") for t in s]

        s = [t for t in s if t not in self.stop_words]

        s = [t.strip() for t in s if t not in self.stop_words]

        s = " ".join(s)

        s = s.strip()

        return s

    def process_docs(
        self,
        docs: List[str],
        batch_size: int = 100,
        n_process: int = -1,
        output_path: Optional[str] = None,
        return_docs: bool = False,
        reload_model_every=10000,  # helps with memory leakage issues
        progress_bar: bool = True,
    ):
        """
        SpaCy pipeline with multiprocessing.
        """

        spacy_docs = self._nlp.pipe(docs, batch_size=batch_size, n_process=n_process)

        if progress_bar:
            spacy_docs = tqdm(spacy_docs, total=len(docs))

        counter = 0

        if output_path is None:
            docs_clean = []
            for doc in spacy_docs:
                cleaned_text = self.clean_text(doc)
                docs_clean.append(cleaned_text)
                counter += 1
                if counter > reload_model_every:
                    self._nlp = spacy.load(self.spacy_model, disable=["parser"])
                    counter = 0
        else:
            with open(output_path, "w", newline="") as csvfile:
                fieldnames = ["doc_clean"]
                writer = csv.writer(csvfile)
                writer.writerow(fieldnames)
                for doc in spacy_docs:
                    cleaned_text = self.clean_text(doc)
                    writer.writerow([cleaned_text])
                    counter += 1
                    if counter > reload_model_every:
                        self._nlp = spacy.load(
                            self.spacy_model, disable=["parser"]
                        )
                        counter = 0
            if return_docs:
                docs_clean = pd.read_csv(output_path)["doc_clean"].tolist()
            else:
                docs_clean = None

        return docs_clean
