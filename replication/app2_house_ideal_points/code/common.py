"""Paths and shared helpers for Application 2 (House ideal points from votes and speeches).

Everything is relative to the module folder (on the HPC: /home/Gauthier/app2_house_ideal_points,
synced from replication/app2_house_ideal_points; datasets live only there, not in git):
    code/, slurm/    this code and the job scripts
    data/raw/        links to the raw inputs: hein-bound/, hein-daily/ (Gentzkow-Shapiro-Taddy
                     speeches and speaker maps), HSall_members.csv and HSall_votes.csv (Voteview)
    data/crosswalk/  01: GST speaker ids -> Voteview members (icpsr, bioguide)
APP2_DATA_ROOT overrides the data folder. Speeches already built and cleaned by Application 1 are
read from APP1_DATA (default /home/Gauthier/app1_us_congress/data), never modified.
"""

import os
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("APP2_DATA_ROOT", MODULE / "data"))
RAW = DATA / "raw"
CROSSWALK = DATA / "crosswalk"
APP1_DATA = Path(os.environ.get("APP1_DATA", "/home/Gauthier/app1_us_congress/data"))

# Hein editions as released by Gentzkow, Shapiro and Taddy: bound 43rd-111th, daily 97th-114th.
HEIN = {"bound": range(43, 112), "daily": range(97, 115)}
# Main build (as GST and Application 1): bound through the 111th, daily for the 112th-114th,
# govinfo (bioguide ids already attached) for the 115th-118th.
MAIN_EDITION = {**{c: "bound" for c in range(43, 112)}, **{c: "daily" for c in range(112, 115)}}
