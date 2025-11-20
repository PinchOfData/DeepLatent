# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-10-30

- Original public release.

## [0.1.1] - 2025-11-20

- Added support for a new fusion strategy, "corrected_poe", that corrects some shortcomings of the naive PoE used in the computer science literature. Corrected PoE notably ensures that the encoder class can, in principle, contain the true posterior (this is not the case for a mixture of experts or a naive PoE).

- Added a method get_mutual_information() that answers the following question: "How much does each modality move us away from the prior?"