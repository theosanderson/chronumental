#!/usr/bin/env bash
# Run chronumental over the public UCSC SARS-CoV-2 tree and metadata.
# Download those two files first, e.g. from
# http://hgdownload.soe.ucsc.edu/goldenPath/wuhCor1/UShER_SARS-CoV-2/
set -euo pipefail

chronumental \
    --tree public-latest.all.nwk.gz \
    --dates public-latest.metadata.tsv.gz
