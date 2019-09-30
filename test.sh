#!/bin/sh

test_tgt="data/test.de-en.en"

python nmt.py \
    decode

perl multi-bleu.perl ${test_tgt} < decode.txt
