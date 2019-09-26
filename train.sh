#!/bin/sh

vocab="data/vocab.bin"
train_src="data/train.de-en.de.wmixerprep"
train_tgt="data/train.de-en.en.wmixerprep"
dev_src="data/valid.de-en.de"
dev_tgt="data/valid.de-en.en"
test_src="data/test.de-en.de"
test_tgt="data/test.de-en.en"

work_dir="work_dir/`date +%s`"
gradient_dir="$work_dir/gradients"
attention_dir="$work_dir/attention"

mkdir -p ${work_dir}
mkdir -p ${gradient_dir}
mkdir -p ${attention_dir}
echo save results to ${work_dir}

python nmt.py \
    train \
    --train-src ${train_src} \
    --train-tgt ${train_tgt} \
    --dev-src ${dev_src} \
    --dev-tgt ${dev_tgt} \
    --vocab ${vocab} \
    --cuda \
    --save-to ${work_dir}/model/ \
    --gradient-path ${gradient_dir} \
    --attention-path ${attention_dir} \
    --batch-size 256 \
    --hidden-size 256 \
    --embed-size 256 \
    --dropout 0.3 \
    --dev-output ${work_dir}/dev_decode.txt \
    --clip-grad 5.0

python nmt.py \
    decode \
    --vocab ${vocab} \
    --beam-size 5 \
    --max-decoding-time-step 100 \
    ${work_dir}/model/ \
    ${test_src} \
    ${work_dir}/decode.txt

perl multi-bleu.perl ${test_tgt} < ${work_dir}/decode.txt