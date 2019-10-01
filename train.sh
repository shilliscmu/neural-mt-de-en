#!/bin/sh

vocab="data/char_vocab.bin"
train_src="data/train.de-en.de"
train_tgt="data/train.de-en.en"
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

python -u nmt.py \
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
    --batch-size 512 \
    --hidden-size 512 \
    --embed-size 256 \
    --dropout 0.1 \
    --dev-output ${work_dir}/dev_decode.txt \
    --valid-every 4 \
    --load-from work_dir/1569620668/model/epoch_22_trainLoss_69.65_TF_0.80 \
    --clip-grad 5.0 2>${work_dir}/output.txt