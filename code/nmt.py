"""
A very basic implementation of neural machine translation

Usage:
    nmt.py train --train-src=<file> --train-tgt=<file> --dev-src=<file> --dev-tgt=<file> --vocab=<file> [options]
    nmt.py decode [options] MODEL_PATH TEST_SOURCE_FILE OUTPUT_FILE
    nmt.py decode [options] MODEL_PATH TEST_SOURCE_FILE TEST_TARGET_FILE OUTPUT_FILE

Options:
    -h --help                               show this screen.
    --cuda                                  use GPU
    --train-src=<file>                      train source file
    --train-tgt=<file>                      train target file
    --dev-src=<file>                        dev source file
    --dev-tgt=<file>                        dev target file
    --vocab=<file>                          vocab file
    --seed=<int>                            seed [default: 0]
    --batch-size=<int>                      batch size [default: 32]
    --embed-size=<int>                      embedding size [default: 256]
    --hidden-size=<int>                     hidden size [default: 256]
    --dev-output=<string>                   dev decoded output filepath
    --clip-grad=<float>                     gradient clipping [default: 5.0]
    --log-every=<int>                       log every [default: 100]
    --max-epoch=<int>                       max epoch [default: 30]
    --patience=<int>                        wait for how many iterations to decay learning rate [default: 5]
    --max-num-trial=<int>                   terminate training after how many trials [default: 5]
    --weight-decay=<float>                  Weight decay [default: 1e-5]
    --beam-size=<int>                       beam size [default: 5]
    --lr=<float>                            learning rate [default: 0.001]
    --save-to=<string>                      model save path
    --load-from=<string>                    model load path
    --gradient-path=<string>                path to save gradients to
    --attention-path=<string>               path to save attention plots to
    --valid-every=<int>                     perform validation after how many epochs [default: 1]
    --dropout=<float>                       dropout [default: 0.2]
    --teacher-forcing=<float>               teacher forcing ratio [default: 1.0]
    --max-decoding-time-step=<int>          maximum number of decoding time steps [default: 50]
"""
import math
import os
import pickle
import sys
import time

import torch
from docopt import docopt
from torch import nn

from model import Seq2Seq
from code.utils import init_weights, read_corpus, batch_iter, evaluate_ppl, beam_search, compute_corpus_level_bleu_score, \
    plot_grad_flow, plot_attention


def train(args):
    vocab = pickle.load(open(args['--vocab'], 'rb'))
    device = torch.device('cuda' if args['--cuda'] else 'cpu')
    net = Seq2Seq(args['--embed-size'], args['--hidden-size'], len(vocab.src), len(vocab.tgt), args['--dropout'], device, args['--max-decoding-time-step'], args['--beam-size'])

    if args['--load-from']:
        if not args['--cuda']:
            args['--batch-size'] = 2
            net.load_state_dict(torch.load(args['--load-from'], map_location='cpu'))
        else:
            net.load_state_dict(torch.load(args['--load-from']))
        print('Loaded model from', args['--load-from'])
    else:
        net.apply(init_weights)
        print('Initialized model weights')

    net = net.to(device)

    args['--dev-output'] = args['--attention-path'] + '/dev_decode.txt'

    print('Prepping training and dev data.')
    train_data_src = read_corpus(args['--train-src'], source='src')
    train_data_tgt = read_corpus(args['--train-tgt'], source='tgt')

    dev_data_src = read_corpus(args['--dev-src'], source='src')
    dev_data_tgt = read_corpus(args['--dev-tgt'], source='tgt')

    train_data = list(zip(train_data_src, train_data_tgt))
    dev_data = list(zip(dev_data_src, dev_data_tgt))

    train_batch_size = int(args['--batch-size'])
    lr = float(args['--lr'])
    weight_decay = args['--weight-decay']
    clip_grad = float(args['--clip-grad'])
    teacher_forcing = float(args['--teacher-forcing'])
    log_every = int(args['--log-every'])
    valid_every = int(args['--valid-every'])
    model_save_path = args['--save-to']
    gradient_path = args['--gradient-path']
    attention_path = args['--attention-path']

    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.Adam(net.parameters(), lr, weight_decay=float(weight_decay))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=int(args['--patience']), threshold=0.01, verbose=True)

    print('begin Maximum Likelihood training')
    epoch = 0
    while True:
        if epoch > 0 and epoch % 5 == 0 and teacher_forcing > 0.1:
            teacher_forcing -= 0.1

        train_iter = cum_loss = cum_perp = cumulative_tgt_words = report_tgt_words = 0
        cumulative_examples = report_examples = 0
        train_time = begin_time = time.time()

        net = net.to(device)

        for src_sents, src_sent_lengths, tgt_sents, tgt_sent_lengths in batch_iter(train_data, vocab, batch_size=train_batch_size, shuffle=True):
            net.train()
            train_iter += 1
            batch_size = len(src_sents)

            tgt_words_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
            report_tgt_words += tgt_words_num_to_predict
            cumulative_tgt_words += tgt_words_num_to_predict
            report_examples += batch_size
            cumulative_examples += batch_size

            src_sents, tgt_sents = src_sents.to(device), tgt_sents.to(device)
            output, attention_across_timesteps = net(src_sents, src_sent_lengths, tgt_sents, TF=teacher_forcing)
            loss = criterion(torch.cat(tuple(output), 0).to(device), torch.cat(tuple(tgt_sents), 0).to(device))
            mask = torch.zeros_like(tgt_sents)
            for batch_num, length in enumerate(tgt_sent_lengths):
                mask[batch_num, :length] = 1
            loss = (loss * torch.cat(tuple(mask), 0).type(torch.FloatTensor).to(device)).sum() / batch_size
            cum_loss += loss.item()
            cum_perp += math.exp(loss.item() / cumulative_tgt_words * batch_size)
            loss.backward()
            if train_iter == 1:
                plot_grad_flow(net.named_parameters(), gradient_path, epoch, train_iter)
            nn.utils.clip_grad_norm_(net.parameters(), clip_grad)
            optimizer.step()
            optimizer.zero_grad()

            if train_iter == 1:
                plot_attention('Source words', 'Timesteps', attention_across_timesteps, attention_path, epoch)

            if train_iter % log_every == 0:
                print('epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f ' \
                      'cum. examples %d, speed %.2f words/sec, time elapsed %.2f min' % (epoch, train_iter,
                                                                                         cum_loss / train_iter,
                                                                                         cum_perp / train_iter,
                                                                                         cumulative_examples,
                                                                                         report_tgt_words / (time.time() - train_time),
                                                                                         (time.time() - begin_time) / 60))

                train_time = time.time()
                report_tgt_words = report_examples = 0.

        print('epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (epoch, train_iter,
                                                                                 cum_loss / train_iter,
                                                                                 cum_perp / train_iter,
                                                                                 cumulative_examples))
        if epoch % valid_every == 0:
            net.eval()
            print('begin validation ...', file=sys.stderr)

            dev_ppl, dev_bleu = evaluate_ppl(net, criterion, vocab, dev_data, args['--dev-output'], device)
            print('validation: iter %d, dev ppl %f, dev bleu %f' % (train_iter, dev_ppl, dev_bleu))
            scheduler.step(dev_bleu)

            if not os.path.exists(model_save_path):
                os.mkdir(model_save_path)
            backup_file = model_save_path + "/epoch_{:}_trainLoss_{:.2f}_devPerp_{:.2f}_devBleu_{:.2f}".format(epoch, cum_loss, dev_ppl, dev_bleu)
            torch.save(net.state_dict(), backup_file)

        epoch += 1
        if epoch == int(args['--max-epoch']):
            print('reached maximum number of epochs!', file=sys.stderr)
            exit(0)


def test(args):
    device = torch.device('cuda' if args['--cuda'] else 'cpu')
    vocab = pickle.load(open(args['--vocab'], 'rb'))
    net = Seq2Seq(args['--embed-size'], args['--hidden-size'], len(vocab.src), len(vocab.tgt), args['--dropout'], device, args['--max-decoding-time-step'])
    print(f"load model from {args['MODEL_PATH']}", file=sys.stderr)
    if not args['--cuda']:
        net.load_state_dict(torch.load(args['--load-from'], map_location='cpu'))
    else:
        net.load_state_dict(torch.load(args['--load-from']))
    net = net.to(device)
    net.eval()

    test_data_src = read_corpus(args['TEST_SOURCE_FILE'], source='src')
    if args['TEST_TARGET_FILE']:
        test_data_tgt = read_corpus(args['TEST_TARGET_FILE'], source='tgt')

    hypotheses = beam_search(net, test_data_src, vocab)

    if args['TEST_TARGET_FILE']:
        top_hypotheses = [hyps[0] for hyps in hypotheses]
        bleu_score = compute_corpus_level_bleu_score(test_data_tgt, top_hypotheses)
        print(f'Corpus BLEU: {bleu_score}', file=sys.stderr)

    with open(args['OUTPUT_FILE'], 'w') as f:
        for src_sent, hyps in zip(test_data_src, hypotheses):
            top_hyp = hyps[0]
            hyp_sent = ' '.join(top_hyp.value)
            f.write(hyp_sent + '\n')


def main():
    args = docopt(__doc__)
    if args['train']:
        train(args)
    elif args['decode']:
        test(args)
    else:
        raise RuntimeError(f'invalid mode')


if __name__ == '__main__':
    main()