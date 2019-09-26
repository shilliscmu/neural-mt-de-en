import gc
import math
import sys
from collections import namedtuple
from typing import List, Any

import numpy as np
import torch
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import corpus_bleu
from torch import nn
from torch.nn.utils import rnn
from tqdm import tqdm

Hypothesis = namedtuple('Hypothesis', ['value', 'score', 'logits'])


def read_corpus(file_path, source):
    data = []
    for line in open(file_path, encoding="utf8"):
        sent = list(line.strip())
        # only append <s> and </s> to the target sentence
        if source == 'tgt':
            sent = ['<s>'] + sent + ['</s>']
        data.append(sent)

    return data


def batch_iter(data, vocab, batch_size, shuffle=False):
    """
    Given a list of examples, shuffle and slice them into mini-batches
    """
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_sents = [torch.LongTensor([vocab.src.word2id[word] if word in vocab.src else vocab.src.word2id['<unk>'] for word in e[0]]) for e in examples]
        src_lengths = torch.LongTensor([len(sent) for sent in src_sents])
        padded_src_sents = rnn.pad_sequence(src_sents, batch_first=True)
        tgt_sents = [torch.LongTensor([vocab.tgt.word2id[word] if word in vocab.tgt else vocab.tgt.word2id['<unk>']for word in e[1]]) for e in examples]
        tgt_lengths = torch.LongTensor([len(sent) for sent in tgt_sents])
        padded_tgt_sents = rnn.pad_sequence(tgt_sents, batch_first=True)

        yield padded_src_sents, src_lengths, padded_tgt_sents, tgt_lengths


def compute_corpus_level_bleu_score(references: List[List[str]], hypotheses: List[Hypothesis]) -> float:
    """
    Given decoding results and reference sentences, compute corpus-level BLEU score

    Args:
        references: a list of gold-standard reference target sentences
        hypotheses: a list of hypotheses, one for each reference

    Returns:
        bleu_score: corpus-level BLEU score
    """
    if references[0][0] == '<s>':
        references = [ref[1:-1] for ref in references]

    # for character model
    references = [''.join(ref) for ref in references]

    bleu_score = corpus_bleu([[ref] for ref in references],
                             [hyp.value for hyp in hypotheses])

    return bleu_score


def evaluate_ppl(model, criterion, vocab, dev_data: List[Any], dev_output_path, device):
    """
    Evaluate perplexity on dev sentences

    Args:
        dev_data: a list of dev sentences
        batch_size: batch size

    Returns:
        ppl: the perplexity on dev sentences
    """
    with torch.set_grad_enabled(False):
        cum_loss = 0.
        cum_ppl = 0.
        cum_tgt_words = 0.

        print_output = True
        batch_size = 1

        dev_tgts = [example[1] for example in dev_data]
        hyps = list()

        for src_sents, src_sent_lengths, tgt_sents, tgt_sent_lengths in tqdm(batch_iter(dev_data, vocab, batch_size)):
            src_sents, tgt_sents = src_sents.to(device), tgt_sents.to(device)

            output = model(src_sents, src_sent_lengths, tgt_sents, TF=0)
            output = output[:, :tgt_sents.shape[1]]
            loss = criterion(torch.cat(tuple(output), 0).to(device), torch.cat(tuple(tgt_sents), 0).to(device))
            mask = torch.zeros_like(tgt_sents)
            for batch_num, length in enumerate(tgt_sent_lengths):
                mask[batch_num, :length] = 1
            loss = (loss * torch.cat(tuple(mask), 0).type(torch.FloatTensor).to(device)).sum() / len(tgt_sent_lengths)

            tgt_word_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting the leading `<s>`
            cum_tgt_words += tgt_word_num_to_predict
            cum_loss += loss.item()
            cum_ppl += math.exp(loss.item() / cum_tgt_words * batch_size)

            output = output.cpu().detach()
            output = [np.argmax(output[batch_num, :length].numpy(), axis=1) for batch_num, length in
                      enumerate(tgt_sent_lengths)]
            output = [[vocab.tgt.get_word(char.item()) for char in o] for o in output]
            output = ''.join([item for sublist in output for item in sublist])

            hyps.append(Hypothesis(output, 0, []))

            if print_output:
                tgt_sents = tgt_sents.cpu().detach()
                tgt_sents = [tgt_sents[batch_num, :length].numpy() for batch_num, length in enumerate(tgt_sent_lengths)]
                tgt_sents = [[vocab.tgt.get_word(char.item()) for char in t] for t in tgt_sents]
                tgt_sents = ''.join([item for sublist in tgt_sents for item in sublist])
                with open(dev_output_path, 'w') as f:
                        f.write('transcripts:\n' + tgt_sents + '\n')
                        f.write('outputs:\n' + output + '\n\n')

            if len(hyps) == 10:
                print_output = False

        bleu = compute_corpus_level_bleu_score(dev_tgts, hyps)

        return cum_ppl, bleu


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Embedding:
        torch.nn.init.xavier_normal_(m.weight)
    elif type(m) == nn.LSTM or type(m) == nn.LSTMCell:
        for name, param in m.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                upper = 1 / np.sqrt(m.hidden_size)
                nn.init.uniform_(param, -upper, upper)


def plot_attention(x_label, y_label, attention_across_timesteps, attention_path, epoch):
    fig = plt.figure()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.imshow(attention_across_timesteps)
    fig.savefig(attention_path + "/epoch{:}.png".format(epoch))
    plt.close()


def plot_grad_flow(named_parameters, gradient_path, epoch_num, batch_num):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if p.requires_grad and ("bias" not in n) and p.grad is not None:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])

    plt.savefig(gradient_path + "/epoch{:}_batch{:}.png".format(epoch_num, batch_num), bbox_inches="tight")


def beam_search(model, test_data_src, vocab, beam_size=None) -> List[List[Hypothesis]]:
    hypotheses = []
    src_sents = [torch.LongTensor(
        [vocab.src.word2id[word] if word in vocab.src else vocab.src.word2id['<unk>'] for word in e]).cpu() for e in
                 test_data_src]

    for src_sent in tqdm(src_sents, desc='Decoding', file=sys.stdout):
        example_hyps = model.beam_search(src_sent, torch.LongTensor([len(src_sent)]).cpu(), beam_size)
        hypotheses.append(example_hyps)

    return hypotheses