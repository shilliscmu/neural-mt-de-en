import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from dropout import WeightDrop, LockedDropout
from utils import Hypothesis


class Encoder(nn.Module):
    # base = hidden_size // 2
    def __init__(self, in_dim, embed_size, base, p):
        super(Encoder, self).__init__()
        self.emb = nn.Embedding(in_dim, embed_size)
        self.lstm1 = nn.LSTM(embed_size, base, bidirectional=True, batch_first=True)
        self.lstm2 = self.__make_layer__(base*4, base)
        self.lstm3 = self.__make_layer__(base*4, base)

        self.fc1 = nn.Linear(base * 2, base * 2)
        self.fc2 = nn.Linear(base * 2, base * 2)
        self.act = nn.SELU(True)

        self.locked_drop = LockedDropout(0.1)
        self.drop = nn.Dropout(p)

    def __make_layer__(self, in_dim, out_dim):
        lstm = nn.LSTM(input_size=in_dim, hidden_size=out_dim, bidirectional=True, batch_first=True)
        return WeightDrop(lstm, ['weight_hh_l0', 'weight_hh_l0_reverse'],
                          dropout=0.1, variational=True)

    def _stride2(self, x):
        x = x[:, :x.size(1)//2*2]
        x = self.locked_drop(x)
        x = x.reshape(x.size(0), x.size(1)//2, x.size(2)*2)
        return x

    def forward(self, x, src_lengths):
        x = self.drop(self.emb(x))
        x = pack_padded_sequence(x, src_lengths, batch_first=True)
        x, _ = self.lstm1(x)

        x, seq_len = pad_packed_sequence(x, batch_first=True)
        x = self._stride2(x)

        x = pack_padded_sequence(x, seq_len//2, batch_first=True)
        x, _ = self.lstm2(x)
        x, _ = pad_packed_sequence(x, batch_first=True)
        x = self._stride2(x)

        x = pack_padded_sequence(x, seq_len//4, batch_first=True)
        x, (hidden, _) = self.lstm3(x)
        x, _ = pad_packed_sequence(x, batch_first=True)
        # x = self._stride2(x)
        #
        # x = pack_padded_sequence(x, seq_len//8, batch_first=True)
        # x, (hidden, _) = self.lstm4(x)
        # x, _ = pad_packed_sequence(x, batch_first=True)

        key = self.drop(self.act(self.fc1(x)))
        value = self.drop(self.act(self.fc2(x)))
        hidden = torch.cat([hidden[0, :, :], hidden[1, :, :]], dim=1)

        return seq_len//4, key, value, hidden


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, hidden, keys, values, mask, device):
        energy = torch.bmm(keys, hidden.unsqueeze(2))
        attention = F.softmax(energy, dim=1)
        masked_attention = F.normalize(attention * mask.unsqueeze(2).type(torch.FloatTensor).to(device), p=1)
        context = torch.bmm(masked_attention.permute(0, 2, 1), values)
        return context.squeeze(1), energy.cpu().squeeze(2).data.numpy()[0]


class Decoder(nn.Module):
    def __init__(self, lstm_dim, out_dim, p):
        super(Decoder, self).__init__()
        self.embed = nn.Embedding(out_dim, lstm_dim)
        self.lstm1 = nn.LSTMCell(lstm_dim * 2, lstm_dim)
        self.lstm2 = nn.LSTMCell(lstm_dim, lstm_dim)
        self.drop = nn.Dropout(p)
        self.fc = nn.Linear(lstm_dim, out_dim)
        self.embed.weight = self.fc.weight

    def forward(self, x, context, hidden1, cell1, hidden2, cell2, first_step):
        x = self.drop(self.embed(x))
        x = torch.cat([x, context], dim=1)
        if first_step:
            hidden1, cell1 = self.lstm1(x)
            hidden2, cell2 = self.lstm2(hidden1)
        else:
            hidden1, cell1 = self.lstm1(x, (hidden1, cell1))
            hidden2, cell2 = self.lstm2(hidden1, (hidden2, cell2))
        x = self.drop(self.fc(hidden2))
        return x, (hidden1, cell1, hidden2, cell2)


class Seq2Seq(nn.Module):

    def __init__(self, embed_size, hidden_size, in_dim, out_dim, dropout_rate, device, max_len=None, beam_size=5):
        super().__init__()
        self.device = device
        self.max_len = int(max_len)
        self.beam_size = int(beam_size)
        self.out_dim = out_dim

        self.encoder = Encoder(int(in_dim), int(embed_size), int(hidden_size) // 2, float(dropout_rate))
        self.decoder = Decoder(int(hidden_size), int(out_dim), float(dropout_rate))
        self.attention = Attention()

    def forward(self, src_sents, src_sent_lengths, tgt_sents, TF=0.7):
        max_len = len(tgt_sents[0])
        prediction = torch.zeros(max_len, tgt_sents.shape[0], self.out_dim).to(self.device)
        word = tgt_sents[:, 0]
        hidden1, cell1, hidden2, cell2 = None, None, None, None

        lens, key, value, hidden2 = self.encoder(src_sents, src_sent_lengths)
        mask = torch.arange(lens.max()).unsqueeze(0) < lens.unsqueeze(1)
        mask = mask.to(self.device)
        figure = []
        for t in range(1, max_len):
            context, attention = self.attention(hidden2, key, value, mask, self.device)
            word_vec, (hidden1, cell1, hidden2, cell2) = self.decoder(word, context, hidden1, cell1, hidden2, cell2,
                                                                    first_step=(t == 1))
            prediction[t] = word_vec
            if TF == 0:
                word = word_vec.max(1)[1]
            else:
                teacher_force = torch.rand(1) < TF
                if teacher_force:
                    word = tgt_sents[:, t]
                else:
                    gumbel_noise = torch.FloatTensor(np.random.gumbel(size=word_vec.size())).to(self.device)
                    noisy_word_vec = word_vec + gumbel_noise
                    word = noisy_word_vec.max(1)[1]

            figure.append(attention)
        prediction = prediction.permute(1, 0, 2)
        if TF == 0:
            return prediction
        if TF > 0:
            return prediction, np.stack(figure)

    def beam_search(self, src_sent, src_sent_length, beam_size=None):
        def score_update(old_score, current_update, time_step):
            return old_score * (time_step - 1) / time_step + current_update / time_step

        if beam_size is None:
            beam_size = self.beam_size
        hidden1, cell1, hidden2, cell2 = None, None, None, None

        lens, key, value, hidden2 = self.encoder(src_sent.unsqueeze(0), src_sent_length)
        mask = torch.arange(lens.max()).unsqueeze(0) < lens.unsqueeze(1)
        mask = mask.to(self.device)

        start = torch.LongTensor([1]).to(self.device)
        stop = torch.LongTensor([2]).to(self.device)
        beams = [Beam(0, start, (hidden1, cell1, hidden2, cell2), [])]

        for t in range(1, self.max_len):
            stopped = True
            temp_beams = list()
            for beam in beams:
                current_word = beam.words if beam.words.shape[0] == 1 else beam.words[-1]
                if current_word == stop:
                    temp_beams.append(beam)
                    continue
                else:
                    stopped = False
                hidden1, cell1, hidden2, cell2 = beam.hiddens
                context, attention = self.attention(hidden2, key, value, mask, self.device)
                logits, hiddens = self.decoder(beam.words if beam.words.shape[0] == 1 else beam.words[-1:], context, hidden1, cell1, hidden2, cell2,
                                                 first_step=(t == 1))
                probs = -nn.LogSoftmax(dim=1)(logits).squeeze()
                probs = probs.detach().cpu().numpy()
                top_words = np.argpartition(probs, beam_size - 1)[:beam_size]
                temp_beams.extend([Beam(score_update(beam.score, probs[i], t),
                                        torch.cat((beam.words, torch.LongTensor([i]).to(self.device))),
                                        hiddens,
                                        beam.logits + [logits]
                                        ) for i in top_words])

            beams = np.array(temp_beams)[np.argpartition(np.array([beam.score for beam in temp_beams]), beam_size-1)[:beam_size]]

            if stopped:
                break

        sorted_beams = sorted(beams, key=lambda x: x.score)
        return [b.get_hyp() for b in sorted_beams]


class Beam(object):
    def __init__(self, score, word, hiddens, logits):
        self.score = score
        self.words = word
        self.hiddens = hiddens
        self.logits = logits

    def get_hyp(self):
        return Hypothesis(self.words, self.score, torch.cat(self.logits))
