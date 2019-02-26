import numpy as np
import torch
import torch.nn as nn

from util.model_utils import printVec, makevar, init_forget
import torch.nn.functional as F


class COMPOSER(nn.Module):
  def __init__(self, wdim, hdim, cdim=0, encoder='bilinear-bilstm', use_hid=True):
    super(COMPOSER, self).__init__()

    self.encoder = encoder
    self.use_hid = use_hid and self.encoder not in set(['fuse', 'ff'])

    self.wdim = wdim
    self.hdim = hdim

    self.cdim = cdim

    if self.use_hid:
      self.out_dim = hdim*2
    else:
      self.out_dim = wdim

    if self.encoder == 'bilinear-bilstm':

      self.Wprj = nn.Linear(self.cdim, self.hdim * 2)

      self.rnn0 = nn.LSTM(input_size=self.wdim, hidden_size=self.hdim,
                          num_layers=1, bidirectional=True)
      self.rnn1 = nn.LSTM(input_size=self.hdim*2, hidden_size=self.hdim,
                          num_layers=1, bidirectional=True, bias=False)
      init_forget(self.rnn0)
      init_forget(self.rnn1)

      self.h00 = makevar(np.zeros((2, 1, self.hdim)), numpy_var=True)
      self.c00 = makevar(np.zeros((2, 1, self.hdim)), numpy_var=True)
      self.h01 = makevar(np.zeros((2, 1, self.hdim)), numpy_var=True)
      self.c01 = makevar(np.zeros((2, 1, self.hdim)), numpy_var=True)

    elif self.encoder == 'lstm':
      self.Wscr = nn.Linear(self.hdim*4, 1)
      self.rnn0 = nn.LSTM(input_size=self.wdim, hidden_size=self.hdim,
                          num_layers=1, bidirectional=True)
      self.rnn1 = nn.LSTM(input_size=self.hdim*2, hidden_size=self.hdim,
                          num_layers=1, bidirectional=True, bias=False)
      init_forget(self.rnn0)
      init_forget(self.rnn1)

      self.h00 = makevar(np.zeros((2, 1, self.hdim)), numpy_var=True)
      self.c00 = makevar(np.zeros((2, 1, self.hdim)), numpy_var=True)
      self.h01 = makevar(np.zeros((2, 1, self.hdim)), numpy_var=True)
      self.c01 = makevar(np.zeros((2, 1, self.hdim)), numpy_var=True)

    elif self.encoder == 'fuse':
      self.Winp2ctx = nn.Linear(self.wdim, self.hdim)
      self.Wscr = nn.Linear(self.hdim, 1)

    elif self.encoder == 'ff':
      self.Wscr = nn.Linear(self.wdim, 1)

    else:
      raise NotImplementedError()

  def forward(self, inp_rep, context=False):

    if self.encoder == 'bilinear-bilstm':
      sequence = inp_rep.view(inp_rep.size(0), 1, inp_rep.size(1))

      output0, (ht0, ct0) = self.rnn0(sequence, (self.h00, self.c00))
      output1, (ht1, ct1) = self.rnn1(output0, (self.h01, self.c01))
      outputs = output1.view(output1.size(0), -1)

      proj = self.Wprj(context)
      scores = outputs.mm(proj.t()).t()
      attention = F.softmax(scores, dim=1).t()

    if self.encoder == 'lstm':
      if not context:
        sequence = inp_rep.view(inp_rep.size(0), 1, inp_rep.size(1))
      else:
        sequence = torch.cat([inp_rep, context.repeat(inp_rep.size(0), 1)], 1)
        sequence = sequence.view(sequence.size(0), 1, sequence.size(1))

      output0, (ht0, ct0) = self.rnn0(sequence, (self.h00, self.c00))
      output1, (ht1, ct1) = self.rnn1(output0, (self.h01, self.c01))
      outputs = torch.cat([output0.view(output0.size(0), -1),
                           output1.view(output1.size(0), -1)], 1)

      scores = self.Wscr(outputs).t()
      attention = F.softmax(scores, dim=1).t()

    elif self.encoder == 'fuse':
      proj = self.Winp2ctx(inp_rep)
      fused = context.expand_as(proj) * proj
      norm = fused / torch.norm(fused, 2, 1).expand_as(fused)
      scores = self.Wscr(norm).t()
      attention = F.softmax(scores, dim=1).t()

    elif self.encoder == 'ff':
      scores = self.Wscr(inp_rep).t()
      attention = F.softmax(scores).t()

    if self.use_hid:
      weighted = attention.expand_as(outputs) * outputs
    else:
      weighted = attention.expand_as(inp_rep) * inp_rep
    waverage = torch.sum(weighted, 0)
    return waverage, attention


class DAN(nn.Module):
  def __init__(self, wdim, hdim, layer=3):
    super(DAN, self).__init__()

    self.hdim = hdim
    self.wdim = wdim

    self.layer = layer

    self.Wff = nn.ModuleList([nn.Linear(self.wdim, self.hdim) if i == 0 else nn.Linear(
        self.hdim, self.hdim) for i in range(self.layer)])

    self.RELU = nn.ReLU()
    self.TANH = nn.Tanh()

  def forward(self, inp_rep):
    averaged = torch.mean(inp_rep, 0)
    for i, ff in enumerate(self.Wff):
      if i == len(self.Wff)-1:
        break
      averaged = self.RELU(ff(averaged))
    return self.TANH(self.Wff[-1](averaged)), ''


class BILSTM(nn.Module):
  def __init__(self, wdim, hdim):
    super(BILSTM, self).__init__()

    self.hdim = hdim
    self.wdim = wdim

    self.rnn0 = nn.LSTM(input_size=self.wdim, hidden_size=self.hdim,
                        num_layers=1, bidirectional=True)
    self.rnn1 = nn.LSTM(input_size=self.hdim*2, hidden_size=self.hdim,
                        num_layers=1, bidirectional=True, bias=False)
    init_forget(self.rnn0)
    init_forget(self.rnn1)

    self.h00 = nn.Parameter(
        makevar(np.zeros((2, 1, self.hdim)), numpy_var=True))
    self.c00 = nn.Parameter(
        makevar(np.zeros((2, 1, self.hdim)), numpy_var=True))
    self.h01 = nn.Parameter(
        makevar(np.zeros((2, 1, self.hdim)), numpy_var=True))
    self.c01 = nn.Parameter(
        makevar(np.zeros((2, 1, self.hdim)), numpy_var=True))

  def forward(self, inp_rep, return_seq=False):

    sequence = inp_rep.view(inp_rep.size(0), 1, inp_rep.size(1))

    sequence = inp_rep.view(inp_rep.size(0), 1, inp_rep.size(1))
    output0, (ht0, ct0) = self.rnn0(sequence, (self.h00, self.c00))
    output1, (ht1, ct1) = self.rnn1(output0, (self.h01, self.c01))

    if return_seq:
      out = output1
    else:
      out = output1[-1]
#    forward_ = output1[-1, :self.hdim]
#    backward = output1[0, self.hdim:]
#    return torch.cat((forward_, backward), dim=1), ''

    return out, ''


from pytorch_pretrained_bert import BertModel
from pytorch_pretrained_bert import BertTokenizer, BertModel
from torch.autograd import Variable


class BERT(nn.Module):
  def __init__(self, layer_number, method='sum'):
    super(BERT, self).__init__()

    if method not in set(['sum', 'mean', 'max']):
      raise NotImplementedError()

    self.layer_number = layer_number
    self.method = method
    self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    self.embedder = BertModel.from_pretrained('bert-base-uncased')

  def forward(self, tokens):

    tokenized = Variable(torch.tensor(
        [self.tokenizer.vocab.get(tok, self.tokenizer.vocab['[UNK]']) for tok in tokens])).view(1, -1).cuda()

    encoded_layers, _ = self.embedder.forward(tokenized)
    sequence = encoded_layers[self.layer_number]
    if self.method == 'sum':
      return torch.sum(sequence, 1)
    elif self.method == 'mean':
      return torch.mean(sequence, 1)
    elif self.method == 'max':
      return torch.max(sequence, 1)
