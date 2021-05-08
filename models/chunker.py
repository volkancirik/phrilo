# from __future__ import absolute_import
'''
TODO
- add references & descriptions to models
'''
__author__ = 'volkan cirik'

from collections import defaultdict
import numpy as np
import spacy
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from models.lp_solvers import LPChunker
from models.composer import COMPOSER, DAN, BILSTM, BERT
from util.model_utils import makevar, init_forget
from util.model_utils import embed_symbols, get_context_vector, encode_phrase
from util.model_utils import chunk_positional_encoding
from util.model_utils import get_context_emmbedding
import pdb


class CHUNKER(nn.Module):
  def __init__(self, config, w2i, i2w,
               bce_loss=0.0,
               mmloss_scale=1.0):
    super(CHUNKER, self).__init__()
    self.config = config

    self.wdim = self.config['word_dim']
    self.hdim = self.config['hid_dim']
    self.layer = self.config['layer']
    self.encoder = self.config['encoder']
    self.use_pos = self.config['use_pos']
    self.context_vector = self.config['context_vector']
    self.context_embedding = self.config['context_embedding']
    self.use_bert = self.config['use_bert']

    self.RELU = nn.ReLU()
    self.TANH = nn.Tanh()
    self.SIGM = nn.Sigmoid()

    if config['nonlinearity'] == 'none':
      self.nonlinearity = None
    elif config['nonlinearity'] == 'relu':
      self.nonlinearity = nn.ReLU()
    elif config['nonlinearity'] == 'hardtanh':
      self.nonlinearity = nn.Hardtanh(inplace=False)
    elif config['nonlinearity'] == 'selu':
      self.nonlinearity = nn.SELU(inplace=False)
    else:
      raise NotImplementedError()

    self.decoder = LPChunker()

    if self.use_pos:
      self.pos_coeff = 2
    else:
      self.pos_coeff = 1

    if self.encoder == 'bilstm+att':
      self.ph_rnn = COMPOSER(self.wdim * self.pos_coeff, self.hdim,
                             encoder='lstm', use_hid=False)
      self.ph_dim = self.wdim * 2
    elif self.encoder == 'dan':
      self.ph_rnn = DAN(self.wdim * self.pos_coeff,
                        self.hdim, layer=self.layer)
      self.ph_dim = self.hdim
    elif self.encoder == 'bilstm':
      self.ph_rnn = BILSTM(self.wdim * self.pos_coeff, self.hdim)
      self.ph_dim = self.hdim * 2
    else:
      raise NotImplementedError()

    if self.use_bert:
      self.bert = BERT(layer_number=11, method='sum')
      self.ph_dim += 768

    self.w2i = w2i
    self.i2w = i2w
    self.We_wrd = nn.Embedding(len(w2i), self.wdim)

    mlist = []
    for i in range(self.layer):
      if i == self.layer - 1:
        mlist.append(nn.Linear(self.hdim, self.ph_dim))
      else:
        mlist.append(nn.Linear(self.hdim, self.hdim))
    self.Wff = nn.ModuleList(mlist)
    nlp = spacy.load('en_core_web_sm')
    # print('HERE!')
    # pdb.set_trace()
    self.t2i = {t: i for i, t in enumerate(
        nlp.pipeline[1][1].labels + ('<go>', '<eos>', '<unk>'))}
    self.i2t = {i: t for i, t in enumerate(
        nlp.pipeline[1][1].labels + ('<go>', '<eos>', '<unk>'))}
    self.We_pos = nn.Embedding(len(self.t2i), self.wdim)

    plist = []

    self.pos_dim = 32
    context_dim = self.pos_dim*4
    if self.context_vector:
      context_dim += len(self.w2i)*2 + len(self.t2i)*2
    if self.context_embedding:
      context_dim += self.wdim*2

    for i in range(self.layer):
      if i == self.layer-1 and i == 0:
        plist.append(nn.Linear(self.ph_dim+context_dim, 1))

      elif i == self.layer-1:
        plist.append(nn.Linear(self.hdim, 1))

      elif i == 0:
        plist.append(
            nn.Linear(self.ph_dim+context_dim, self.hdim))

      else:
        plist.append(nn.Linear(self.hdim, self.hdim))

    self.Wpscore = nn.ModuleList(plist)
    self.criterion = nn.BCEWithLogitsLoss()
    self.bce_loss = bce_loss
    self.mmloss_scale = mmloss_scale


  def score(self, sentence, pos_tags, start_idx, end_idx):
    emb_tokens = embed_symbols(
        self.w2i, self.We_wrd, sentence, start_idx, end_idx)
    if self.use_pos:
      emb_tags = embed_symbols(
          self.t2i, self.We_pos, pos_tags, start_idx, end_idx)
      emb_rep = torch.cat([torch.cat(emb_tags, 0),
                           torch.cat(emb_tokens, 0)], 1)
      phrase, _ = self.ph_rnn(emb_rep)
    else:
      phrase, _ = self.ph_rnn(torch.cat(emb_tokens, 0))

    phrase = phrase.contiguous().view(1, -1)
    phrase_feats = chunk_positional_encoding(start_idx, end_idx, self.pos_dim)

    phrase_rep_list = [phrase, phrase_feats]
    if self.context_embedding:
      ctx_emb = get_context_emmbedding(
          sentence, start_idx, end_idx, self.w2i, self.We_wrd, window_size=self.context_embedding)
      phrase_rep_list += [ctx_emb]
    if self.context_vector:
      tok_left, tok_right = get_context_vector(
          sentence, start_idx, end_idx, self.w2i, window_size=self.context_vector)
      pos_left, pos_right = get_context_vector(
          pos_tags, start_idx, end_idx, self.t2i, window_size=self.context_vector)
      phrase_rep_list += [tok_left, tok_right, pos_left, pos_right]

    if self.use_bert:
      bert_rep = self.bert(sentence[start_idx:end_idx])
      phrase_rep_list += [bert_rep]
    phrase_rep = torch.cat(
        phrase_rep_list, 1)
    phrase_score = encode_phrase(
        self.Wpscore, phrase_rep, self.layer, self.nonlinearity)
    return phrase_score, phrase_rep

  def forward(self, sentence, pos_tags, _box_reps, gt_chunks, _gt_alignments, debug=False):

    n_tokens = len(sentence)

    if self.training:
      max_chunk_length = np.max([ch[1]-ch[0]+1 for ch in gt_chunks]) + 5 # arbitrary number
      # max_chunk_length = 19
    else:
      max_chunk_length = 10

    leftover_length = max(n_tokens - max_chunk_length, 0)
    n_max_len_chunks = int((leftover_length*(leftover_length+1))/2)
    n_chunks = int((n_tokens*(n_tokens+1))/2) - n_max_len_chunks

    score = [None]*n_chunks
    score_np = np.zeros((1, n_chunks))

    idx = 0
    gt_chunks = set(gt_chunks)

    expected = [1.0]*n_chunks
    for i in range(0, n_tokens):
      for j in range(i, min(n_tokens, i+max_chunk_length)):

        phrase_score, _ = self.score(sentence, pos_tags, i, j+1)
        if self.training and (i, j) not in gt_chunks:
          phrase_score += makevar(1.0).float()
          expected[idx] = 0.0
        score[idx] = phrase_score
        score_np[0, idx] = phrase_score

        idx += 1

    pred_chunks, _, tok2chunks, id2chunk, chunk2id = self.decoder.solve(
        score_np, n_tokens, None, max_chunk_length=max_chunk_length)

    gold = makevar(np.array([expected]), numpy_var=True)
    ch_score = torch.cat(score, 1)
    bce_loss = self.criterion(ch_score, gold)

    pred_score = 0
    gt_score = 0
    for chunk in pred_chunks:
      pred_score += score[chunk2id[chunk]]

    pred_chunks_set = set(pred_chunks)
    for chunk in gt_chunks:
      gt_score += score[chunk2id[chunk]]
    mmloss = pred_score - gt_score
    loss = bce_loss * self.bce_loss + mmloss * self.mmloss_scale

    return loss, pred_chunks, {}
