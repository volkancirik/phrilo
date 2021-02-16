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
from torch.nn.utils.weight_norm import weight_norm

from models.lp_solvers import LPAligner
from models.composer import COMPOSER, DAN, BILSTM, BERT
from util.model_utils import embed_symbols
from util.model_utils import encode_phrase
from util.model_utils import get_context_vector
from util.model_utils import encode_box
from util.model_utils import score_box
from util.model_utils import makevar
from util.model_utils import chunk_positional_encoding
from util.model_utils import get_context_emmbedding


class ALIGNER(nn.Module):
  def __init__(self, config, w2i, i2w):
    super(ALIGNER, self).__init__()
    self.config = config
    self.wdim = self.config['word_dim']
    self.hdim = self.config['hid_dim']
    self.bdim = self.config['box_dim']
    self.layer = self.config['layer']
    self.encoder = self.config['encoder']
    self.use_pos = self.config['use_pos']
    self.context_vector = self.config['context_vector']
    self.context_embedding = self.config['context_embedding']
    self.use_bert = self.config['use_bert']

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

    if self.use_pos:
      self.pos_coeff = 2
    else:
      self.pos_coeff = 1

    self.w2i = w2i
    self.i2w = i2w
    self.We_wrd = nn.Embedding(len(w2i), self.wdim)

    nlp = spacy.load('en_core_web_sm')
    self.t2i = {t: i for i, t in enumerate(
        nlp.pipeline[1][1].labels + ('<go>', '<eos>', '<unk>'))}
    self.i2t = {i: t for i, t in enumerate(
        nlp.pipeline[1][1].labels + ('<go>', '<eos>', '<unk>'))}
    self.We_pos = nn.Embedding(len(self.t2i), self.wdim)

    self.pos_dim = 32
    context_dim = 4*self.pos_dim
    if self.context_vector:
      context_dim += len(self.w2i)*2 + len(self.t2i)*2
    if self.context_embedding:
      context_dim += self.wdim*2

    if self.encoder == 'bilstm+att':
      self.ph_rnn = COMPOSER(self.wdim * self.pos_coeff, self.hdim,
                             encoder='lstm', use_hid=False)
      self.ph_dim = self.wdim*2
    elif self.encoder == 'average':
      self.ph_rnn = DAN(self.wdim * self.pos_coeff,
                        self.hdim, layer=self.layer)
      self.ph_dim = self.hdim
    elif self.encoder == 'bilstm':
      self.ph_rnn = BILSTM(self.wdim * self.pos_coeff, self.hdim)
      self.ph_dim = self.hdim * 2
    else:
      raise NotImplementedError()
    self.ph_dim += context_dim
    if self.use_bert:
      self.bert = BERT(layer_number=11, method='sum')
      self.ph_dim += 768

    plist = []
    for i in range(self.layer):
      if i == self.layer-1 and i == 0:
        plist.append(nn.Linear(self.ph_dim, self.bdim))

      elif i == self.layer-1:
        plist.append(nn.Linear(self.hdim, self.bdim))

      elif i == 0:
        plist.append(nn.Linear(self.ph_dim, self.hdim))

      else:
        plist.append(nn.Linear(self.hdim, self.hdim))
    self.Wpscore = nn.ModuleList(plist)

    flist = []
    for i in range(self.layer):
      if i == self.layer-1 and i == 0:
        flist.append(nn.Linear(self.bdim, 1))

      elif i == self.layer-1:
        flist.append(nn.Linear(self.hdim, 1))

      elif i == 0:
        flist.append(nn.Linear(self.bdim, self.hdim))

      else:
        flist.append(nn.Linear(self.hdim, self.hdim))
    self.Wfscore = nn.ModuleList(flist)

    self.criterion = nn.BCEWithLogitsLoss()

  def get_phrase_rep(self, sentence, pos_tags, start_idx, end_idx):
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

    phrase_feats = chunk_positional_encoding(start_idx, end_idx, self.pos_dim)
    phrase_rep_list = [phrase, phrase_feats]

    if self.use_bert:
      bert_rep = self.bert(sentence[start_idx:end_idx])
      phrase_rep_list += [bert_rep]

    if self.context_embedding:
      ctx_emb = get_context_emmbedding(
          sentence, start_idx, end_idx, self.w2i, self.We_wrd, window_size=self.context_embedding
      )
      phrase_rep_list += [ctx_emb]

    if self.context_vector:
      tok_left, tok_right = get_context_vector(
          sentence, start_idx, end_idx, self.w2i, window_size=self.context_vector)
      pos_left, pos_right = get_context_vector(
          pos_tags, start_idx, end_idx, self.t2i, window_size=self.context_vector)
      phrase_rep_list += [tok_left, tok_right, pos_left, pos_right]
    phrase_rep = torch.cat(phrase_rep_list, 1)
    return phrase_rep

  def score(self, phrase_rep, box_feats):
    phrase = encode_phrase(
        self.Wpscore, phrase_rep, self.layer, self.nonlinearity, dim=self.bdim)
    prod = phrase.expand_as(phrase) * box_feats
    n_boxes = box_feats.size(0)
    norm = prod / \
        (torch.norm(prod, 2, 1).view(n_boxes, 1)).expand_as(prod)
    score_batch = score_box(self.Wfscore, norm)
    return score_batch.view(1, n_boxes)

  def forward(self, sentence, pos_tags, box_reps, gt_chunks, gt_alignments, debug=False):

    if self.training:
      self.decoder = LPAligner(max_boxes_per_chunk=4,
                               min_boxes_per_chunk=1)
    else:
      self.decoder = LPAligner(max_boxes_per_chunk=1,
                               min_boxes_per_chunk=1)

    cnn, spat = box_reps
    feats = [cnn, spat]
    box_feats = torch.cat(feats, 1)
    # dummy box
    box_dummy = makevar(np.ones((1, box_feats.size(1))),
                        numpy_var=True)
    box_feats = torch.cat([box_feats, box_dummy], 0)

    n_boxes = cnn.size(0)

    grounded_chunks = []
    grounded_alignments = []
    for ii, al in enumerate(gt_alignments):
      if al != []:
        grounded_alignments += [al]
        grounded_chunks += [gt_chunks[ii]]

    n_boxes = box_feats.size(0)
    n_chunks = len(grounded_alignments)

    scores_batch = []
    score_np_batch = np.zeros((1, n_boxes * n_chunks))
    loss = 0.

    for ii in range(n_chunks):

      expected = [0]*n_boxes
      for b in grounded_alignments[ii]:
        expected[b] = 1.0
      gold = makevar(np.array([expected]), numpy_var=True)

      phrase_rep = self.get_phrase_rep(
          sentence, pos_tags, grounded_chunks[ii][0], grounded_chunks[ii][1]+1)
      score_batch = self.score(phrase_rep, box_feats)
      loss += self.criterion(score_batch, gold)

      augmentation = np.zeros((1, score_batch.size(1)))
      for k in range(n_boxes):
        if k not in grounded_alignments[ii] and self.training:
          augmentation[0, k] = 1.0
      score_batch = score_batch + makevar(augmentation, numpy_var=True).float()

      scores_batch.append(score_batch)
      score_np_batch[0, ii*n_boxes: (ii+1) *
                     n_boxes] = score_batch.data.cpu().numpy()

    score_batch = torch.cat(scores_batch, 1)
    pred_alignments = self.decoder.solve(
        score_np_batch, n_chunks, n_boxes)
    pred_score = 0.0

    for chunk in pred_alignments[1]:
      for box in pred_alignments[1][chunk]:
        idx = box + n_boxes*chunk
        pred_score += score_batch[0, idx]

    gt_score = 0.0
    for ii in range(n_chunks):
      for box in grounded_alignments[ii]:
        idx = box + n_boxes*ii
        gt_score += score_batch[0, idx]

    loss = loss + pred_score - gt_score

    converted = defaultdict(list)
    for ch in pred_alignments[1]:
      converted[grounded_chunks[ch]] = pred_alignments[1][ch]

    return loss, [], converted
