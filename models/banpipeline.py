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
from util.model_utils import makevar
from models.composer import COMPOSER, DAN, BILSTM, BERT

from util.model_utils import embed_symbols
from util.model_utils import encode_phrase
from util.model_utils import get_context_vector
from util.model_utils import encode_box
from util.model_utils import score_box


class BANPIPELINE(nn.Module):
  def __init__(self, config):
    super(BANPIPELINE, self).__init__()
    self.config = config
    self.ALIGNER = torch.load(self.config['aligner'])
    self.CHUNKER = torch.load(self.config['chunker'])

    for param in self.ALIGNER.parameters():
      param.requires_grad = False
    for param in self.CHUNKER.parameters():
      param.requires_grad = False

  def forward(self, sentence, pos_tags, box_reps, gt_chunks, gt_alignments, debug=False):

    cnn, spat = box_reps
    feats = [cnn, spat]
    box_feats = torch.cat(feats, 1)

    # dummy box
    box_dummy = makevar(np.zeros((1, box_feats.size(1))), numpy_var=True)
    box_feats = torch.cat([box_feats, box_dummy], 0)

    n_boxes = cnn.size(0)

    for ii, ch in enumerate(gt_chunks):
      if not gt_alignments[ii]:
        gt_alignments[ii] = [n_boxes]

    n_tokens = len(sentence)
    n_chunks = int((n_tokens*(n_tokens+1))/2)
    n_boxes = box_feats.size(0)

    score = [None]*n_chunks*(1 + n_boxes)
    score_np = np.zeros((1, n_chunks*(1 + n_boxes)))
    gt_chunk_set = set(gt_chunks)

    idx = 0
    box_idx = 0

    ch_box2idx = {}
    idx2ch_box = {}

    expected = [1.0]*n_chunks*(1 + n_boxes)
    self.CHUNKER.eval()
    self.ALIGNER.eval()
    _, pred_chunks, _ = self.CHUNKER(sentence, pos_tags, None, gt_chunks, None)

    pred_alignments = defaultdict(list)
    for ch in pred_chunks:
      phrase_rep = self.ALIGNER.get_phrase_rep(
          sentence, pos_tags, ch[0], ch[1]+1)
      al_score = self.ALIGNER.score(phrase_rep, box_feats)
      argmax_prediction = int(torch.max(al_score, 1)[1].data.cpu().numpy())
      pred_alignments[ch].append(argmax_prediction)
    return 0, pred_chunks, pred_alignments
