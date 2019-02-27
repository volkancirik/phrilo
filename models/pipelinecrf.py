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

from lm_lstm_crf.model.crf import CRFDecode_vb
from lm_lstm_crf.model.lstm_crf import LSTM_CRF
from lm_lstm_crf.model.utils import encode_safe


def convert_bio(bio_pred, use_end=True):
  st = 0
  end = 0
  chunks = []

  prev_tag = -1

  bio_pred = bio_pred if use_end else bio_pred[:-1]
  for i, p in enumerate(bio_pred):
    if (prev_tag == 2 and prev_tag == p) or (prev_tag == 0 and p == 1) \
            or (prev_tag == 1 and p == 1):
      end = i
      prev_tag = p
      continue
    chunks += [(st, max(end, st))]
    st = i
    prev_tag = p

  chunks += [(st, max(end, st))]
  if chunks[-1][1] < len(bio_pred)-1:
    chunks += [(chunks[-1][1]+1, len(bio_pred)-1)]
  return chunks[1:]


class PIPELINECRF(nn.Module):
  def __init__(self, config):
    super(PIPELINECRF, self).__init__()
    self.config = config
    self.ALIGNER = torch.load(self.config['aligner'])

    checkpoint_file = torch.load(self.config['chunker'])
    self.f_map = checkpoint_file['f_map']
    self.l_map = checkpoint_file['l_map']

    self.CHUNKER = LSTM_CRF(len(self.f_map), len(
        self.l_map), self.config['word_dim'], self.config['hid_dim'],
        self.config['layer'], 0, large_CRF=True)
    self.CHUNKER.load_state_dict(checkpoint_file['state_dict'])
    self.CHUNKER.cuda()

    for param in self.ALIGNER.parameters():
      param.requires_grad = False
    for param in self.CHUNKER.parameters():
      param.requires_grad = False

    self.crf_decoder = CRFDecode_vb(
        len(self.l_map), self.l_map['<start>'], self.l_map['<pad>'])

  def forward(self, sentence, pos_tags, box_reps, gt_chunks, _gt_alignments, debug=False):

    cnn, spat = box_reps
    feats = [cnn, spat]
    box_feats = torch.cat(feats, 1)

    # dummy box
    box_dummy = makevar(np.zeros((1, box_feats.size(1))), numpy_var=True)
    box_feats = torch.cat([box_feats, box_dummy], 0)

    n_boxes = cnn.size(0)

    n_tokens = len(sentence)
    n_chunks = int((n_tokens*(n_tokens+1))/2)
    n_boxes = box_feats.size(0)

    score = [None]*n_chunks*(1 + n_boxes)
    score_np = np.zeros((1, n_chunks*(1 + n_boxes)))
    gt_chunk_set = set(gt_chunks)

    encoded = encode_safe(
        [sentence + [self.f_map['<eof>']]], self.f_map, self.f_map['<unk>'])
    encoded_tensor = torch.LongTensor(encoded).t().cuda()
    mask = torch.ByteTensor([[1]*len(sentence) + [1]]).t().cuda()
    scores, _ = self.CHUNKER(encoded_tensor)
    decoded = self.crf_decoder.decode(scores.data, mask.data)

    pred_chunks = convert_bio(decoded.t().data.numpy().tolist()[0])

    pred_alignments = defaultdict(list)
    for ch in pred_chunks:
      phrase_rep = self.ALIGNER.get_phrase_rep(
          sentence, pos_tags, ch[0], ch[1]+1)
      al_score = self.ALIGNER.score(phrase_rep, box_feats)
      argmax_prediction = int(torch.max(al_score, 1)[1].data.cpu().numpy())
      pred_alignments[ch].append(argmax_prediction)
    return 0, pred_chunks, pred_alignments
