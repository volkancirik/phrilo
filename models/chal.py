# from __future__ import absolute_import
'''
TODO
- add references & descriptions to models
'''
__author__ = 'volkan cirik'

from collections import defaultdict
import numpy as np

import torch
import torch.nn as nn


from models.lp_solvers import LPChunkerAligner

from util.model_utils import makevar
import ipdb
EPS = 1e-10


class CHAL(nn.Module):
  def __init__(self, config):
    super(CHAL, self).__init__()
    self.config = config

    self.ALIGNER = torch.load(self.config['aligner'])
    self.CHUNKER = torch.load(self.config['chunker'])

    for param in self.CHUNKER.parameters():
      param.requires_grad = False

    print('aligner and chunker is loaded from {} and {}'.format(
        self.config['aligner'], self.config['chunker']))

    self.criterion = nn.BCEWithLogitsLoss()
    self.decoder = LPChunkerAligner(max_boxes_per_chunk=10,
                                    min_boxes_per_chunk=1)

  def forward(self, sentence, pos_tags, box_reps, gt_chunks, gt_alignments, debug=False):

    if self.training:
      for param in self.CHUNKER.parameters():
        param.requires_grad = True

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

#    n_gr_chunks = len(grounded_alignments)
    n_tokens = len(sentence)
    n_chunks = int((n_tokens*(n_tokens+1))/2)
    n_boxes = box_feats.size(0)

    score = [None]*n_chunks*(1 + n_boxes)
    score_np = np.zeros((1, n_chunks*(1 + n_boxes)))
    gt_chunk_set = set(grounded_chunks)

    idx = 0
    box_idx = 0

    ch_box2idx = {}
    idx2ch_box = {}

    expected = [1.0]*n_chunks*(1 + n_boxes)

    for i in range(n_tokens):
      for j in range(i, n_tokens):

        phrase_score, phrase_rep = self.CHUNKER.score(
            sentence, pos_tags, i, j+1)

        if self.training and (i, j) not in gt_chunk_set:
          phrase_score += makevar(1.0).float()
          expected[idx] = 0.0
        score[idx] = phrase_score
        score_np[0, idx] = phrase_score

        aligner_phrase_rep = self.ALIGNER.get_phrase_rep(
            sentence, pos_tags, i, j+1)
        al_logits = self.ALIGNER.score(aligner_phrase_rep, box_feats)

        for k in range(n_boxes):
          alignment_score = al_logits[0, k]

          if self.training:
            if (i, j) in gt_chunk_set:
              phrase_idx = gt_chunks.index((i, j))
              if k not in set(gt_alignments[phrase_idx]):
                alignment_score = alignment_score + \
                    makevar(1.0).float()
                expected[idx] = 0.0
            else:
              alignment_score = alignment_score + \
                  makevar(1.0).float()
              expected[idx] = 0.0

          score_np[0, n_chunks + box_idx] = alignment_score
          score[n_chunks + box_idx] = alignment_score.view(1, 1)

          ch_box2idx[(idx, k)] = box_idx
          idx2ch_box[box_idx] = (idx, k)
          box_idx += 1
        idx += 1

    gold = makevar(np.array([expected]), numpy_var=True)
    loss = self.criterion(torch.cat(score, 1), gold)

    mean_scale = np.mean(score_np[0, n_chunks:]) / \
        np.mean(score_np[0, 0:n_chunks])
    score_np[0, n_chunks:] = score_np[0, n_chunks:] / \
        ((mean_scale*1))
    # score_np = score_np - np.min(score_np)

    pred_chunks, pred_alignments, tok2chunks, id2chunk, chunk2id = self.decoder.solve(
        score_np, n_tokens, n_boxes)

    pred_al_score = 0
    pred_ch_score = 0
    gt_al_score = 0
    gt_ch_score = 0

    for chunk in pred_chunks:
      pred_ch_score += score[chunk2id[chunk]]
    for chunk in gt_chunks:
      gt_ch_score += score[chunk2id[chunk]]

    converted = defaultdict(list)
    for ch in pred_alignments[1]:
      converted[id2chunk[ch]] = pred_alignments[1][ch]

    pred_al2chunk = pred_alignments[0]

    for box in pred_al2chunk:
      for chunk in pred_al2chunk[box]:
        idx = box + n_chunks + n_boxes*chunk
        pred_al_score += score[idx]

    for ii, alist in enumerate(gt_alignments):
      for box in alist:
        idx = box + n_chunks + n_boxes*chunk2id[gt_chunks[ii]]
        gt_al_score += score[idx]

    loss_scale_joint = ((pred_al_score - gt_al_score) /
                        (pred_ch_score - gt_ch_score) + EPS)
    loss_scale_type = loss / ((pred_al_score - gt_al_score) +
                              (pred_ch_score - gt_ch_score))
    # print("\n")
    # print("n_boxes | n_tokens | n_chunks", n_boxes,
    #       n_tokens, n_chunks)
    # print("ch_box2idx\n", ch_box2idx)
    # print("\nidx2ch_box\n", idx2ch_box)
    # print("pred_ch_score - gt_ch_score", pred_ch_score - gt_ch_score)
    # print("pred_al_score - gt_al_score", pred_al_score - gt_al_score)
    # print("loss", loss)
    # print("joint loss_scale al/ch", loss_scale_joint)
    # print("type loss_scale bce/lamm", loss_scale_type)
    # print("mean weight scale al/ch", mean_scale)
    # print("")
    # input()
    # loss_scale = 0.0001

    loss_scale = 1.0
    loss = loss + (pred_ch_score*loss_scale + pred_al_score) - \
        (gt_ch_score*loss_scale + gt_al_score)

    return loss, pred_chunks, converted
