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

from models.lp_solvers import LPAligner
from models import aligner, chunker
from util.model_utils import makevar
from util.arguments import get_flickr30k_train


class PIPELINEILP(nn.Module):
  def __init__(self, config, reader):
    super(PIPELINEILP, self).__init__()
    self.config = config

    aligner_meta_file = self.config['aligner'] + '.meta'

    l = [line.strip() for line in open(aligner_meta_file)][0]
    aligner_command = l.split('[')[1].split(']')[0][1: -1]
    aligner_parser = get_flickr30k_train(return_parser=True)
    aligner_args = aligner_parser.parse_args(aligner_command.split())
    aligner_config = vars(aligner_args)
    self.ALIGNER = aligner.ALIGNER(aligner_config, reader.w2i, reader.i2w,
                                   args=aligner_args)

    chunker_meta_file = self.config['chunker'] + '.meta'

    l = [line.strip() for line in open(chunker_meta_file)][0]
    chunker_command = l.split('[')[1].split(']')[0][1: -1]
    chunker_parser = get_flickr30k_train(return_parser=True)
    chunker_args = chunker_parser.parse_args(chunker_command.split())
    chunker_config = vars(chunker_args)
    self.CHUNKER = chunker.CHUNKER(chunker_config, reader.w2i, reader.i2w)

    for param in self.ALIGNER.parameters():
      param.requires_grad = False
    for param in self.CHUNKER.parameters():
      param.requires_grad = False

    self.decoder = LPAligner(max_boxes_per_chunk=1,
                             min_boxes_per_chunk=1,
                             max_chunks_per_box=100,
                             min_chunks_per_box=0)

  def load_weights(self, reader):
    print('Model weights are loading...')
    aligner_model_file = self.config['aligner'] + '.model.best'
    self.ALIGNER.load_state_dict(torch.load(aligner_model_file))
    chunker_model_file = self.config['chunker'] + '.model.best'
    self.CHUNKER.load_state_dict(torch.load(chunker_model_file))

  def forward(self, sentence, pos_tags, box_reps, gt_chunks, _gt_alignments, debug=False):

    cnn, spat = box_reps
    feats = [cnn, spat]
    box_feats = torch.cat(feats, 1)

    # dummy box
    box_dummy = makevar(np.ones((1, box_feats.size(1))), numpy_var=True)
    box_feats = torch.cat([box_feats, box_dummy], 0)

    n_boxes = box_feats.size(0)

    self.CHUNKER.eval()
    self.ALIGNER.eval()

    _, pred_chunks, _ = self.CHUNKER(sentence, pos_tags, None, gt_chunks, None)

    pred_alignments = defaultdict(list)
    n_chunks = len(pred_chunks)
    score_np_batch = np.zeros((1, n_boxes * n_chunks))
    for ii, ch in enumerate(pred_chunks):
      phrase_rep = self.ALIGNER.get_phrase_rep(
          sentence, pos_tags, ch[0], ch[1]+1)
      al_score = self.ALIGNER.score(phrase_rep, box_feats)

      score_np_batch[0, ii*n_boxes: (ii+1) *
                     n_boxes] = al_score.data.cpu().numpy()
      # argmax_prediction = int(torch.max(al_score, 1)[1].data.cpu().numpy())
      # pred_alignments[ch].append(argmax_prediction)
    res_alignments = self.decoder.solve(
        score_np_batch, n_chunks, n_boxes)

    for ii, ch in enumerate(pred_chunks):
      pred_alignments[ch] += res_alignments[1][ii]
    return 0, pred_chunks, pred_alignments
