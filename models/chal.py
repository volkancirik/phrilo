# from __future__ import absolute_import
'''
TODO
- add references & descriptions to models
'''
__author__ = 'volkan cirik'
from collections import defaultdict
import numpy as np
import torch.nn as nn
import torch

from util.model_utils import embed_symbols
from util.model_utils import makevar
from util.arguments import get_flickr30k_train

from models.lp_solvers import LPChunkerAligner
from models import aligner, chunker

EPS = 1e-10


class CHAL(nn.Module):
  def __init__(self, config, reader,
               bce_loss=0.0,
               mmloss_scale=1.0):

    super(CHAL, self).__init__()
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

    print('aligner and chunker is loaded from {} and {}'.format(
        self.config['aligner'], self.config['chunker']))

    self.criterion = nn.BCEWithLogitsLoss()
    self.decoder_tst = LPChunkerAligner(max_boxes_per_chunk=1,
                                        min_boxes_per_chunk=1,
                                        max_chunks_per_box=100,
                                        min_chunks_per_box=0)
    self.decoder_trn = LPChunkerAligner(max_boxes_per_chunk=5,
                                        min_boxes_per_chunk=0,
                                        max_chunks_per_box=100,
                                        min_chunks_per_box=0)
    self.bce_loss = bce_loss
    self.mmloss_scale = mmloss_scale

  def load_weights(self, reader):
    print('Model weights are loading...')
    aligner_model_file = self.config['aligner'] + '.model.best'
    self.ALIGNER.load_state_dict(torch.load(aligner_model_file))
    chunker_model_file = self.config['chunker'] + '.model.best'
    self.CHUNKER.load_state_dict(torch.load(chunker_model_file))

  def forward(self, sentence, pos_tags, box_reps, gt_chunks, gt_alignments, debug=False):

    n_tokens = len(sentence)
    if self.training:
      decoder = self.decoder_trn
      max_chunk_length = 19
    else:
      o
      decoder = self.decoder_tst
      max_chunk_length = 10
    cnn, spat = box_reps
    feats = [cnn, spat]
    box_feats = torch.cat(feats, 1)

    # dummy box
    box_dummy = makevar(np.ones((1, box_feats.size(1))),
                        numpy_var=True)
    box_feats = torch.cat([box_feats, box_dummy], 0)

    grounded_chunks = []
    grounded_alignments = []
    for ii, al in enumerate(gt_alignments):
      if al != []:
        grounded_alignments += [al]
        grounded_chunks += [gt_chunks[ii]]

    leftover_length = max(n_tokens - max_chunk_length, 0)
    n_max_len_chunks = int((leftover_length*(leftover_length+1))/2)
    n_chunks = int((n_tokens*(n_tokens+1))/2) - n_max_len_chunks
    n_boxes = box_feats.size(0)

    score = [None]*n_chunks*(1 + n_boxes)
    score_np = np.zeros((1, n_chunks*(1 + n_boxes)))
    gt_chunk_set = set(grounded_chunks)

    chunk_idx = 0
    box_idx = 0

    ch_box2idx = {}
    idx2ch_box = {}

    expected = [1.0]*n_chunks*(1 + n_boxes)

    if self.ALIGNER.use_lxmert:
      lxrt_feat_seq, lxrt_pooled = self.ALIGNER.lxrt(
          [' '.join(sentence)], (cnn.unsqueeze(0), spat[:, :4].unsqueeze(0)))
      visual_sentence_context = lxrt_feat_seq.squeeze(0)[:len(sentence), :]
    else:
      visual_sentence_context = None
    if self.ALIGNER.contextualized:
      emb_tokens = embed_symbols(
          self.ALIGNER.w2i, self.ALIGNER.We_wrd, sentence, 0, len(sentence))
      if self.ALIGNER.use_pos:
        emb_tags = embed_symbols(
            self.ALIGNER.t2i, self.ALIGNER.We_pos, pos_tags, 0, len(sentence))
        emb_rep = torch.cat([torch.cat(emb_tags, 0),
                             torch.cat(emb_tokens, 0)], 1)

        sentence_context, _ = self.ALIGNER.context_fn(emb_rep,
                                                      return_seq=True)
      else:
        sentence_context, _ = self.ALIGNER.context_fn(torch.cat(emb_tokens, 0),
                                                      return_seq=True)
    else:
      sentence_context = None

    chunk2ch_idx = {}
    ch_idx2chunk = {}
    for i in range(n_tokens):
      for j in range(i, min(n_tokens, i+max_chunk_length)):

        chunk2ch_idx[(i, j)] = chunk_idx
        ch_idx2chunk[chunk_idx] = (i, j)
        phrase_score, phrase_rep = self.CHUNKER.score(
            sentence, pos_tags, i, j+1)

        if self.training and (i, j) not in gt_chunk_set:
          phrase_score += makevar(1.0).float()
          expected[chunk_idx] = 0.0

        score[chunk_idx] = phrase_score.view(1, 1)
        score_np[0, chunk_idx] = phrase_score.item()

        aligner_phrase_rep = self.ALIGNER.get_phrase_rep(
            sentence, pos_tags, i, j+1,
            sentence_context=sentence_context,
            visual_sentence_context=visual_sentence_context)
        al_logits = self.ALIGNER.score(aligner_phrase_rep, box_feats)

        for k in range(n_boxes):
          alignment_score = al_logits[0, k]

          if self.training:
            if (i, j) in gt_chunk_set:
              phrase_idx = gt_chunks.index((i, j))
              if k not in set(gt_alignments[phrase_idx]):
                alignment_score = alignment_score + \
                    makevar(1.0).float()
                expected[chunk_idx] = 0.0
            else:
              alignment_score = alignment_score + \
                  makevar(1.0).float()
              expected[chunk_idx] = 0.0

          score_np[0, n_chunks + box_idx] = alignment_score.item()
          score[n_chunks + box_idx] = alignment_score.view(1, 1)

          ch_box2idx[(chunk_idx, k)] = box_idx
          idx2ch_box[box_idx] = (chunk_idx, k)
          box_idx += 1
        chunk_idx += 1

    gold = makevar(np.array([expected]), numpy_var=True)
    predicted = torch.cat(score, 1)
    bce_loss = self.criterion(predicted, gold)

    pred_chunks, pred_alignments, tok2chunks, id2chunk, chunk2id = decoder.solve(
        score_np, n_tokens, n_boxes, max_chunk_length=max_chunk_length)

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

    mmloss_al = pred_al_score - gt_al_score
    mmloss_ch = pred_ch_score - gt_ch_score
    mmloss = mmloss_al + mmloss_ch

    loss = bce_loss * self.bce_loss + mmloss * self.mmloss_scale
    return loss, pred_chunks, converted
