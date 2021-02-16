from __future__ import absolute_import

#import setGPU
import os
import pickle
import sys
from random import shuffle, seed
import h5py
import numpy as np
from tqdm import tqdm

import torch.optim as optim
import torch
import torch.nn as nn

from util.reader import Reader
from util.model_utils import get_box_feats, makevar, get_n_params, f1_score, alignment_score
from util.arguments import get_flickr30k_train
from models.get_model import get_model


def evaluate(net, split, box_data,
             task_chunk=False, task_align=False, target_metric="chunking",
             use_gt=False, use_predicted=False, details=False):

  pbar = tqdm(list(range(len(split[0]))))
  net.eval()

  f1_chunk = 0.0
  count_chunk = 1.0
  hit_align = 0.0
  count_align = 1.0

  for i in pbar:
    words = split[0][i]
    pred_boxes, gt_boxes = split[1][i]
    boxes = gt_boxes if use_gt else pred_boxes
    box_type = 'gt' if use_gt else 'pred'

    gt_chunks, gt_alignments, pos_tags = split[3][i]
    if all([al == [] for al in gt_alignments[box_type]]):
      continue

    box_reps = get_box_feats(
        boxes, box_data, box_type)
    _, pred_chunks, pred_alignments = net.forward(
        words, pos_tags, box_reps, gt_chunks, gt_alignments[box_type], debug=False)

    if use_predicted:
      for jj, (ph, al) in enumerate(zip(gt_chunks, gt_alignments[box_type])):
        if al == []:
          gt_alignments[box_type][jj] = [box_reps[0].size(0)]

    gt_al_tuples = []
    for ch, al in zip(gt_chunks, gt_alignments[box_type]):
      if al:
        gt_al_tuples.append(list(ch)+al)
    pred_al_tuples = [list(ch)+pred_alignments[ch] for ch in pred_alignments]

    if task_chunk and pred_chunks:
      f1_chunk += f1_score(pred_chunks, gt_chunks)
      count_chunk += 1
    if task_align and gt_al_tuples:
      hit, tot = alignment_score(
          pred_al_tuples, gt_al_tuples, use_predicted=box_reps[0].size(0))
      hit_align += hit
      count_align += tot

    pbar_str = "chunk {:3.4f} alignment {:3.4f}".format(
        f1_chunk / count_chunk, hit_align / count_align)
    if details:
      pbar_str += '>>>> {} ||| {}'.format(pred_al_tuples, gt_al_tuples)
    pbar.set_description(pbar_str)

  metrics = {"chunking":  f1_chunk/count_chunk,
             "alignment": hit_align/count_align}
  net.train()
  return metrics[target_metric]


def train():
  seed(2)

  args = get_flickr30k_train()
  config = vars(args)
  config['command'] = [" ".join(sys.argv[1:])]

  # create experiment folder
  if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)

  pid = str(os.getpid())
  pidx = 0
  while True:
    snapshot = os.path.join(args.save_path, pid + '_' + str(pidx))
    if not os.path.exists(snapshot + '.meta'):
      break
    pidx += 1

  snapshot_model = snapshot + '.model'

  meta_f = open(snapshot + '.meta', 'a+')
  meta_f.write("|".join([key.upper()+'-'+str(config[key]) for key in config]))
  meta_f.close()

  # load data
  reader = pickle.load(open(args.reader_file, 'rb'))
  print("reader {} loaded".format(args.reader_file))
  box_data = h5py.File(args.box_file, 'r')
  print("box feats {} loaded".format(args.box_file))
  print("predicted vocab  size {}".format(len(reader.i2w)))

  # create/load model
  print("loading model..")
  if args.resume == '':
    net = get_model(reader, config)
  else:
    net = torch.load(args.resume)
    net.config['command'] += [config['command']]
  print("model loaded.")

  task_chunk = args.model in set(
      ["chunker", "chal", "pipelinecrf", "pipelineilp"])
  task_align = args.model in set(
      ["aligner", "ban", "chal", "pipelinecrf", "pipelineilp"])
  print("Tasks: chunk {} | align {}".format(task_chunk, task_align))

  val_score = evaluate(
      net, reader.data['val'], box_data, task_chunk=task_chunk, task_align=task_align, target_metric="alignment" if task_align else "chunking", use_gt=args.use_gt, use_predicted=args.use_predicted, details=args.details)
  print("\nVal Score: {:3.4f}".format(val_score))


if __name__ == '__main__':
  train()
