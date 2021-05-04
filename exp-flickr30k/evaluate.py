from __future__ import absolute_import
import paths
#import setGPU
import os
import pickle
import sys
from random import seed
import h5py
from tqdm import tqdm

import torch

from util.reader import Reader
from util.model_utils import get_box_feats, makevar, get_n_params, f1_score, alignment_score
from util.arguments import get_flickr30k_train
from models.get_model import get_model


def evaluate(net, split, box_data,
             task_chunk=False, task_align=False, target_metric="chunking",
             use_gt=False, use_predicted=False, details=False,
             verbose=True):

  pbar = tqdm(list(range(len(split[0]))))
  net.eval()

  f1_chunk = 0.0
  count_chunk = 1.0
  hit_align = 0.0
  count_align = 1.0

  hit_align_predicted = 0.0
  count_align_predicted = 0.0
  al_score = 0
  chunking_score = 0
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
    n_boxes = int(box_reps[0].size(0))

    if use_predicted:
      for jj, (ph, al) in enumerate(zip(gt_chunks, gt_alignments[box_type])):
        if al == []:
          gt_alignments[box_type][jj] = [box_reps[0].size(0)]

    _, pred_chunks, pred_alignments = net.forward(
        words, pos_tags, box_reps, gt_chunks, gt_alignments[box_type], debug=False)

    if task_chunk:
      pred_al_tuples = [list(ch)+pred_alignments.get(ch, [])
                        for ch in pred_chunks]
    else:
      pred_al_tuples = [list(ch)+pred_alignments.get(ch, [])
                        for ch in pred_alignments]
    gt_al_tuples = []
    for ch, al in zip(gt_chunks, gt_alignments[box_type]):
      if al != []:
        gt_al_tuples.append(list(ch)+al)

    if task_chunk:
      if pred_chunks:
        f1_chunk += f1_score(pred_chunks, gt_chunks)
      count_chunk += 1

    if task_align:
      hit, tot = alignment_score(pred_al_tuples, gt_al_tuples)
      hit_align += hit
      count_align += tot
      if use_predicted:
        hit, tot = alignment_score(
            pred_al_tuples, gt_al_tuples, use_predicted=n_boxes)
        hit_align_predicted += hit
        count_align_predicted += tot

    if verbose:
      pbar_str = '[{}|{}]'.format(n_boxes, len(words))
      if task_chunk:
        chunking_score = f1_chunk / count_chunk
        pbar_str += ' ch_acc {:3.4f}'.format(chunking_score)
      if task_align:
        al_score = hit_align / count_align
        pbar_str += ' al_acc {:3.4f}'.format(al_score)
      if use_predicted:
        al_score = hit_align_predicted / count_align_predicted
        pbar_str += ' gr_al_acc {:3.4f}'.format(al_score)
      if details:
        pbar_str += '>>>C {:3.2f}|{} A {:3.2f}|{} >>>> {} ||| {}'.format(f1_chunk,
                                                                         count_chunk,
                                                                         hit_align_predicted,
                                                                         count_align_predicted,
                                                                         pred_al_tuples,
                                                                         gt_al_tuples)
      pbar.set_description(pbar_str)

  metrics = {"chunking":  chunking_score,
             "alignment": al_score}
  net.train()
  return metrics[target_metric]


def run_evaluate():

  args = get_flickr30k_train()

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

  # load data
  reader = pickle.load(open(args.reader_file, 'rb'))
  print("reader {} loaded".format(args.reader_file))
  box_data = h5py.File(args.box_file, 'r')
  print("box feats {} loaded".format(args.box_file))
  print("predicted vocab  size {}".format(len(reader.i2w)))

  # create/load model
  print("loading model..")

  if args.resume == '':
    if args.model in ['pipelineilp', 'chal']:
      config = vars(args)
      net = get_model(reader, config,
                      args=args)
      net.load_weights(reader)
    else:
      raise NotImplementedError()
  else:
    print('loading an existing model')
    meta_file = args.resume + '.meta'
    model_file = args.resume + '.model.best'

    l = [line.strip() for line in open(meta_file)][0]
    command = l.split('[')[1].split(']')[0][1: -1]
    parser = get_flickr30k_train(return_parser=True)
    model_args = parser.parse_args(command.split())
    config = vars(model_args)
    net = get_model(reader, config,
                    args=model_args)
    net.load_state_dict(torch.load(model_file))

  task_chunk = config['model'] in set(
      ["chunker", "chal", "pipelinecrf", "pipelineilp"])
  task_align = config['model'] in set(
      ["aligner", "ban", "chal", "pipelinecrf", "pipelineilp"])
  print("Tasks: chunk {} | align {}".format(task_chunk, task_align))

  val_score = evaluate(
      net, reader.data['val'], box_data, task_chunk=task_chunk, task_align=task_align, target_metric="alignment" if task_align else "chunking", use_gt=args.use_gt, use_predicted=args.use_predicted, details=args.details)
  print("\nVal Score: {:3.4f}".format(val_score))


if __name__ == '__main__':
  run_evaluate()
