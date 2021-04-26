import paths
# import setGPU
import os
import pickle
import sys
import sys
sys.path.append('..')
sys.path.append('../ban_vqa/')
sys.path.append('./lm_lstm_crf/')
from random import shuffle, seed
import h5py
import numpy as np
from tqdm import tqdm

import torch.optim as optim
import torch
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("exp/logs/crf/")
import torch.nn as nn
sys.path.append(".")
print(sys.path)
#import util
from util.reader import Reader
from util.model_utils import get_box_feats, makevar, get_n_params, f1_score, alignment_score
from util.arguments import get_flickr30k_train
from models.get_model import get_model
from util.model_utils import dump_tensors

def evaluate(net, split, box_data, task_chunk=False, task_align=False, target_metric='chunking', use_gt=False, use_predicted=False, details=False, max_length=50):

  pbar = tqdm(list(range(len(split[0]))))
  net.eval()

  f1_chunk = 0.0
  count_chunk = 1.0
  hit_align = 0.0
  count_align = 1.0
  torch.cuda.empty_cache()
  output_chunks_dict={}

  for i in pbar:
    
    box_type = 'gt' if use_gt else 'pred'
    gt_chunks, gt_alignments, pos_tags = split[3][i]
    words = split[0][i]
    output_chunks_dict["id"]=i
    resobj={}
    resobj["sentence"] = words
    resobj["gt_chunks"]=  gt_chunks 
   
    if all([al == [] for al in gt_alignments[box_type]]):
      continue
    if (task_align and
            len(words) > max_length):
      continue

    pred_boxes, gt_boxes = split[1][i]
    boxes = gt_boxes if use_gt else pred_boxes

    with torch.no_grad():
      box_reps = get_box_feats(
          boxes, box_data, box_type, volatile=True)
      loss, pred_chunks, pred_alignments = net.forward(
          words, pos_tags, box_reps, gt_chunks, gt_alignments[box_type], debug=False)
      n_boxes = int(box_reps[0].size(0))
      torch.cuda.empty_cache()

      if use_predicted:
        for jj, (ph, al) in enumerate(zip(gt_chunks, gt_alignments[box_type])):
          if al == []:
            gt_alignments[box_type][jj] = [n_boxes]

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
            pred_al_tuples, gt_al_tuples, use_predicted=n_boxes)
        hit_align += hit
        count_align += tot

      pbar_str = 'L{:2d} chunk {:3.4f} alignment {:3.4f} hit_groundings {:6.4f} total_groundings {:6.4f}'.format(len(words),
                                                                 f1_chunk / count_chunk, hit_align / count_align, hit_align, count_align)
      if details:
        pbar_str += '>>>> {} ||| {}'.format(pred_al_tuples, gt_al_tuples)

      pbar.set_description(pbar_str)

      output_chunks_dict["result"] = resobj
  
  file_to_write = open("chunks_output.pickle", "wb")
  pickle.dump(output_chunks_dict, file_to_write)

  metrics = {'chunking':  f1_chunk/count_chunk,
             'alignment': hit_align/count_align}
  net.train()
  return metrics[target_metric]


def train():
  seed(2)

  args = get_flickr30k_train()
  config = vars(args)
  config['command'] = [' '.join(sys.argv[1:])]

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
  meta_f.write('|'.join([key.upper()+'-'+str(config[key]) for key in config]))
  meta_f.close()

  # load data
  reader = pickle.load(open(args.reader_file, 'rb'))
  print('reader {} loaded'.format(args.reader_file))
  box_data = h5py.File(args.box_file, 'r')
  print('box feats {} loaded'.format(args.box_file))
  print('predicted vocab  size {}'.format(len(reader.i2w)))

  # create/load model
  print('loading model..')
  if args.resume == '':
    net = get_model(reader, config)

  else:
    net = torch.load(args.resume)
    #net.config['command'] += [config['command']]
  print("printing model")
  print(net)
  print('model loaded.')

  if config['optim'] == 'sgd':
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters(
    )), lr=config['lr'], weight_decay=config['w_decay'], momentum=0.95)
  else:
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters(
    )), lr=config['lr'], weight_decay=config['w_decay'])

  print('experiment configuration')
  print('='*30)
  for key in config:
    print('{} : {}'.format(key, config[key]))
  print('='*30)
  # print('starting training for:\nsnapshot {}\n#of parameters {}'.format(
  #     snapshot, get_n_params(net)))

  best_val = 0.0
  split = reader.data['trn']
  print("split type", type(split))

  net.train()
  device = torch.device('cuda')
  print("device ",device)
  net.to(device)

  task_chunk = args.model in set(['chunker', 'chal','pipelinecrf'])
  task_align = args.model in set(
      ['aligner', 'chal', 'ban'])
  print('Tasks: chunk {} | align {}'.format(task_chunk, task_align))
  printflag=1
  for epoch in range(args.epochs):
    print("epoch ",epoch)

    length = args.val_freq if args.val_freq > 0 else len(split[0])

    total_length = len(split[0])
    #total_length=100
    print(total_length,length,epoch)
    indexes = range(total_length)
    shuf_idx = list(range(total_length))
    shuffle(shuf_idx)

    for update in range(int(total_length/length) + int(total_length % length > 0)):
      if args.verbose:
        pbar = tqdm(
            indexes[update*length:min((update+1)*length, total_length)])
      else:
        pbar = indexes[update*length:min((update+1)*length, total_length)]
      closs = 0.0
      cinst = 0.0

      f1_chunk = 0.0
      count_chunk = 1.0
      hit_align = 0.0
      count_align = 1.0
      hit_align_predicted = 0.0
      count_align_predicted = 0.0

      for i in pbar:
        idx = shuf_idx[i]
        if(printflag):
          print(split[0][idx])
          print("next")
          print(split[1][idx])
          print("next")
          print(split[2][idx])
          print("next")
          print(split[3][idx])
          printflag =0
        words = split[0][idx]
        pred_boxes, gt_boxes = split[1][idx]

        boxes = gt_boxes if args.use_gt else pred_boxes
        box_type = 'gt' if args.use_gt else 'pred'

        gt_chunks, gt_alignments, pos_tags = split[3][idx]
        if (not task_align and len(words) > args.max_length) or all([al == [] for al in gt_alignments[box_type]]):
          continue
        max_len = epoch*5
        if task_align and task_chunk and len(words) > 10 + max_len:
          continue

        box_reps = get_box_feats(
            boxes, box_data, box_type)
        n_boxes = int(box_reps[0].size(0))

        if args.use_predicted:
          for jj, (ph, al) in enumerate(zip(gt_chunks, gt_alignments[box_type])):
            if al == []:
              gt_alignments[box_type][jj] = [box_reps[0].size(0)]

        loss, pred_chunks, pred_alignments = net.forward(
            words, pos_tags, box_reps, gt_chunks, gt_alignments[box_type], debug=False)
        #print("chunks",pred_chunks)

        gt_al_tuples = []
        for ch, al in zip(gt_chunks, gt_alignments[box_type]):
          if al != []:
            gt_al_tuples.append(list(ch)+al)

        if task_align and task_chunk:
          pred_al_tuples = [list(ch)+pred_alignments[ch]
                            for ch in pred_chunks]
        elif task_align:
          pred_al_tuples = [list(ch)+pred_alignments[ch]
                            for ch in pred_alignments]

        if task_chunk and pred_chunks != []:
          f1_chunk += f1_score(pred_chunks, gt_chunks)
          count_chunk += 1
        if task_align and gt_al_tuples != []:
          hit, tot = alignment_score(pred_al_tuples, gt_al_tuples)
          hit_align += hit
          count_align += tot
          if args.use_predicted:
            hit, tot = alignment_score(
                pred_al_tuples, gt_al_tuples, use_predicted=n_boxes)
            hit_align_predicted += hit
            count_align_predicted += tot

        if isinstance(loss, int):
          continue
        closs += float(loss.data.item())
        cinst += 1
        trn_loss = closs / cinst

        if args.verbose:
          if task_align:
            pbar_str = 'E:{} best_val:{:3.4f} loss {:5.4f} L{:2d} ch_acc {:3.4f} al_acc {:3.4f}'.format(
                epoch+1, best_val, trn_loss, len(words), f1_chunk / count_chunk, hit_align / count_align)
            if args.use_predicted:
              pbar_str += ' gr_al_acc {:3.4f}'.format(
                  hit_align_predicted / count_align_predicted)
            if args.details:
              pbar_str += '>>>> {} ||| {}'.format(pred_al_tuples, gt_al_tuples)
          else:
            pbar_str = 'E:{} best_val:{:3.4f} loss {:5.4f} ch_acc {:3.4f}'.format(
                epoch+1, best_val, trn_loss, f1_chunk / count_chunk)
            if(i%5000 ==0):
              writer.add_scalar("Loss/train", trn_loss, update*length+i)
              writer.add_scalar("ch_acc f1/train", f1_chunk / count_chunk, update*length+i)
              print(pbar_str)
              writer.flush()
          pbar.set_description(pbar_str)
        

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), config['clip'])
        optimizer.step()
        torch.cuda.empty_cache()


      val_score = evaluate(
          net, reader.data['val'], box_data, task_chunk=task_chunk, task_align=task_align, target_metric='alignment' if task_align else 'chunking', use_gt=args.use_gt, use_predicted=args.use_predicted, details=args.details)
      writer.add_scalar("val chunk_acc", val_score, update)
      writer.flush()
      print('\nScore: {:3.4f}'.format(val_score))
      if best_val < val_score:
        torch.save(net, snapshot_model + '.best')
        best_val = val_score
        print('best model is updated with score {:3.4f} to {}'.format(
            best_val, snapshot_model + '.best'))

      config['lr'] = max(config['lr']*config['lr_decay'], config['lr_min'])
      print('lr is now {:3.4f}'.format(config['lr']))
      if config['optim'] == 'sgd':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters(
        )), lr=config['lr'], weight_decay=config['w_decay'], momentum=0.95)
      else:
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters(
        )), lr=config['lr'], weight_decay=config['w_decay'])
      torch.save(net, snapshot_model+'.EPOCH'+str(epoch))


if __name__ == '__main__':
  train()
