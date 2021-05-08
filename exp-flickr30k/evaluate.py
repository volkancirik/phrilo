from __future__ import absolute_import

#import setGPU
import os
import pickle
import sys
from random import shuffle, seed
import h5py
import numpy as np
from tqdm import tqdm
sys.path.append(".")
sys.path.append('..')
sys.path.append('../ban_vqa/')
sys.path.append('./lm_lstm_crf/')
import torch.optim as optim
import torch
import torch.nn as nn

from util.reader import Reader
from util.model_utils import get_box_feats, makevar, get_n_params, f1_score, f1_score_grounded, alignment_score
from util.arguments import get_flickr30k_train
from models.get_model import get_model

def countgt_chunksize(gt_chunks_arr):
  sum =0
  minlen=1000
  maxlen =0
  print(type(gt_chunks_arr))
  n = len(gt_chunks_arr)
  print(n)
  ct =0
  for gt_chunks in gt_chunks_arr:
    gt_chunks =  gt_chunks[0]
    for tuple in gt_chunks: 
      start = tuple[0]
      end = tuple[1]
      length = end-start+1
      minlen = min(length, minlen)
      maxlen = max(length,maxlen)
      sum += length
      ct +=1
  avglen = float(sum)/float(ct)
  print("maxlen ", maxlen)
  print("minlen ", minlen)
  print("avglen", avglen)





def gestStats(split,box_data):
  pbar = tqdm(list(range(len(split[0]))))
  sentence_map={}
  for i in pbar:
    words = split[0][i]
    gt_alignments = split[3][i]
    sentence= " ".join(words)
    sentence_map[sentence]= gt_alignments
  file_to_write=open("devset_stats_phrilo_old.pkl","wb")
  pickle.dump(sentence_map, file_to_write)
  print("NUm caption in dev ", len(sentence_map))


def evaluate(net, split, box_data,args,
             task_chunk=False, task_align=False, target_metric="chunking",
             use_gt=False, use_predicted=False, details=False):
  print(type(split[3]))
  print(len(split[3]))
  #gt_chunks, gt_alignments, pos_tags = split[3]
  countgt_chunksize(split[3][:])
  pbar = tqdm(list(range(len(split[0]))))
  net.eval()

  f1_chunk = 0.0
  f1_chunk_grounded =0.0
  f1_chunk_wordlevel=0.0
  count_chunk = 1.0
  hit_align = 0.0
  count_align = 1.0
  output_chunks_dict={}

  for i in pbar:
    words = split[0][i]
    pred_boxes, gt_boxes = split[1][i]
    boxes = gt_boxes if use_gt else pred_boxes
    box_type = 'gt' if use_gt else 'pred'

    gt_chunks, gt_alignments, pos_tags = split[3][i]
    
    resobj={}
    resobj["sentence"] = words

    resobj["gt_chunks"]=  gt_chunks 
    #print("gt_chunks",gt_chunks)
    #print("and preds")
    if all([al == [] for al in gt_alignments[box_type]]):
      continue

    box_reps = get_box_feats(
        boxes, box_data, box_type)
    _, pred_chunks, pred_alignments = net.forward(
        words, pos_tags, box_reps, gt_chunks, gt_alignments[box_type], debug=False)
    #print("pred_chunks",pred_chunks)
    if use_predicted:
      for jj, (ph, al) in enumerate(zip(gt_chunks, gt_alignments[box_type])):
        if al == []:
          gt_alignments[box_type][jj] = [box_reps[0].size(0)]

    gt_al_tuples = []
    for ch, al in zip(gt_chunks, gt_alignments[box_type]):
      if al:
        gt_al_tuples.append(list(ch)+al)
    pred_al_tuples = [list(ch)+pred_alignments[ch] for ch in pred_alignments]
    resobj["pred_chunks"]= pred_chunks
    #print("pred_chunks",pred_chunks )

    if task_chunk and pred_chunks:
      f1_chunk += f1_score(pred_chunks, gt_chunks)
      f1_chunk_grounded += f1_score_grounded(pred_chunks, gt_chunks, gt_alignments[box_type])
      if(args.wordwise_chunkingeval): 
        pred_chunks =[tuple([i,i]) for i in range(len(words))]
        f1_chunk_wordlevel += f1_score(pred_chunks, gt_chunks)
      count_chunk += 1
    if task_align and gt_al_tuples:
      sentence = " ".join(words)
      target_sentence= ["a boy in a red shirt and a boy in a yellow shirt are jumping on a trampoline outside ."]
      target_sentence.append("two religious men glancing off to their right while standing in front of a church .")
      target_sentence.append("a closeup shot of a long-haired man playing a red electric guitar .")
      target_sentence.append("several men and women stand in front of a yellow table that is covered in piles of potatoes .")
      target_sentence.append("the two tan colored dogs are in a field , and one is jumping in the air .")
      if(sentence in target_sentence): 
        print("yay")
      hit, tot = alignment_score(
          pred_al_tuples, gt_al_tuples, use_predicted=box_reps[0].size(0))
      hit_align += hit
      count_align += tot

    pbar_str = "chunk {:3.4f}, chunk_grounded {:3.4f} chunking_wordwise{:3.4f} alignment {:3.4f}".format(
        f1_chunk / count_chunk,f1_chunk_grounded / count_chunk, f1_chunk_wordlevel/ count_chunk, hit_align / count_align)
    if details:
      pbar_str += '>>>> {} ||| {}'.format(pred_al_tuples, gt_al_tuples)
    pbar.set_description(pbar_str)
    output_chunks_dict[i]=resobj
    
    
  
  file_to_write = open("chunks_output.pickle", "wb")
  pickle.dump(output_chunks_dict, file_to_write)
  #print(pbar_str)
  metrics = {"chunking":  f1_chunk/count_chunk,
             "chunking_grounded" : f1_chunk_grounded/count_chunk,
             "alignment": hit_align/count_align}
  net.train()
  return metrics


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
  gestStats(reader.data['val'], box_data)

  # val_score = evaluate(
  #     net, reader.data['val'], box_data,args, task_chunk=task_chunk, task_align=task_align, target_metric="alignment" if task_align else "chunking", use_gt=args.use_gt, use_predicted=args.use_predicted, details=args.details)
  # print("\nVal Scores: chunking:{:3.4f}, chunking_ggrounded:{:3.4f}, allignment:{:3.4f} ".format(val_score["chunking"],val_score["chunking_grounded"],val_score["alignment"]))
  

if __name__ == '__main__':
  train()
