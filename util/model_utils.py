from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.nn.init import xavier_uniform_
from functools import reduce


def list2str(s):
  return ",".join([str(i) for i in s])


def encode_box(modulelist, W, box, ph_dim, nonlinearity):
  layer_in = W(box)
  for i, ff in enumerate(modulelist):
    if nonlinearity:
      layer_in = nonlinearity(ff(layer_in))
    else:
      layer_in = ff(layer_in)
  return layer_in.resize(1, ph_dim)


def score_box(modulelist, layer_in, nonlinearity=None):
  for i, ff in enumerate(modulelist):
    if nonlinearity:
      layer_in = nonlinearity(ff(layer_in))
    else:
      layer_in = ff(layer_in)
  return layer_in


def encode_phrase(modulelist, layer_in, layer, nonlinearity, dim=1):
  for i, ff in enumerate(modulelist):
    if nonlinearity:
      layer_in = nonlinearity(ff(layer_in))
    else:
      layer_in = ff(layer_in)
  return layer_in.contiguous().view(1, dim)


def embed_symbols(vocab, emb, symbols, start, end):
  embedded = []
  for idx in range(start, end):
    symb = symbols[idx]
    idx = makevar(vocab.get(symb, 0))
    embedded.append(emb(idx))
  return embedded


def get_context_emmbedding(tok_list, start, end, vocab, emb, window_size=2):
  start += window_size
  end += window_size
  tok_list = ["<go>"]*window_size + tok_list + ["<eos>"]*window_size

  toks_left = []
  toks_right = []
  for i in range(window_size):
    toks_left.append(tok_list[start - window_size + i])
    toks_right.append(tok_list[end + i])

  emb_left_seq = embed_symbols(vocab, emb, toks_left, 0, len(toks_left))
  emb_right_seq = embed_symbols(vocab, emb, toks_left, 0, len(toks_left))

  emb_left = torch.sum(torch.cat(emb_left_seq, 0), 0).view(1, -1)
  emb_right = torch.sum(torch.cat(emb_right_seq, 0), 0).view(1, -1)

  emb_ctx = torch.cat([emb_left, emb_right], 1)

  return emb_ctx


def get_context_vector(tok_list, start, end, vocab, window_size=2):
  start += window_size
  end += window_size
  tok_list = ["<go>"]*window_size + tok_list + ["<eos>"]*window_size
  ctx_left = np.zeros((1, len(vocab)))
  ctx_right = np.zeros((1, len(vocab)))

  toks_left = []
  toks_right = []
  for i in range(window_size):
    tok_left = vocab.get(tok_list[start - window_size + i], vocab["<unk>"])
    toks_left.append(tok_list[start - window_size + i])
    ctx_left[0, tok_left] += 1.0
    tok_right = vocab.get(tok_list[end + i], vocab["<unk>"])
    toks_right.append(tok_list[end + i])
    ctx_right[0, tok_right] += 1.0

  return makevar(ctx_left, numpy_var=True).float(), makevar(ctx_right, numpy_var=True).float()


def phrase_positional_encoding(start, end, emb_dim):
  position_enc = np.array([
      [pos / np.power(10000, 2 * (j // 2) / emb_dim) for j in range(emb_dim)]
      if pos != 0 else np.zeros(emb_dim) for pos in range(start, end+1)])
  position_enc[:, 0:: 2] = np.sin(position_enc[:, 0:: 2])
  # apply cos on 1st,3rd,5th...emb_dim
  position_enc[:, 1:: 2] = np.cos(position_enc[:, 1:: 2])

  position_enc = np.sum(position_enc, 0).reshape(1, emb_dim)
  return makevar(position_enc, numpy_var=True).float()


def chunk_positional_encoding(start, end, emb_dim):
  enc_idx = makevar(
      np.repeat(np.array([end-start, start, end]), emb_dim)).float()
  enc_pos = phrase_positional_encoding(start, end, emb_dim)

  return torch.cat([enc_idx, enc_pos], 1)


def init_forget(l):
  for names in l._all_weights:
    for name in filter(lambda n: "bias" in n,  names):
      bias = getattr(l, name)
      n = bias.size(0)
      start, end = n//4, n//2
      bias.data[start: end].fill_(1.)


def get_n_params(model, verbose=False):
  n = 0
  for param in model.parameters():
    if param.requires_grad:
      if verbose:
        print(param.size())
      n += reduce(lambda x, y: x*y, param.size())
  return n


def get_box_feats(boxes, CNN,
                  box_type="pred", cuda=True,
                  fc7='fc7', spat='spat', smax="smax", attr="attr",
                  convert_fc7=True, convert_spat=True,
                  convert_smax=False, convert_attr=False, volatile=False):
  feats_fc7 = []
  feats_spat = []
  feats_smax = []
  feats_attr = []

  for box in boxes:
    feats_fc7.append(CNN[box_type + "_" + fc7][box])
    feats_spat.append(CNN[box_type + "_" + spat][box])

  if convert_fc7:
    feats_fc7 = makevar(np.stack(tuple(feats_fc7)),
                        numpy_var=True, cuda=cuda, volatile=volatile)
  else:
    feats_fc7 = np.stack(tuple(feats_fc7))
  if convert_spat:
    feats_spat = makevar(np.stack(tuple(feats_spat)),
                         numpy_var=True, cuda=cuda, volatile=volatile)
  else:
    feats_spat = np.stack(tuple(feats_spat))

  return feats_fc7, feats_spat


def printVec(vec, logprob=False, width=3):
  r = ""
  try:
    if logprob == True:
      vec = torch.exp(vec)
    r = " ".join(["{:.{width}f}".format(float(v), width=width)
                  for v in vec.data[0]])
  except:
    print("ERR:", vec)
    pass
  return r


def makevar(x, numpy_var=False, cuda=True, volatile=False):
  '''
  makes a variable from x. use numpy_var=True if x is already a numpy var
  '''
  if numpy_var:
    v = torch.from_numpy(x).float()
  else:
    v = torch.from_numpy(np.array([x]))
  if cuda:
    return Variable(v.cuda(), volatile=volatile)
  return Variable(v, volatile=volatile)


def zeros(dim, cuda=True):
  '''
  zero variable of dimension dim
  '''
  v = torch.zeros(dim)
  if cuda:
    return Variable(v.cuda())
  return Variable(v)


def weight_init(m):
  if len(m.size()) == 2 and m.requires_grad:
    xavier_uniform_(m)


def f1_score(predicted, ground_truth):
  correct = 0.0

  if isinstance(ground_truth[0], list):
    ground_truth = [tuple(gt) for gt in ground_truth]

  if isinstance(predicted[0], list):
    predicted = [tuple(pred) for pred in predicted]

  ground_truth = set(ground_truth)
  for chunk in predicted:
    if chunk in ground_truth:
      correct += 1.0
  precision = correct / len(predicted)
  recall = correct / len(ground_truth)
  if not precision+recall:
    return 0.0
  f_one = 2*(precision*recall) / (precision + recall)
  return f_one


def alignment_score(predicted, ground_truth, use_predicted=0):

  gt = defaultdict(set)

  for al in ground_truth:
    if use_predicted > 0 and al[2:] == [use_predicted]:
      continue
    gt[(al[0], al[1])] = set(al[2:])

  total = len(gt)*1.0
  hit = 0.0
  for al in predicted:
    if (al[0], al[1]) not in gt:
      continue
    if len(set(al[2:]).intersection(gt[(al[0], al[1])])) > 0 and len(set(al[2:])):
      hit += 1.0

  return hit, total


def run_tests():
  true_chunks = [(0, 1), (2, 2), (3, 4), (5, 6)]
  pred_chunks = [(0, 2), (3, 4), (5, 5), (6, 6)]

  gt_alignments = [[1], [50], [3], [50]]

  print(f1_score(pred_chunks, true_chunks))

  gt_al_tuples = []
  pred_alignments = {(0, 1): [1], (2, 2): [50], (3, 4): [2], (5, 6): [50]}
  for ch, al in zip(true_chunks, gt_alignments):
    if al != []:
      gt_al_tuples.append(list(ch)+al)
  pred_al_tuples = [list(ch)+pred_alignments[ch]
                    for ch in pred_alignments]

  print(alignment_score(pred_al_tuples, gt_al_tuples))
  print(alignment_score(pred_al_tuples, gt_al_tuples, use_predicted=50))


def pretty_size(size):
  """Pretty prints a torch.Size object"""
  assert(isinstance(size, torch.Size))
  return " × ".join(map(str, size))


def dump_tensors(gpu_only=True):
  """Prints a list of the Tensors being tracked by the garbage collector."""
  import gc
  total_size = 0
  for obj in gc.get_objects():
    try:
      if torch.is_tensor(obj):
        if not gpu_only or obj.is_cuda:
          print("%s:%s%s %s" % (type(obj).__name__,
                                " GPU" if obj.is_cuda else "",
                                " pinned" if obj.is_pinned else "",
                                pretty_size(obj.size())))
          total_size += obj.numel()
      elif hasattr(obj, "data") and torch.is_tensor(obj.data):
        if not gpu_only or obj.is_cuda:
          print("%s → %s:%s%s%s%s %s" % (type(obj).__name__,
                                         type(obj.data).__name__,
                                         " GPU" if obj.is_cuda else "",
                                         " pinned" if obj.data.is_pinned else "",
                                         " grad" if obj.requires_grad else "",
                                         " volatile" if obj.volatile else "",
                                         pretty_size(obj.data.size())))
          total_size += obj.data.numel()
    except Exception as e:
      pass
  print("Total size:", total_size)


if __name__ == '__main__':
  run_tests()
