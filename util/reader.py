from __future__ import absolute_import

from collections import Counter, defaultdict
from random import shuffle
import os
import h5py
import sys
import codecs
import numpy as np
import pickle
import json
from copy import deepcopy

import skimage.io
import spacy
from tqdm import tqdm

from util.oracle import Phrase, parse_sentence, get_chunk_oracle, get_tokens
from util.spellchecker import Spellchecker


def get_intersection(obj1, obj2):
  left = obj1[0] if obj1[0] > obj2[0] else obj2[0]
  top = obj1[1] if obj1[1] > obj2[1] else obj2[1]
  right = obj1[2] if obj1[2] < obj2[2] else obj2[2]
  bottom = obj1[3] if obj1[3] < obj2[3] else obj2[3]
  if left > right or top > bottom:
    return [0, 0, 0, 0]
  return [left, top, right, bottom]


def calculate_iou(obj1, obj2):
  area1 = calculate_area(obj1)
  area2 = calculate_area(obj2)
  intersection = get_intersection(obj1, obj2)
  area_int = calculate_area(intersection)
  return area_int / (area1 + area2 - area_int)


def calculate_area(obj):
  return (obj[2] - obj[0]) * (obj[3] - obj[1])


def IoU(b1, b2):
  bbox1 = [b1[0], b1[1], b1[2]-b1[0], b1[3]-b1[1]]
  bbox2 = [b2[0], b2[1], b2[2]-b2[0], b2[3]-b2[1]]

  bbox_ov_x = max(bbox1[0], bbox2[0])
  bbox_ov_y = max(bbox1[1], bbox2[1])
  bbox_ov_w = min(bbox1[0] + bbox1[2] - 1, bbox2[0] +
                  bbox2[2] - 1) - bbox_ov_x + 1
  bbox_ov_h = min(bbox1[1] + bbox1[3] - 1, bbox2[1] +
                  bbox2[3] - 1) - bbox_ov_y + 1

  area1 = area_bbox(bbox1)
  area2 = area_bbox(bbox2)
  area_o = area_bbox([bbox_ov_x, bbox_ov_y, bbox_ov_w, bbox_ov_h])
  area_u = area1 + area2 - area_o
  if area_u < 0.000001:
    return 0.0
  else:
    return area_o / area_u


def get_spatial(bboxes, im_path, min_size=600,
                max_size=1000):

  im = skimage.io.imread(im_path)
  if im.ndim == 2:
    im = np.tile(im[..., np.newaxis], (1, 1, 3))
  im_h, im_w = im.shape[:2]
  scale = min(max(min_size/im_h, min_size/im_w), max_size/im_h, max_size/im_w)

  # resize and process the image
  im_h, im_w = int(scale*im_h), int(scale*im_w)

  assert(np.all(bboxes[:, 0:2] >= 0))
  assert(np.all(bboxes[:, 0] <= bboxes[:, 2]))
  assert(np.all(bboxes[:, 1] <= bboxes[:, 3]))
  assert(np.all(bboxes[:, 2] <= im_w))
  assert(np.all(bboxes[:, 3] <= im_h))
  feats = np.zeros((bboxes.shape[0], 5))
  feats[:, 0] = bboxes[:, 0] * 2.0 / im_w - 1  # x1
  feats[:, 1] = bboxes[:, 1] * 2.0 / im_h - 1  # y1
  feats[:, 2] = bboxes[:, 2] * 2.0 / im_w - 1  # x2
  feats[:, 3] = bboxes[:, 3] * 2.0 / im_h - 1  # y2
  feats[:, 4] = (feats[:, 2] - feats[:, 0]) * (feats[:, 3] - feats[:, 1])  # S
  return feats


def area_bbox(bbox):
  '''Return the area of a bounding box.'''
  if bbox[2] <= 0 or bbox[3] <= 0:
    return 0.0
  return float(bbox[2]) * float(bbox[3])


def get_phrase2box(imdb, unify_multiple_boxes=False):
  phrase2box = defaultdict(list)
  minbox = 1000
  maxd1 = 0
  maxd2 = 0

  for ins in imdb:
    for reg in ins['regions']:
      bbox = reg[0]

      w = (bbox[2] - bbox[0])*1.0
      h = (bbox[3] - bbox[1])*1.0
      size = w*h
      if maxd1 < w/h:
        maxd1 = w/h
      if maxd2 < h/w:
        maxd2 = h/w
      if size < minbox:
        minbox = size
      for phrase in reg[1]:
        phrase2box[phrase].append(reg[0])
  if unify_multiple_boxes:
    for phrase in phrase2box:
      if len(phrase2box[phrase]) > 1:
        maxx = -1
        maxy = -1
        minx = 10000
        miny = 10000
        for box in phrase2box[phrase]:
          minx = min(minx, box[0])
          miny = min(miny, box[1])
          maxx = max(maxx, box[2])
          maxy = max(maxy, box[3])
        phrase2box[phrase] = [[minx, miny, maxx, maxx, maxy]]
  return phrase2box


def get_img2fbox(box_data, imgid2idx):
  img2fbox = {}
  idx = 0

  split_names = ['train', 'val', 'test']
  for split in split_names:
    for img_id in imgid2idx[split]:

      idx = imgid2idx[split][img_id]
      img_id = str(img_id)

      start = box_data[split]['pos_boxes'][idx][0]
      end = box_data[split]['pos_boxes'][idx][1]
      img2fbox[img_id] = box_data[split]['image_bb'][start:end]
  return img2fbox


def get_hits(imdb, img2fbox, phrase2box):
  pbar = tqdm(range(len(imdb)))
  phrase2hit = defaultdict(list)
  hit = 0.0
  total_boxes = 0.0
  nbox = []
  skipped = 0
  phrase2imgid = {}
  for i in pbar:
    ins = imdb[i]
    imgid = ins['im_path'].split('/')[-1][:-4]

    if imgid not in img2fbox:
      skipped += 1
      continue

    candidates = img2fbox[imgid]
    nbox.append(len(candidates))
    phrases = set()
    for reg in ins['regions']:
      for phrase in reg[1]:
        phrases.add(phrase)

    for phrase in phrases:
      hit_list = []
      for ii, c in enumerate(candidates):
        max_score = 0.0
        for src_box in phrase2box[phrase]:
          max_score = max(max_score, calculate_iou(src_box, c))
        if max_score >= 0.5:
          #          hit_list.append(tuple(c))
          hit_list.append(ii)
      if len(hit_list) >= 1:
        hit += 1.0
      total_boxes += 1.0
      phrase2hit[imgid + '_' + phrase] = hit_list
      if phrase in phrase2imgid:
        print('\nERROR!!!! phrase was there',
              phrase, phrase2imgid[phrase], imgid)
        quit()
      phrase2imgid[imgid + '_' + phrase] = int(imgid)

    pbar.set_description('Upperbound {:5.4f} Mean # of boxes {:5.4f} Min Box {:2d} Max Box {:2d} skipped {:2d}'.format(
        hit/total_boxes, sum(nbox) / float(len(nbox)), min(nbox), max(nbox), skipped))
  return phrase2hit, phrase2imgid


def entity2boxid(regions, imgid, phrase2hit={}, phrase2imgid={}, imgid2idx={}, box_data={}):
  e2box = defaultdict(set)
  box2e = defaultdict(set)

  for i, region in enumerate(regions):
    if len(region) == 3:
      bbox, ann_id, sentences = region
    elif len(region) == 4:
      bbox, ann_id, sentences, _ = region
    else:
      raise NotImplementedError()

    if phrase2hit == {}:
      for ann in ann_id:
        e2box[ann].add(i)
        box2e[i].add(ann)
    else:
      for ann in ann_id:
        for hit in phrase2hit[imgid + '_' + ann]:
          e2box[ann].add(hit)
          box2e[hit].add(ann)

  return {'e2box': e2box, 'box2e': box2e}


class Reader:
  def __init__(self):
    self.w2i = {}
    self.i2w = {}
    self.a2i = {}
    self.i2a = {}
    self.data = {}
    self.tuple = {}

  def get_vocabs(self, wordvec_file, phrases, sentences, reduce_vocab=0):
    wordvec = {line.strip().split()[0].lower(): [float(
        n) for n in line.strip().split()[1:]] for line in open(wordvec_file)}
    self.i2w = wordvec.keys()
    self.w2i = {w: i for i, w in enumerate(self.i2w)}
    self.i2a = ['unk', 'gen', 'reduce', 'nt_unk']
    self.i2nt = ['nt_unk']
    self.a2i = {w: i for i, w in enumerate(self.i2a)}
    self.nt2i = {w: i for i, w in enumerate(self.i2nt)}

    if reduce_vocab > 0:
      count = defaultdict(int)
      for sentence in sentences:
        for token in sentence:
          count[token] += 1
      new_voc = []
      for w in count:
        if count[w] >= reduce_vocab and w in self.w2i:
          new_voc += [w]

      self.i2w = ['<unk>', '<go>', '<eos>', '<s>', '</s>'] + new_voc
      self.w2i = {w: i for i, w in enumerate(self.i2w)}
      print('vocabulary size was {} now {}'.format(
          len(wordvec.keys()), len(self.i2w)))

    ntok = 0.0
    miss = 0.0
    for sentence in sentences:
      for tok in sentence:
        if tok not in self.w2i:
          miss += 1
        ntok += 1

    vectors = []
    for w in self.i2w:
      vectors.append(wordvec[w])

    print('wordvec vocab with coverage {}\n |w2i|={}'.format(
        1 - (miss/ntok), len(self.i2w)))
    self.vectors = np.stack(tuple(vectors))

  def prepare_flickr30k_data(self, ban_path, imdb_file, vgg_file, sentence_root, wordvec_file, split_file, whitelist_file, features_out_file,
                             dump, reduce_vocab=0):

    split_names = ['train', 'val', 'test']
    box_data = {}
    imgid2idx = {}

    verification = {}
    for split in split_names:
      box_data[split] = h5py.File(os.path.join(ban_path, split + '.hdf5'), 'r')
      print(os.path.join(ban_path, split + '.hdf5'), 'loaded.')
      imgid2idx[split] = pickle.load(
          open(os.path.join(ban_path, split + '_imgid2idx.pkl'), 'rb'))
      print(os.path.join(ban_path, split + '_imgid2idx.pkl'), 'loaded.')

      verification[split] = pickle.load(
          open(os.path.join(ban_path, split + '.verification.pkl'), 'rb'))

    imdb = np.load(imdb_file,  encoding='latin1')[()]
    print('loaded imdb {}!'.format(imdb_file))

    phrase2box = get_phrase2box(imdb)
    print('phrase -> box created.')

    img2fbox = get_img2fbox(box_data, imgid2idx)
    print('img -> box created.')

    phrase2hit, phrase2imgid = get_hits(imdb, img2fbox, phrase2box)
    print('phrase -> hits created.')

    error_src = defaultdict(list)
    error_trg = defaultdict(list)
    for split in split_names:
      for phrase in verification[split]['phrase2box']:

        src_ban = verification[split]['phrase2src'][phrase]
        src_our = phrase2box[str(phrase)]

        ban_set = set()
        for box in src_ban:
          ban_set.add(tuple(box))
        our_set = set()
        for box in src_our:
          our_set.add(tuple(box))

        # if phrase == '81000' or phrase == 81000 or phrase == 80923 or phrase == '80923':
        #   imgid = phrase2imgid[str(phrase)]
        #   ban_box_idx = imgid2idx[split][imgid]
        #   box_start = box_data[split]['pos_boxes'][ban_box_idx][0]
        #   box_end = box_data[split]['pos_boxes'][ban_box_idx][1]
        #   print(phrase, phrase2hit[str(imgid) + '_' + str(phrase)])
        #   print('imgid | split', imgid, split)
        #   print('st end # of boxes', box_start, box_end, box_end - box_start)
        #   print('target', trg_ban, trg_our)
        #   print('_'*10)

        if our_set.intersection(ban_set) == 0:
          error_src[split].append((split, phrase, src_ban, src_our))
          continue
        # trg_ban = sorted(verification[split]['phrase2target'][phrase])
        # trg_our = phrase2hit[str(phrase)]

        # if trg_ban != trg_our:
        #   error_trg[split].append((split, phrase, trg_ban, trg_our))

    # for split in split_names:
    #   print('='*20)
    #   print('SPLIT', split)
    #   print('len(error_src)', len(error_src[split]))
    #   print('len(error_trg)', len(error_trg[split]))
    #   print('_'*10)
    #   print('sample src', error_src[split][0]
    #         if len(error_src[split]) > 0 else 'NONE')
    #   print('sample trg', error_trg[split][0]
    #         if len(error_trg[split]) > 0 else 'NONE')
    #   print('')

    self.n_tuple = 4
    # create mapping file_id->split
    id2split = {}
    with open(split_file, 'r') as f:
      for line in f:
        l = line.strip().split()
        id2split[l[0]] = l[1]

    # create set for blacklisted sentences
    whitelist = set()
    with open(whitelist_file, 'r') as f:
      for line in f:
        l = line.strip()
        whitelist.add(l)

    n_boxes_pred = 0
    n_boxes_gt = 0

    feats_pred_fc7 = []
    feats_pred_spat = []
    feats_pred_boxes = []

    feats_gt_fc7 = []
    feats_gt_spat = []
    feats_gt_boxes = []

    VGG = h5py.File(vgg_file, 'r')
    vgg_fnames = list(VGG['filenames'])
    fnames_idx = {fname.decode('UTF-8'): i for i,
                  fname in enumerate(vgg_fnames)}

    Xtrn_idx, Xtrn_raw, Xtrn_txt, Xtrn_box, Xtrn_img, Xtrn_map, Ttrn = [], [], [], [], [], [], []
    Xval_idx, Xval_raw, Xval_txt, Xval_box, Xval_img, Xval_map, Tval = [], [], [], [], [], [], []
    Xtst_idx, Xtst_raw, Xtst_txt, Xtst_box, Xtst_img, Xtst_map, Ttst = [], [], [], [], [], [], []

    split2idx = {'tst': Xtst_idx, 'trn': Xtrn_idx, 'val': Xval_idx}
    split2raw = {'tst': Xtst_raw, 'trn': Xtrn_raw, 'val': Xval_raw}
    split2txt = {'tst': Xtst_txt, 'trn': Xtrn_txt, 'val': Xval_txt}
    split2box = {'tst': Xtst_box, 'trn': Xtrn_box, 'val': Xval_box}
    split2img = {'tst': Xtst_img, 'trn': Xtrn_img, 'val': Xval_img}
    split2map = {'tst': Xtst_map, 'trn': Xtrn_map, 'val': Xval_map}
    split2gld = {'tst': Ttst,     'trn': Ttrn,     'val': Tval}
    nlp = spacy.load('en')

    print('preprocessing..')
    split2ban = {'trn': 'train', 'val': 'val', 'tst': 'test'}
    ban2split = {'train': 'trn', 'val': 'val', 'test': 'tst'}
    missing = []
    wordvec = {line.strip().split()[0].lower(): [float(
        n) for n in line.strip().split()[1:]] for line in open(wordvec_file)}
    spellcheck = Spellchecker(wordvec.keys())

    self.box2idx = {'unk': 0}
    self.idx2box = {0: 'unk'}
    n_box_type = 1
    box_missing = 0.0

    empty = {'pred': 0.0, 'gt': 0.0}
    total_box = {'pred': 0.0, 'gt': 0.0}

    stats = defaultdict(int)
    total_ph = 0

    imgid2instance = {}
    instance2imgid = {}
    pbar = tqdm(range(len(imdb)))
    for i in pbar:
      instance = imdb[i]
      idx = int(instance['misc']['image_id'])
      imgid2instance[idx] = i
      instance2imgid[i] = idx
    print('img -> instance and instance -> img created.')

    missing = []
    skipped = 0
    for ban_split in split_names:
      pbar = tqdm(imgid2idx[ban_split].keys())
      split = ban2split[ban_split]
      for imgid in pbar:
        if imgid not in imgid2instance:
          missing += [imgid]
          continue

        instance_idx = imgid2instance[imgid]
        instance = imdb[instance_idx]
        idx = str(imgid)

        mapping_pred = entity2boxid(
            instance['regions'], idx, phrase2hit=phrase2hit, phrase2imgid=phrase2imgid, imgid2idx=imgid2idx, box_data=box_data)

        gt_boxes = []
        for region in instance['regions']:
          gt_boxes.append(region[0])
        gt_boxes = np.array(gt_boxes)

        mapping_gt = entity2boxid(instance['regions'], idx, phrase2hit={})

        feats_gt_boxes.append(gt_boxes)
        feats_gt_fc7.append(instance['box_feats'])
        feats_gt_spat.append(get_spatial(gt_boxes, instance['im_path']))

        ban_box_idx = imgid2idx[split2ban[split]][imgid]
        box_start = box_data[split2ban[split]]['pos_boxes'][ban_box_idx][0]
        box_end = box_data[split2ban[split]]['pos_boxes'][ban_box_idx][1]

        pred_boxes = box_data[split2ban[split]]['image_bb'][box_start: box_end]
        pred_fc7 = box_data[split2ban[split]
                            ]['image_features'][box_start: box_end]
        pred_spat = box_data[split2ban[split]
                             ]['spatial_features'][box_start: box_end]

        feats_pred_fc7.append(pred_fc7)
        feats_pred_spat.append(pred_spat)
        feats_pred_boxes.append(pred_boxes)

        nfixed = 0
        for j, line in enumerate(open(os.path.join(sentence_root, idx + '.txt'))):
          if idx + '.' + str(j+1) not in whitelist:
            skipped += 1
            continue

          sentence = line.strip().lower()
          parsed_f30k = parse_sentence(sentence)
          img_idx = fnames_idx[idx]

          alignments = {'pred': [], 'gt': []}

          words = []
          chunks = []
          n_tok = 0

          for ph in parsed_f30k:
            words += ph.tokens
            if n_tok > n_tok + len(ph.tokens)-1:
              print('')
              print('i j idx', i, j, idx)
              print('sentence', sentence)
              print('ph.tokens', ph.tokens)
              print('n_tok', n_tok)
              print('words\n', words)
              print('chunks\n', chunks)
              quit()
            chunks += [(n_tok, max(n_tok, n_tok + len(ph.tokens)-1))]
            n_tok += len(ph.tokens)

            for btype, mapping in zip(['pred', 'gt'], [mapping_pred, mapping_gt]):
              entity_id = str(ph.entity)
              if ph.etype in set(['N/A', 'notvisual', 'scene', 'other']) or list(mapping['e2box'][entity_id]) == []:
                if ph.etype not in set(['N/A', 'notvisual', 'scene', 'other']) and list(mapping['e2box'][entity_id]) == []:
                  empty[btype] += 1.0
                alignments[btype].append([])
              else:
                alignments[btype].append(
                    [ii for ii in list(mapping['e2box'][entity_id])])
                total_box[btype] += 1

            if split == 'tst':
              stats[ph.etype] += 1
              total_ph += 1

          sent_tagged = nlp(' '.join(words))
          pos_tags = []
          for t in sent_tagged:
            pos_tags.append(t.tag_)

          split2raw[split].append(sentence)
          split2txt[split].append(words)
          split2box[split].append((
              [n_boxes_pred + ii for ii in range(pred_boxes.shape[0])], [n_boxes_gt + ii for ii in range(gt_boxes.shape[0])]))
          split2img[split].append((img_idx, idx))
          split2map[split].append((mapping_pred, mapping_gt))
          split2gld[split].append((chunks, alignments, pos_tags))

        n_boxes_pred += pred_boxes.shape[0]
        n_boxes_gt += gt_boxes.shape[0]

        hit_pred = (total_box['pred'])/(total_box['pred']+empty['pred'])
        hit_gt = (total_box['gt'])/(total_box['gt']+empty['gt'])
        pbar_str = '{} Ubound: pred->{:1.3f} gt->{:1.3f} MISSED{:2d} SKIPPED{:3d}'.format(
            idx, hit_pred, hit_gt, len(missing), skipped)
        pbar.set_description(pbar_str)

    stack_gt_fc7 = np.concatenate(tuple(feats_gt_fc7))
    stack_gt_boxes = np.concatenate(tuple(feats_gt_boxes))
    stack_gt_spat = np.concatenate(tuple(feats_gt_spat))

    stack_pred_fc7 = np.concatenate(tuple(feats_pred_fc7))
    stack_pred_boxes = np.concatenate(tuple(feats_pred_boxes))
    stack_pred_spat = np.concatenate(tuple(feats_pred_spat))

    print('='*10)
    print('Stats for phrases')
    print('total phrases {}'.format(total_ph))
    for etype in stats:
      print('{}  :  {} '.format(etype, stats[etype]))
    print('='*10)

    print('creating vocabs..')
    self.get_vocabs(wordvec_file, Xtrn_txt, Xtrn_txt,
                    reduce_vocab=reduce_vocab)

    print('dumping {} ground-truth and {} predicted box features to {}..'.format(
        stack_gt_boxes.shape[0], stack_pred_boxes.shape[0], features_out_file))
    h5file = h5py.File(features_out_file, 'w')

    h5file.create_dataset('gt_boxes',  data=stack_gt_boxes)
    h5file.create_dataset('gt_fc7',  data=stack_gt_fc7)
    h5file.create_dataset('gt_spat', data=stack_gt_spat)

    h5file.create_dataset('pred_boxes',  data=stack_pred_boxes)
    h5file.create_dataset('pred_fc7',  data=stack_pred_fc7)
    h5file.create_dataset('pred_spat', data=stack_pred_spat)

    h5file.close()

    self.data['trn'] = Xtrn_txt, Xtrn_box, Xtrn_map, Ttrn, Xtrn_raw, Xtrn_img, Xtrn_idx
    self.data['val'] = Xval_txt, Xval_box, Xval_map, Tval, Xval_raw, Xval_img, Xval_idx
    self.data['tst'] = Xtst_txt, Xtst_box, Xtst_map, Ttst, Xtst_raw, Xtst_img, Xtst_idx

    print('dumping dataset reader to {}..'.format(dump))
    pickle.dump(self, open(dump, 'wb'))
    print('DONE! bye.')


if __name__ == '__main__':
  r = Reader()

  r.prepare_flickr30k_data('/projects1/ban-vqa-flickr/data/flickr30k',
                           '../data/flickr30k.imdb.bottomup.npy',
                           '../data/flickr30k.vgg.h5',
                           '../data/flickr30k-entities/Sentences/',
                           '../data/wordvec.glove',
                           '../data/split.txt',
                           '../data/whitelist.txt',
                           'flickr30k.chunk_align.ban.cnn.h5',
                           'flickr30k.chunk_align.ban.dataset.pkl', reduce_vocab=5)
