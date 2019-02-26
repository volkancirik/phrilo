#from __future__ import absolute_import,print_function

from models import align, ban, banpipeline, chal, chunker
from util.model_utils import weight_init
import torch


def get_model(reader, config):

  if config['model'] == 'aligner':
    net = align.ALIGNER(config, reader.w2i, reader.i2w)
    net.We_wrd.weight.data.copy_(torch.from_numpy(reader.vectors).cuda())
  elif config['model'] == 'ban':
    net = ban.BAN(config, reader.w2i, reader.i2w)
    net.We_wrd.weight.data.copy_(torch.from_numpy(reader.vectors).cuda())
  elif config['model'] == 'banpipeline':
    net = banpipeline.BANPIPELINE(config)
  elif config['model'] == 'chunker':
    net = chunker.CHUNKER(config, reader.w2i, reader.i2w)
    net.We_wrd.weight.data.copy_(torch.from_numpy(reader.vectors).cuda())
  elif config['model'] == 'chal':
    net = chal.CHAL(config)
  else:
    raise NotImplementedError()

  if not config['finetune']:
    if config['model'] in set(['aligner', 'ban', 'chunker']):
      net.We_wrd.weight.requires_grad = False
  net.cuda()
  for param in net.parameters():
    weight_init(param)
  return net
