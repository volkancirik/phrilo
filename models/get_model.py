#from __future__ import absolute_import,print_function

from models import aligner, pipelineilp, chal, chunker,bilstm_crf
#from models import aligner, pipelineilp, chal, chunker

from util.model_utils import weight_init
import torch


def get_model(reader, config):

  if config['model'] == 'aligner':
    net = aligner.ALIGNER(config, reader.w2i, reader.i2w)
    net.We_wrd.weight.data.copy_(torch.from_numpy(reader.vectors).cuda())
  elif config['model'] == 'ban':
    net = ban.BAN(config, reader.w2i, reader.i2w)
    net.We_wrd.weight.data.copy_(torch.from_numpy(reader.vectors).cuda())
  elif config['model'] == 'pipelineilp':
    net = pipelineilp.PIPELINEILP(config)
  elif config['model'] == 'pipelinecrf':
    net = pipelinecrf.PIPELINECRF(config,reader.w2i, reader.i2w)
  elif config['model'] == 'chunker':

    net = chunker.CHUNKER(config, reader.w2i, reader.i2w)
    net.We_wrd.weight.data.copy_(torch.from_numpy(reader.vectors).cuda())
  elif config['model'] == 'chal':
    net = chal.CHAL(config)
  elif config['model'] == 'bilstm_crf':
    tag_to_ix = {"B": 0, "I": 1,  "<START>": 2, "<END>": 3}
    reader.w2i['<UNK>']= len(reader.w2i)
    reader.w2i['<PAD>']= len(reader.w2i)
    reader.i2w.append('<UNK>')
    reader.i2w.append('<PAD>')
    print("ix2w len",len(reader.i2w))
    print(reader.i2w[-20:])
    net = bilstm_crf.BiLSTMCRF(reader.w2i, (tag_to_ix), config['word_dim'], config['hid_dim'], config['layer'], 0,
          False,torch.device(config['device']),reader.i2w,use_bert=False)
    #net = bilstm_crf.BiLSTM_CRF(len(reader.w2i), tag_to_ix, config['word_dim'],  config['hid_dim'])
  else:
    raise NotImplementedError()

  if not config['finetune']:
    if config['model'] in set(['aligner', 'ban', 'chunker']):
      net.We_wrd.weight.requires_grad = False
  net.cuda()
  for param in net.parameters():
    weight_init(param)
  return net
