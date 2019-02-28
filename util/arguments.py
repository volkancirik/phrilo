#from __future__ import absolute_import
import argparse


def get_flickr30k_train():
  parser = argparse.ArgumentParser()

  parser.add_argument('--root', dest='root',
                      help='root path for experiments', default='../')

  parser.add_argument('--reader', dest='reader_file', help='load a dataset dump, default:../data/flickr30k.chunk_align.ban.dataset.pkl',
                      default='../data/flickr30k.chunk_align.ban.dataset.pkl')

  parser.add_argument('--boxes', dest='box_file', default='../data/flickr30k.chunk_align.ban.cnn.h5',
                      help='bottomup features for boxes, default: ../data/flickr30k.chunk_align.ban.cnn.h5')

  parser.add_argument('--model', dest='model',
                      help='model type aligner|ban|chunker|chal default=chal', default='chal')

  parser.add_argument('--use-gt', dest='use_gt',
                      action='store_true', help='use gt boxes')

  parser.add_argument('--use-pos', dest='use_pos',
                      action='store_true', help='use pos tags as features')

  parser.add_argument('--use-bert', dest='use_bert',
                      action='store_true', help='use bert encoder')

  parser.add_argument('--encoder', dest='encoder',
                      help='phrase encoder average|bilstm|bilstm+att default=bilstm', default='bilstm')

  parser.add_argument('--hidden', dest='hid_dim', type=int,
                      help='# of hidden units, default = 128', default=128)

  parser.add_argument('--layer', dest='layer', type=int,
                      help='# of layers for stack encoding projections, default = 1', default=1)

  parser.add_argument('--context-vector', dest='context_vector', type=int,
                      help='context vector for one hot encoding size default = 0', default=0)

  parser.add_argument('--context-embedding', dest='context_embedding', type=int,
                      help='context word embedding size default = 2', default=2)

  parser.add_argument('--word-dim', dest='word_dim', type=int,
                      help='dimension for word embeddings, default = 300', default=300)

  parser.add_argument('--box-dim', dest='box_dim', type=int,
                      help='dimension for box representations, default = 2048+6', default=2048+6)

  parser.add_argument('--no-finetune', dest='finetune',
                      action='store_false', help='do not finetune word embeddings')

  parser.add_argument('--verbose', dest='verbose',
                      action='store_true', help='print to stdout')

  parser.add_argument('--epochs', type=int, default=10,
                      help='# of epochs, default = 10')

  parser.add_argument('--val-freq', dest='val_freq', type=int, default=50000,
                      help='validate every n instances, 0 is for full pass over trn data, default = 50000')

  parser.add_argument('--save-path', dest='save_path', type=str,
                      default='exp', help='folder to save experiment')

  parser.add_argument('--resume', dest='resume', type=str,
                      default='', help='resume from this model snapshot')

  parser.add_argument('--clip', dest='clip',
                      help='gradient clipping, default=1.0', type=float, default=1.0)

  parser.add_argument('--optim', dest='optim',
                      help='optimization method adam|sgd, default:sgd', default='sgd')

  parser.add_argument(
      '--lr', dest='lr', help='initial learning rate, default = 0.0003', default=0.0001, type=float)

  parser.add_argument('--lr-min', dest='lr_min',
                      help='minimum lr, default = 0.000001', default=0.000001, type=float)

  parser.add_argument('--lrdecay', dest='lr_decay',
                      help='learning rate decay, default = 0.95', default=0.95, type=float)

  parser.add_argument('--w-decay', dest='w_decay',
                      help='weight decay, default = 0.0001', default=0.0001, type=float)

  parser.add_argument('--aligner', dest='aligner',
                      help='aligner model', default='')

  parser.add_argument('--chunker', dest='chunker',
                      help='chunker model', default='')

  parser.add_argument('--nonlinearity', dest='nonlinearity',
                      help='nonlinearity relu,sigmoid,tanh,none default=none', default='none')

  parser.add_argument('--use-predicted', dest='use_predicted',
                      action='store_true', help='use predicted chunks')

  parser.add_argument('--details', dest='details',
                      action='store_true', help='print details to progress bar.')

  parser.add_argument('--max-length', type=int, default=20,
                      help='max sentence length, default = 20')

  args = parser.parse_args()
  return args
