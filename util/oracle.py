#!/usr/bin/python


class Phrase(object):
  def __init__(self, tokens=[], entity='N/A', etype='N/A'):
    self.tokens = tokens
    self.entity = entity
    self.etype = etype


def parse_sentence(tokens, tokenized=False):
  if not tokenized:
    tokens = tokens.lower().split(' ')

  phrase = Phrase([])
  sentence = []
  open_bracket = False
  for token in tokens:
    if token[0] == '[':
      if open_bracket:
        print('ERROR! open bracket seen multiple times')
        raise NotImplementedError()
      if len(phrase.tokens) > 0:
        sentence.append(phrase)
        phrase = Phrase([])
      phrase.entity = token.split('/')[1].split('#')[1]
      phrase.etype = token.split('/')[2]
      open_bracket = True
    elif token[-1] == ']':
      if not open_bracket:
        print('ERROR! closed bracket without opening one')
        raise NotImplementedError()
      token = token[:-1]
      phrase.tokens.append(token.lower())
      sentence.append(phrase)
      open_bracket = False
      phrase = Phrase([])
    else:
      phrase.tokens.append(token.lower())

  if phrase.tokens != []:
    sentence.append(phrase)
  return sentence


def get_sentence(sentence):
  return ' '.join([' '.join(phrase.tokens) for phrase in sentence])


def get_tokens(sentence):
  return sum([phrase.tokens for phrase in sentence], [])


def get_entities(sentence):
  return [phrase.entity for phrase in sentence if phrase.entity != 'N/A']


def get_chunk_oracle(sentence, e2box=set()):

  parsed = parse_sentence(sentence)
  actions = ['NT_SENTENCE']
  idx2a = {0: 'NT', 1: 'SHIFT', 2: 'REDUCE', 3: 'GROUND'}

  for ph in parsed:
    # new NT
    if ph.entity == 'N/A' or (len(e2box) > 0 and len(e2box[ph.entity]) == 0):
      nt_type = 'N/A'
    else:
      nt_type = ph.etype
    actions += [idx2a[0] + '_' + nt_type]

    # shift tokens
    for tok in ph.tokens:
      actions += [idx2a[1] + '_' + tok]

    # reduce phrase
    if ph.entity == 'N/A' or (len(e2box) > 0 and len(e2box[ph.entity]) == 0):
      red_act = [idx2a[2]]
    else:
      if len(e2box) == 0:
        red_act = ['_'.join([idx2a[3], ph.entity])]
      else:
        red_act = ['_'.join([idx2a[3], '-'.join([str(box)
                                                 for box in e2box[ph.entity]])])]
    actions += red_act
  actions += ['REDUCE']
  return actions

# def get_parse_oracle(, e2box = set())


def sentence2bio(sentence, e2box=set(), tokenized=False):
  tags = []
  parsed = parse_sentence(sentence, tokenized=tokenized)
  for phrase in parsed:
    if phrase.entity == 'N/A' or (len(e2box) > 0 and phrase.entity in e2box and len(e2box[phrase.entity]) == 0):
      tags += ['O' for tok in phrase.tokens]
    else:
      tag = phrase.etype
      tags += ['B-' + tag]
      tags += ['I-'+tag for tok in phrase.tokens[1:]]
  return tags


if __name__ == '__main__':

  s = '[/EN#176052/people People] moving [/EN#176053/other some new equipment] into [/EN#176056/scene a house] .'
  e2box = {'176051': set([0]), '176052': set([1, 2]), '176053': set([3, 4, 5]), '176054': set([6]), '176055': set(
      [7]), '176056': set([8]), '176057': set([1, 10, 11, 12, 9]), '176060': set([1, 11, 13, 9]), '176061': set([14])}

  actions = get_chunk_oracle(s, e2box=e2box)
  print('\n'.join(actions))

  parsed_f30k = parse_sentence(s)
  print(' '.join(get_tokens(parsed_f30k)))
  print(sentence2bio(s, e2box))

  s = '[/EN#251483/people Two motorcyclists] racing [/EN#251490/bodyparts neck] and [/EN#251491/bodyparts neck] around [/EN#251486/scene a corner] .'
  parsed_f30k = parse_sentence(s)
  for ph in parsed_f30k:
    print(ph.entity, ph.tokens)
