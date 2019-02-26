'''(I)LP solvers for chunking, alingment, and joing chunking and alignment.

'''
from collections import defaultdict
import numpy as np
from scipy.sparse import coo_matrix
import scipy.sparse as sp

try:
  from ortools.linear_solver import pywraplp
except ImportError:
  print('or-tools does not seem to be installed. Skipping for now')


def linprog(f, A, b, is_integer=[]):
  '''
  Solve the following linear programming problem
          maximize_x (f.T).dot(x)
          subject to A.dot(x) <= b
  where   A is a sparse matrix (coo_matrix)
          f is column vector of cost function associated with variable
          b is column vector
  '''

  # flatten the variable
  f = f.ravel()
  b = b.ravel()

  solver = pywraplp.Solver('Solve(I)LP',
                           pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)

  infinity = solver.Infinity()
  n, m = A.shape
  x = [[]] * m
  c = [0] * n

  for j in range(m):
    x[j] = solver.IntVar(0, 1, 'x_%u' % j)

  # state objective function
  objective = solver.Objective()
  for j in range(m):
    objective.SetCoefficient(x[j], f[j])

  objective.SetMaximization()

  # state the constraints
  for i in range(n):
    c[i] = solver.Constraint(-infinity, b[i])
    for j in A.col[A.row == i]:
      c[i].SetCoefficient(
          x[j], A.data[np.logical_and(A.row == i, A.col == j)][0])

  result_status = solver.Solve()
#    if result_status != 0:
#        print('The final solution might not converged')

  x_sol = np.array([x_tmp.SolutionValue() for x_tmp in x])

  return {'x': x_sol, 'status': result_status}


class LPChunker():
  '''Chunks a sentence given an input of scored chunks.
  '''

  def __init__(self):
    self.cache = {}
    self.cache_hit = defaultdict(int)

  def solve(self, phi, n_tokens, _n_boxes, use_coeff=True):
    '''Solve LP for given chunk scores.

    Args:
      phi: (np array) scores for each chunk.
      n_tokens: (int) number of tokens.
      use_coeff: (bool) normalize chunk scores by length.

    Returns:
      chunks: (list) list of (start,end) tuples for chunks.
    '''
    if n_tokens not in self.cache:

      n_chunks = int((n_tokens*(n_tokens+1))/2)
      n_conditions = (n_chunks+n_tokens)*2 + 2
      coeff = []
      for i in range(n_tokens):
        coeff += [i+1]*(n_tokens-i)

      coeff = np.array(coeff)

      tok2chunks = defaultdict(list)
      id2chunk = {}
      chunk2id = {}
      idx = 0
      for i in range(0, n_tokens):
        for j in range(i, n_tokens):
          id2chunk[idx] = (i, j)
          chunk2id[(i, j)] = idx
          idx += 1
          for k in range(i, j+1):
            tok2chunks[k].append((i, j))
      a_chunk = sp.dok_matrix(
          (n_conditions, n_chunks), dtype=np.float)
      constraint = np.zeros((1, n_conditions))

      c_offset = 0
      for i in range(n_chunks):
        constraint[0, 2*i] = 1
        constraint[0, 2*i+1] = 0
        a_chunk[2*i, i] = 1
        a_chunk[2*i+1, i] = -1
        c_offset += 2

      for i in range(n_tokens):
        for chunk in tok2chunks[i]:
          idx = chunk2id[chunk]
          a_chunk[c_offset+2*i, idx] = 1
          a_chunk[c_offset+2*i + 1, idx] = -1
          constraint[0, c_offset + 2*i] = 1
          constraint[0, c_offset + 2*i+1] = -1

      c_offset += 2*n_tokens

      for i in range(n_chunks):
        a_chunk[c_offset, i] = 1
        a_chunk[c_offset+1, i] = -1
      constraint[0, c_offset] = n_tokens-1
      constraint[0, c_offset + 1] = -2

      self.cache[n_tokens] = (
          a_chunk, constraint, n_chunks, tok2chunks, id2chunk, chunk2id, coeff)

    a_chunk, constraint, n_chunks, tok2chunks, id2chunk, chunk2id, coeff = self.cache[
        n_tokens]

    if use_coeff:
      x_sol = linprog(phi*coeff, coo_matrix(a_chunk), constraint)
    else:
      x_sol = linprog(phi, coo_matrix(a_chunk), constraint)

    chunks = []

    if x_sol['status'] == 0:
      for i, val in enumerate(x_sol['x']):
        if val > 0.:
          chunks.append(id2chunk[i])

    return chunks, None, tok2chunks, id2chunk, chunk2id


class LPAligner():
  '''Aligns chunks to boxes.
  '''

  def __init__(self, max_chunks_per_box=1,
               max_boxes_per_chunk=5,
               min_chunks_per_box=0,
               min_boxes_per_chunk=0):
    '''Initialize the aligner.

    Args:
      max_chunks_per_box: (int) constraint for max chunk per box.
      max_boxes_per_chunk: (int) constraint for max box per chunk.
      min_chunks_per_box: (int) constraint for min chunk per box.
      min_boxes_per_chunk: (int) constraint for min box per chunk.
    '''
    self.max_chunks_per_box = max_chunks_per_box
    self.max_boxes_per_chunk = max_boxes_per_chunk
    self.min_chunks_per_box = min_chunks_per_box
    self.min_boxes_per_chunk = min_boxes_per_chunk
    self.cache = {}
    self.cache_hit = defaultdict(int)

  def solve(self, phi, n_chunks, n_boxes, chunk_boundary=None):
    '''Solves alignment problem given scores between chunks and boxes.
    Args:
      phi: (np array) scores for each chunk-box pair.
      n_chunks: (int) number of chunks.
      n_boxes: (int) number of boxes.

    Returns:
     alignments: (list) two dicts of chunk->box, box->chunk alignments.
    '''

    if (n_chunks, n_boxes) not in self.cache:
      n_edges = n_chunks*n_boxes
      ii, jj = np.ones((n_boxes, n_chunks)).nonzero()

      a_edges = sp.dok_matrix(
          (n_chunks + n_boxes, n_edges), dtype=np.float)
      a_edges[ii, [v for v in range(n_edges)]] = 1
      a_edges[jj + n_boxes, [v for v in range(n_edges)]] = 1

      a_boxes = sp.dok_matrix((n_boxes, n_edges), dtype=np.float)
      a_boxes[ii, [v for v in range(n_edges)]] = -1

      a_chunks = sp.dok_matrix((n_chunks, n_edges), dtype=np.float)
      a_chunks[jj, [v for v in range(n_edges)]] = -1

      constraint_max = [self.max_chunks_per_box] * \
          n_boxes + [self.max_boxes_per_chunk] * n_chunks
      constraint_min = [-self.min_chunks_per_box] * \
          n_boxes + [-self.min_boxes_per_chunk] * n_chunks

      # a_id1 = sp.dok_matrix((n_edges, n_edges), dtype=np.float)
      # a_id1[range(n_edges), [v for v in range(n_edges)]] = 1

      # a_id2 = sp.dok_matrix((n_edges, n_edges), dtype=np.float)
      # a_id2[range(n_edges), [v for v in range(n_edges)]] = -1

      # a_align = sp.vstack([a_edges, a_boxes, a_chunks, a_id1, a_id2])
      # constraint_id = [1.] * n_edges + [0.] * n_edges
      # constraint = np.array(
      #     constraint_max + constraint_min + constraint_id)

      a_align = sp.vstack([a_edges, a_boxes, a_chunks])
      constraint = np.array(
          constraint_max + constraint_min, dtype=np.float)
      constraint = constraint.reshape(constraint.shape[0], 1)

      self.cache[(n_chunks, n_boxes)] = (
          a_align, constraint)

    a_align, constraint = self.cache[(
        n_chunks, n_boxes)]
    self.cache_hit[(n_chunks, n_boxes)] += 1
    if phi.shape[1] != n_chunks*n_boxes:
      raise ValueError('Shapes do not match!')

    phi = phi.reshape(n_chunks, n_boxes).transpose().reshape(
        1, n_chunks*n_boxes)
    x_sol = linprog(phi, coo_matrix(a_align),
                    constraint)
    alignments = [defaultdict(list), defaultdict(list)]
    if x_sol['status'] == 0:
      for i, val in enumerate(x_sol['x']):
        if val > 0.:
          if chunk_boundary:
            alignments[0][chunk_boundary[i / n_chunks]
                          ].append(int(i % n_chunks))
            alignments[1][int(i % n_chunks)].append(
                chunk_boundary[i / n_boxes])
          else:
            alignments[0][int(i / n_chunks)].append(int(i % n_chunks))
            alignments[1][int(i % n_chunks)].append(int(i / n_chunks))
    return alignments


class LPChunkerAligner():
  '''Given scores for each possible chunk, chunk-box scores
  decodes chunks and alignment.
  '''

  def __init__(self, max_chunks_per_box=1,
               max_boxes_per_chunk=1,
               min_chunks_per_box=0,
               min_boxes_per_chunk=0):
    '''Initialize the aligner.

    Args:
      max_chunks_per_box: (int) constraint for max chunk per box.
      max_boxes_per_chunk: (int) constraint for max box per chunk.
      min_chunks_per_box: (int) constraint for min chunk per box.
      min_boxes_per_chunk: (int) constraint for min box per chunk.
    '''
    self.max_chunks_per_box = max_chunks_per_box
    self.max_boxes_per_chunk = max_boxes_per_chunk
    self.min_chunks_per_box = min_chunks_per_box
    self.min_boxes_per_chunk = min_boxes_per_chunk
    self.cache = {}
    self.cache_hit = defaultdict(int)

  def solve(self, phi, n_tokens, n_boxes, use_coeff=True):
    '''Solves chunking and alignment problem given scores between chunks and boxes.
    Args:
      phi: (np array) scores for each chunk-box pair.
      n_tokens: (int) number of tokens.
      n_boxes: (int) number of boxes.

    Returns:
     alignments: (list) two dicts of chunk->box, box->chunk alignments.
     tok2chunks: (dict) token id -> chunk id
     id2chunk: (dict) id -> chunk tuple (start,end)
     chunk2id: (dict) chunk tuple (start,end) -> id
    '''

    if n_tokens not in self.cache:
      n_chunks = int((n_tokens*(n_tokens+1))/2)
      n_conditions = (n_chunks+n_tokens)*2 + 2
      tok2chunks = defaultdict(list)
      id2chunk = {}
      chunk2id = {}
      idx = 0
      for i in range(0, n_tokens):
        for j in range(i, n_tokens):
          id2chunk[idx] = (i, j)
          chunk2id[(i, j)] = idx
          idx += 1
          for k in range(i, j+1):
            tok2chunks[k].append((i, j))
      ii, jj = np.ones((n_boxes, n_chunks)).nonzero()
      a_chunk = sp.dok_matrix(
          (n_conditions, n_chunks*(1 + n_boxes)), dtype=np.float)
      constraint_chunk = [0.0] * n_conditions

      c_offset = 0
      for i in range(n_chunks):
        constraint_chunk[2*i] = 1
        constraint_chunk[2*i+1] = 0
        a_chunk[2*i, i] = 1
        a_chunk[2*i+1, i] = -1
        c_offset += 2
      for i in range(n_tokens):
        for chunk in tok2chunks[i]:
          idx = chunk2id[chunk]
          a_chunk[c_offset+2*i, idx] = 1
          a_chunk[c_offset+2*i + 1, idx] = -1
        constraint_chunk[c_offset + 2*i] = 1.0
        constraint_chunk[c_offset + 2*i+1] = -1.0

      c_offset += 2*n_tokens

      for i in range(n_chunks):
        a_chunk[c_offset, i] = 1
        a_chunk[c_offset+1, i] = -1
      constraint_chunk[c_offset] = n_tokens-1
      constraint_chunk[c_offset + 1] = -2

      b_offset = n_chunks
      n_edges = n_chunks*n_boxes

      a_edges = sp.dok_matrix(
          (n_chunks + n_boxes, n_edges + b_offset), dtype=np.float)
      a_edges[ii, [v + b_offset for v in range(n_edges)]] = 1
      a_edges[jj + n_boxes, [v + b_offset for v in range(n_edges)]] = 1

      a_boxes = sp.dok_matrix((n_boxes, n_edges + b_offset), dtype=np.int)
      a_boxes[ii, [v + b_offset for v in range(n_edges)]] = -1

      a_chunks = sp.dok_matrix((n_chunks, n_edges + b_offset), dtype=np.int)
      a_chunks[jj, [v + b_offset for v in range(n_edges)]] = -1

      constraint_max = [self.max_chunks_per_box] * \
          n_boxes + [self.max_boxes_per_chunk] * n_chunks
      constraint_min = [-self.min_chunks_per_box] * \
          n_boxes + [-self.min_boxes_per_chunk] * n_chunks

      # a_id1 = sp.dok_matrix((n_edges, n_edges + b_offset), dtype=np.int)
      # a_id1[range(n_edges), [v + b_offset for v in range(n_edges)]] = 1
      # a_id2 = sp.dok_matrix((n_edges, n_edges + b_offset), dtype=np.int)
      # a_id2[range(n_edges), [v + b_offset for v in range(n_edges)]] = -1
      # a_align = sp.vstack([a_edges, a_boxes, a_chunks, a_id1, a_id2])
      # constraint_id = [1.] * n_edges + [0.] * n_edges
      # constraint_align = constraint_max + constraint_min + constraint_id

      a_align = sp.vstack([a_edges, a_boxes, a_chunks])
      constraint_align = constraint_max + constraint_min

      a_joint = sp.dok_matrix((n_edges, n_edges + b_offset), dtype=np.float)
      for ii in range(n_edges):
        a_joint[ii, ii + n_chunks] = 1
        a_joint[ii, ii % n_chunks] = -1

      constraint_joint = [0] * n_edges

      constraint = constraint_chunk + constraint_align + constraint_joint

      A = sp.vstack([a_chunk, a_align, a_joint])

      coeff = []
      for i in range(n_tokens):
        coeff += [i+1]*(n_tokens-i)
      coeff += [1.0] * (n_boxes * n_chunks)
      coeff = np.array(coeff)

      self.cache[(n_tokens, n_boxes)] = (
          A, constraint, n_chunks, tok2chunks, id2chunk, chunk2id)

    A, constraint, n_chunks, tok2chunks, id2chunk, chunk2id = self.cache[(
        n_tokens, n_boxes)]
    self.cache_hit[(n_tokens, n_boxes)] += 1
    if phi.shape[1] != n_chunks*(n_boxes+1):
      raise ValueError('Shapes do not match! {} vs {}'.format(
          phi.shape[1], n_chunks*(n_boxes+1)))

    # convert chunk -> box to box -> chunk
    phi[0, n_chunks:] = phi[0, n_chunks:].reshape(n_chunks, n_boxes).transpose().reshape(
        1, n_chunks*n_boxes)

    x_sol = linprog(phi*coeff if use_coeff else phi, coo_matrix(
        A), np.array(constraint))

    chunks = []
    alignments = [defaultdict(list), defaultdict(list)]
    if x_sol['status'] == 0:
      for i, val in enumerate(x_sol['x']):
        if val > 0.9:
          if i < n_chunks:
            chunks.append(id2chunk[i])
          if i >= n_chunks:
            idx = i - n_chunks
            alignments[0][int(idx / n_chunks)].append(int(idx % n_chunks))
            alignments[1][int(idx % n_chunks)].append(int(idx / n_chunks))

    return chunks, alignments, tok2chunks, id2chunk, chunk2id


def run_tests():
  '''Run a basic decoding test.
  '''
  n_tokens = 3
  n_chunks = int(n_tokens*(n_tokens+1)/2)
  n_boxes = 5

  n_chunks = 3
  n_boxes = 4

  phi = np.array([[-10.000, -10.000, -0.050, -0.100,
                   -10.000, -0.600, -10.000, -0.050,
                   -0.050, -10.000, -0.060, -10.000,
                   ]])
  aligner = LPAligner(max_chunks_per_box=3,
                      max_boxes_per_chunk=1, min_boxes_per_chunk=1, min_chunks_per_box=0)
  alignments = aligner.solve(
      phi, n_chunks, n_boxes)
  print('phi\n', (phi - np.min(phi) - 0.0001) .reshape(n_chunks, n_boxes))

  print('\nAligner output:')
  print(alignments)
  print('_'*30)

  n_tokens = 4
  n_chunks = int(n_tokens*(n_tokens+1)/2)
  n_boxes = 3
  phi = np.array(
      [[0.00, 1.01, 0.02, 0.03, 0.04, 0.05, 0.06, 1.07, 0.08, 1.09]])
  chunker = LPChunker()
  print('phi\n', phi)
  chunks, _, _, _, _ = chunker.solve(phi, n_tokens, n_boxes)
  print('\nChunker output:')
  print(chunks)
  print('_'*30)

  n_tokens = 4
  n_chunks = int(n_tokens*(n_tokens+1)/2)
  n_boxes = 3
  phi = np.array([[0.00, 1.01, 0.02, 0.03, 0.04, 0.05, 0.06, 1.07, 0.08, 1.09,
                   -0.00, -0.01, 0.02,
                   1.10, -0.01, -0.02,
                   -0.20, -0.01, -0.02,
                   -0.30, -0.01, -0.02,
                   -0.40, -0.01, -0.02,
                   -0.50, -0.01, -0.02,
                   -0.60, -0.01, -0.02,
                   -0.70, 0.01, -0.02,
                   -0.80, -0.01, -0.02,
                   1.90, -0.01,  1.02]])

  chunkeraligner = LPChunkerAligner(max_chunks_per_box=n_tokens,
                                    max_boxes_per_chunk=3,
                                    min_boxes_per_chunk=0,
                                    min_chunks_per_box=0)
  chunks, alignments, _, id2chunk, chunk2id = chunkeraligner.solve(
      phi, n_tokens, n_boxes)
  print('\nChunker+Aligner output:')
  print(chunks)
  print(id2chunk)
  print(chunk2id)
  print(alignments)


if __name__ == '__main__':
  run_tests()
