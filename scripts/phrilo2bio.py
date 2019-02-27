import pickle
import sys
from tqdm import tqdm

from util.reader import Reader


def print_bio(words, tags, output_file, merge_o=False):
  for tag in tags:
    start = tag[0]
    end = tag[1]
    if start == end:
      output_file.write(words[start] + " TAG B-PH\n")
      continue

    if start >= len(words):
      continue

    output_file.write(words[start] + " TAG B-PH\n")
    for j in range(start+1, end+1):
      output_file.write(words[j] + " TAG I-PH\n")
  output_file.write("\n")


def convert_data(reader_file, output_prefix):

  reader = pickle.load(open(reader_file, 'rb'))

  for split_name in ["trn", "val", "tst"]:
    output_file = open(output_prefix + "." + split_name + ".txt", "w")
    output_file.write("-DOCSTART- -X- O O\n\n")
    split = reader.data[split_name]
    indexes = range(len(split[0]))
    pbar = tqdm(indexes)
    for idx in pbar:
      words = split[0][idx]
      tags, _, _ = split[3][idx]
      print_bio(words, tags, output_file)

    output_file.close()


if __name__ == "__main__":
  READER_FILE = sys.argv[1]
  OUTPUT_PREFIX = sys.argv[2]
  convert_data(READER_FILE, OUTPUT_PREFIX)
