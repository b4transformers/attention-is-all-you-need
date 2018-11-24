"""
input:
    raw text file
output:
    model file
    vocab file
"""

import argparse
import sys, os
import sentencepiece as spm

parser = argparse.ArgumentParser(
    prog=sys.argv[0],
    usage='python3 {}'.format(sys.argv[0]),
    add_help=True
)
parser.add_argument('-i', '--input', help='input file', required=True)
parser.add_argument('-o', '--output', help='output directory', default='./')
parser.add_argument('-v', '--vocab', help='vocabulary size', type=int, default=32000)
parser.add_argument('-m', '--model', help='model type of sentencepiece', default='bpe')
args = parser.parse_args()

input = args.input
if not args.output.endswith(os.path.sep):
    args.output += '/'
model_prefix = args.output + args.model + str(args.vocab)
vocab_size = args.vocab
model_type = args.model

spm.SentencePieceTrainer.Train('--input={} --model_prefix={} --vocab_size={} --model_type={}'
    .format(input, model_prefix, vocab_size, model_type))
