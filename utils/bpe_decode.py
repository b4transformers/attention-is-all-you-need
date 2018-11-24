"""
input:
    model file
    encoded text file
output:
    raw text file
"""

import argparse
import sys
import sentencepiece as spm

parser = argparse.ArgumentParser(
    prog=sys.argv[0],
    usage='python3 {}'.format(sys.argv[0]),
    add_help=True
)
parser.add_argument('-m', '--model', help='model file', required=True)
parser.add_argument('-i', '--input', help='input file', required=True)
parser.add_argument('-o', '--output', help='output file', required=True)
args = parser.parse_args()

sp = spm.SentencePieceProcessor()
if not sp.Load(args.model):
    print('Failed to load the model file: {}'.format(args.model))
    sys.exit()

print('Reading input file ...', end='')
lines = []
with open(args.input, 'r') as f:
    for line in f:
        lines.append(sp.DecodePieces(line.strip().split()))

print('\tDone ({} sentences)'.format(len(lines)))
print('Writing decoded sentences ...', end='')
with open(args.output, 'w') as f:
    for line in lines:
        f.write(line+'\n')
print('\tDone')
