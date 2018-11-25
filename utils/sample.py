import torch
from torch.utils.data import DataLoader
from dataset import Vocab, AspecDataset
from dataloader import MiniBatchProcess
import sys, argparse

parser = argparse.ArgumentParser(
    prog=sys.argv[0],
    usage='python3 {}'.format(sys.argv[0]),
    add_help=True
)
parser.add_argument('-v', '--vocab', help='vocab file', required=True)
parser.add_argument('-i', '--input', help='input file', required=True)
parser.add_argument('-t', '--target', help='target file', required=True)
args = parser.parse_args()

vocab = Vocab(args.vocab)
train_dataset = AspecDataset(args.input, args.target, vocab)
train_loader = DataLoader(train_dataset, collate_fn=MiniBatchProcess(), shuffle=False, batch_size=50)

max_epoch = 5
for epoch in range(max_epoch):
    print('epoch {}'.format(epoch+1))
    for i, batch in enumerate(train_loader):
        print('\rProgress: {}/{}'.format(i+1, len(train_loader)), end='')
        """
        batch: [id, input, input_mask, target, target_mask]
            id: input & targetペアのid (あったら便利かなと。)
            input: (batch_size, max_len)
            input_mask: (batch_size, 1, max_len)
            target: (batch_size, max_len)
            target_mask: (batch_size, max_len, max_len)
        """
    print()
print('Completed!')
