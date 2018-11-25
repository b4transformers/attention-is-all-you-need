"""
TODO:
論文ではミニバッチ内のtoken数が一定になるようにミニバッチを組んでいるが、
ここではその処理をしておらず、inputのtoken数にソートしたのちに順に取り出している。
時間があれば修正する。
"""


import torch
from torch.utils.data import DataLoader
from dataset import Vocab, AspecDataset
import sys, argparse
import numpy as np

class MiniBatchProcess:
    def __init__(self, device=torch.device('cpu')):
        self.device = device

    def __call__(self, batch):
        return self._body(batch)

    def _body(self, batch):
        id, input, target = [], [], []
        for record in batch:
            id.append(record[0])
            input.append(record[1])
            target.append(record[2])
        id = torch.tensor(id, device=self.device)
        input = self._pad(input)
        target = self._pad(target)
        input_mask = self._mask(input, is_target=False)
        target_mask = self._mask(target, is_target=True)
        return id, input.to(self.device), input_mask.to(self.device), target.to(self.device), target_mask.to(self.device)

    def _pad(self, tensors, pad_id=0):
        max_len = max([len(t) for t in tensors])
        padded = [torch.cat([t, torch.tensor([pad_id]*(max_len-len(t)), dtype=torch.long)], dim=0) for t in tensors]
        padded = torch.stack(padded, dim=0) # (max_len,)
        return padded

    def _mask(self, padded_tensor, pad_id=0, is_target=False):
        mask = (padded_tensor != pad_id).unsqueeze(-2) # (batch_size, 1, max_len)
        if is_target:
            s_mask = self._subsequent_mask(padded_tensor.shape[-1]) # (1, max_len, max_len)
            mask = mask & s_mask # (batch_size, max_len, max_len)
        return mask

    def _subsequent_mask(self, max_len):
        shape = (1, max_len, max_len) # (1, max_len, max_len) --BC-> (batch_size, max_len, max_len)
        mask = np.triu(np.ones(shape), k=1).astype('uint8')
        mask = torch.from_numpy(mask) == 0
        return mask


### for test ###
if __name__ == '__main__':
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
    dataset = AspecDataset(args.input, args.target, vocab)
    loader = DataLoader(dataset, collate_fn=MiniBatchProcess(), shuffle=False, batch_size=50)

    for i, batch in enumerate(loader):
        # test1
        #if i>=2: break
        #print('--- batch_idx={} ---'.format(i))
        #print('id: {}'.format(batch[0]))
        #print('input: {}\n{}'.format(batch[1].shape, batch[1]))
        #print('input mask: {}\n{}'.format(batch[2].shape, batch[2]))
        #print('target: {}\n{}'.format(batch[3].shape, batch[3]))
        #print('target mask: {}\n{}'.format(batch[4].shape, batch[4]))

        # test2
        input_mask = batch[2]
        input_max_len = input_mask.shape[-1]
        input_avg_len = torch.mean(input_mask[:,0].sum(dim=-1).float())
        target_mask = batch[4]
        target_max_len = target_mask.shape[-1]
        target_avg_len = torch.mean(target_mask[:,-1].sum(dim=-1).float())
        print('INPUT: max_len={}, avg_len={}'.format(input_max_len, input_avg_len))
        print('TARGET: max_len={}, avg_len={}'.format(target_max_len, target_avg_len))
        print()

    print()
