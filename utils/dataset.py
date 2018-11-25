
from torch.utils.data import Dataset
import pandas as pd
import torch
import argparse, sys

class Vocab:

    def __init__(self, vocab_file):
        self.stoi = {'<pad>': 0}
        self.itos = {0: '<pad>'}
        with open(vocab_file, 'r') as f:
            for line in f:
                token = line.split()[0]
                self.stoi[token] = len(self.stoi)
                self.itos[len(self.itos)] = token

class AspecDataset(Dataset):

    def __init__(self, input_file, target_file, vocab, transform=None):
        df_input = pd.read_csv(input_file, names=['input'], delimiter='\t')
        df_target = pd.read_csv(target_file, names=['target'], delimiter='\t')
        df_id = pd.Series(list(range(len(df_input))), name='id')
        df_input_len = pd.Series(self._get_num_tokens(df_input['input']), name='input_len')
        df_target_len = pd.Series(self._get_num_tokens(df_target['target']), name='target_len')
        self.df_pair = pd.concat([df_id, df_input, df_target, df_input_len, df_target_len], axis=1).sort_values(['input_len', 'target_len'], ascending=False).reset_index(drop=True)
        self.vocab = vocab

    def _get_num_tokens(self, df_string):
        return [len(string.split(' ')) for string in df_string]

    def __len__(self):
        return len(self.df_pair)

    def __getitem__(self, idx):
        sample = self.df_pair.ix[idx]
        return self._preprocess(sample)

    def _preprocess(self, sample):
        id = torch.tensor(sample['id'].item())
        input = self._to_tensor(sample['input'])
        target = self._to_tensor(sample['target'])
        return id, input, target

    def _to_tensor(self, string):
        token_ids = [self.vocab.stoi[token if token in self.vocab.stoi.keys() else '<unk>'] for token in string.split()]
        token_ids = [self.vocab.stoi['<s>']] + token_ids + [self.vocab.stoi['</s>']]
        return torch.tensor(token_ids)


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

    print('vocab_size: ', len(vocab.stoi))
    print(dataset.df_pair.info())

    for i, (id, input, target) in enumerate(dataset):
        if i >= 2: break
        print('id: {}'.format(id))
        print('input: {}'.format(input))
        print('target: {}'.format(target))
