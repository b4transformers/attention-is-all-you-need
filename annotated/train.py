from prelims import *
from model import *
from utils import *
from collections import Counter
import os
import argparse


class Vocab:
    def __init__(self, vocab_file):
        self.stoi = {'<pad>': 0}
        self.itos = {0: '<pad>'}
        with open(vocab_file, 'r') as f:
            for line in f:
                token = line.split()[0]
                self.stoi[token] = len(self.stoi)
                self.itos[len(self.itos)] = token


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab", type=str, required=True,
                        help="Path to the vocabulary file.")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to the data directory.")
    parser.add_argument("--gpu", type=int, required=True,
                        default=-1,
                        help="GPU core id to use. Single gpu only. Default cpu.")
    args = parser.parse_args()

    # -- en-ja dataset -- #
    print('Creating dataset ...')
    vocab_path = args.vocab
    BOS_WORD = '<s>'
    EOS_WORD = '</s>'
    BLANK_WORD = "<pad>"

    def tokenize_bpe(text):
        return text.split(' ')

    TEXT = data.Field(tokenize=tokenize_bpe, pad_token=BLANK_WORD,
        use_vocab=True, sequential=True
    )
    train, val, test = datasets.TranslationDataset.splits(
        path=args.data, train='train', validation='dev', test='test',
        exts=('.bpe.en', '.bpe.ja'), fields=(TEXT, TEXT))
    vocab = Vocab(vocab_path)
    TEXT.vocab = torchtext.vocab.Vocab(Counter(vocab.stoi.keys()), specials=[])

    # -- model, iterator -- #
    # GPUs to use
    # devices = [0, 1, 2, 3]
    if args.gpu > -1:
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    print('Creating model & iterator ...')
    pad_idx = TEXT.vocab.stoi[BLANK_WORD]
    model = make_model(len(TEXT.vocab.stoi), len(TEXT.vocab.stoi), N=6)
    model.cuda()
    criterion = LabelSmoothing(size=len(TEXT.vocab.stoi), padding_idx=pad_idx, smoothing=0.1)
    criterion.cuda()
    BATCH_SIZE = 12000
    train_iter = MyIterator(train, batch_size=BATCH_SIZE, device=device,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=True)
    valid_iter = MyIterator(val, batch_size=BATCH_SIZE, device=device,
                            repeat=False, sort_key=lambda x: (len(x.src), len(x.trg)),
                            batch_size_fn=batch_size_fn, train=False)
    # model_par = nn.DataParallel(model, device_ids=devices)
    # --shared embedding-- #
    print('Embedding is shared.')
    model.src_embed[0].lut.weight = model.tgt_embed[0].lut.weight
    model.generator.proj.weight = model.tgt_embed[0].lut.weight


    # --start training-- #
    max_epoch = 100
    save_dir = './models/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_every = 1
    print('Training ...')
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 2000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    for epoch in range(max_epoch):
        model.train()
        # model_par.train()
        run_epoch((rebatch(pad_idx, b) for b in train_iter),
                  model,
                  # model_par,
                  SimpleLossCompute(model.generator, criterion, opt=model_opt))
                  # MultiGPULossCompute(model.generator, criterion, devices=devices, opt=model_opt))
        model.eval()
        # model_par.eval()
        loss = run_epoch((rebatch(pad_idx, b) for b in valid_iter),
                         model,
                         # model_par,
                         SimpleLossCompute(model.generator, criterion, opt=None))
                         # MultiGPULossCompute(model.generator, criterion, devices=devices, opt=None))

        print(loss)

        # save model
        if epoch % save_every == 0:
            save_path = save_dir+'state_dict{}.pth'.format(epoch)
            torch.save(model.state_dict(), save_path)
            print('Saved model as: ', save_path)

#
# else:
#     import glob
#     path = sorted(glob.glob(save_dir+'*'))[-1]
#     model.load_state_dict(torch.load(path))
