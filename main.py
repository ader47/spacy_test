import torch
import spacy
from torchtext.datasets import Multi30k
from collections import Counter
from torchtext.vocab import vocab
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
def tokenize_en(text):
    return [tok.text.lower() for tok in spacy_en.tokenizer(text)][::-1]
def tokenize_de(text):
    return [tok.text.lower() for tok in spacy_de.tokenizer(text)]

spacy_en=spacy.load('en_core_web_sm')
spacy_de=spacy.load('de_core_news_sm')
counter_en = Counter()
counter_de=Counter()
train_iter, test_iter = Multi30k(split=('train', 'test'))
diter = iter(train_iter)
while True:
    try:
        text = next(diter)
    except StopIteration:
        diter = iter(train_iter)
        break
    counter_en.update(tokenize_en(text[1]))
    counter_de.update(tokenize_de(text[0]))
vocab_de = vocab(counter_en, min_freq=2, specials=('<unk>', '<BOS>', '<EOS>', '<PAD>'))
vocab_de.set_default_index(vocab_de.get_stoi()['<unk>'])
en_transfor = lambda x: [vocab_de['<BOS>']] + [vocab_de[token] for token in tokenize_en(x)] + [vocab_de['<EOS>']]
vocab_en = vocab(counter_de, min_freq=2, specials=('<unk>', '<BOS>', '<EOS>', '<PAD>'))
vocab_en.set_default_index(vocab_en.get_stoi()['<unk>'])
de_transfor = lambda x: [vocab_en['<BOS>']] + [vocab_en[token] for token in tokenize_en(x)] + [vocab_en['<EOS>']]

def collate_batch(batch):
    trg_list,src_list=[],[]
    for src,trg in batch:
        print(src,trg)
        trg_list.append(torch.tensor(en_transfor(trg)))
        src_list.append(torch.tensor(de_transfor(src)))
    src_list=pad_sequence(src_list, padding_value=vocab_de.get_stoi()['<PAD>'])
    trg_list = pad_sequence(trg_list, padding_value=vocab_en.get_stoi()['<PAD>'])
    return src_list,trg_list
if __name__ == '__main__':

    dl=DataLoader(train_iter,batch_size=8,collate_fn=collate_batch)
    print(next(iter(dl)))
