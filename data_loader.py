import re
import spacy
from torchtext.data import Field, BucketIterator
from torchtext.datasets import Multi30k
from spacy.lang.en import English
from spacy.lang.de import German


def load_dataset(batch_size):
    # spacy_de = spacy.load('de')
    # spacy_en = spacy.load('en')
    spacy_de = German()
    spacy_en = English()
    url = re.compile('(<url>.*</url>)')

    def tokenize_de(text):
        return [tok.text for tok in spacy_de.tokenizer(url.sub('@URL@', text))]

    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(url.sub('@URL@', text))]

    DE = Field(tokenize=tokenize_de, include_lengths=True,
               init_token='<sos>', eos_token='<eos>')
    EN = Field(tokenize=tokenize_en, include_lengths=True,
               init_token='<sos>', eos_token='<eos>')
    train, val, test = Multi30k.splits(exts=('.de', '.en'), fields=(DE, EN))
    print(train[0].src, train[0].trg)
    # print(train.fields)
    # 建立词汇表，min_freq 最少出现次数为2，出现一次的设为 UNK
    DE.build_vocab(train.src, min_freq=2)
    EN.build_vocab(train.trg, max_size=10000)
    train_iter, val_iter, test_iter = BucketIterator.splits(
        (train, val, test), batch_size=batch_size, repeat=False)
    return train_iter, val_iter, test_iter, DE, EN


if __name__ == '__main__':
    train_iter, val_iter, test_iter, DE, EN = load_dataset(8)
    for batch_idx, batch_v in enumerate(train_iter):
        print(batch_v)
        src, len_src = batch_v.src
        trg, len_trg = batch_v.trg
        print(src.shape, len_src, trg.shape, len_trg)
        if batch_idx == 3:
            break
