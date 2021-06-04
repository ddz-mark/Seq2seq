# -*- coding: utf-8 -*-
# @Time : 2021/6/3 下午5:08
# @Author : ddz

from model import Encoder, Decoder, Seq2Seq
from utils import get_device
import torch
from torch import nn, tensor
from spacy.lang.de import German
import re, json
import numpy as np


def predict(text):
    # 1. 参数配置
    hidden_size = 512
    embed_size = 256
    de_size, en_size = 8014, 10004
    max_len = 30
    device = get_device()

    # 2. 模型加载
    encoder = Encoder(de_size, embed_size, hidden_size,
                      n_layers=2, dropout=0.5).to(device)
    decoder = Decoder(embed_size, hidden_size, en_size,
                      n_layers=1, dropout=0.5).to(device)

    encoder_path = ".save/1/encoder_1.pt"
    decoder_path = ".save/1/decoder_1.pt"
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    print('loading pretrained encoder models from {}.'.format(encoder_path))
    decoder.load_state_dict(torch.load(decoder_path, map_location=device))
    print('loading pretrained decoder models from {}.'.format(decoder_path))

    encoder.eval()
    decoder.eval()

    # 3. 数据预处理，进行预测不需要对句子进行[sos]、[eos]处理
    spacy_de = German()
    url = re.compile('(<url>.*</url>)')
    tokenized = [tok.text for tok in spacy_de.tokenizer(url.sub('@URL@', text))]

    with open(".data/vocab_de.json", 'r') as load_f:
        dn_load_dict = json.load(load_f)
        indexed = [dn_load_dict[t] if t in dn_load_dict else len(dn_load_dict) + 1 for t in tokenized]
    with open(".data/vocab_en.json", 'r') as load_f:
        en_load_dict = json.load(load_f)
    tensor_src = tensor(indexed).long().to(device)
    tensor_src = tensor_src.unsqueeze(1)

    # 4. 模型预测
    encoder_output, hidden = encoder(tensor_src)
    hidden = hidden[:decoder.n_layers]
    # output:[batch] -> [8]
    output = torch.zeros(tensor_src.size()[1]).long()
    decoded_words = []
    for t in range(max_len):
        # output shape:[batch, vocab] -> [8, 10004]
        output, hidden, attn_weights = decoder(output, hidden, encoder_output)
        # top1 shape:[batch] -> [8]
        top1 = output.data.max(1)[1]
        output = top1.clone().detach().to(device)
        if top1.item() == en_load_dict["<eos>"]:
            break
        else:
            decoded_words.append(list(en_load_dict.keys())[top1.item()])

    print(len(decoded_words), decoded_words)


if __name__ == '__main__':
    text = "Zwei junge weiße Männer sind im Freien in der Nähe vieler Büsche."
    predict(text)
