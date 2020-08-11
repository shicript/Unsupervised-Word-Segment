'''
Descripttion: 
version: 
Author: Shicript
Date: 2020-08-11 15:00:54
LastEditors: Shicript
LastEditTime: 2020-08-11 15:03:36
'''
import torch
import numpy as np
from transformers import BertConfig, BertModel, BertTokenizer
from transformers.tokenization_utils_base import BatchEncoding

config_path = "" # bert_config.json 
checkpoint_path = "" # bert pytorch model
vocab_path = "" # bert chinese vocab file 

config = BertConfig.from_pretrained(config_path)
tokenizer = BertTokenizer.from_pretrained(vocab_path)
model = BertModel.from_pretrained(checkpoint_path, config=config)


def dist(x, y):
    """距离函数（默认欧氏距离）
    可以用内积或者cos距离
    """
    return np.sqrt(((x - y) ** 2).sum())


def get_word_seg(text):
    token_batch = tokenizer(
        text,
        return_tensors="pt"
    )

    token_ids = token_batch["input_ids"]
    token_ids = token_ids.numpy().tolist()[0]
    length = len(token_ids) - 2

    batch_token_ids = np.array([token_ids] * (2 * length - 1))
    batch_segment_ids = np.zeros_like(batch_token_ids)

    for i in range(length):
        if i > 0:
            batch_token_ids[2 * i - 1, i] = 103
            batch_token_ids[2 * i - 1, i + 1] = 103
        batch_token_ids[2 * i, i + 1] = 103

    attention_mask = token_batch["attention_mask"].repeat(2 * length - 1, 1)
    input_ids = torch.from_numpy(batch_token_ids)
    token_type_ids = torch.from_numpy(batch_segment_ids)

    input_dict = {
        'input_ids': input_ids,
        'token_type_ids': token_type_ids,
        'attention_mask': attention_mask
    }

    inputs = BatchEncoding(input_dict)

    outputs = model(**inputs)
    vectors, _ = outputs[:2]
    vectors = vectors.detach().numpy()

    seg_list = []
    for threshold in range(length):
        # threshold = 8
        print(threshold)
        word_token_ids = [[token_ids[1]]]
        for i in range(1, length):
            d1 = dist(vectors[2 * i, i + 1], vectors[2 * i - 1, i + 1])
            d2 = dist(vectors[2 * i - 2, i], vectors[2 * i - 1, i])
            d = (d1 + d2) / 2
            if d >= threshold:
                word_token_ids[-1].append(token_ids[i + 1])
            else:
                word_token_ids.append([token_ids[i + 1]])
        words = [tokenizer.decode(ids).replace(" ", "") for ids in word_token_ids]
        print(words)
        seg_list.append(words)

    return seg_list


def run():
    texts = ['毛红椿天然林群落土壤丛枝菌根真菌群落特征研究', '德国电子证据取证制度']
    get_word_seg(texts[1])


if __name__ == '__main__':
    run()
