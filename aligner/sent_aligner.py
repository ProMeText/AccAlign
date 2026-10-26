# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# Modifications copyright (C) 2020 Zi-Yi Dou
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import random
import itertools
import os

import numpy as np
import torch
from tqdm import trange
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, IterableDataset

from transformers import AutoTokenizer, AutoConfig, AutoModel
from aligner.word_align import SentenceAligner_word
from tqdm import tqdm




class LineByLineTextDataset(IterableDataset):
    def __init__(self, tokenizer, file_path_src, file_path_tgt, offsets=None, format="txt"):
        assert os.path.isfile(file_path_src)
        assert os.path.isfile(file_path_tgt)
        print('Loading the dataset...')
        self.examples = []
        self.tokenizer = tokenizer
        self.file_path_src = file_path_src
        self.file_path_tgt = file_path_tgt
        self.offsets = offsets

    def process_line(self, line_src, line_tgt):
        """
        if len(line) == 0 or line.isspace() or not len(line.split(' ||| ')) == 2:
            return None
        """
        if len(line_src) == 0 or len(line_tgt) == 0:
            return None

        sent_src, sent_tgt = line_src.strip().split(), line_tgt.strip().split()
        token_src, token_tgt = [self.tokenizer.tokenize(word) for word in sent_src], [self.tokenizer.tokenize(word) for
                                                                                 word in sent_tgt]
        wid_src, wid_tgt = [self.tokenizer.convert_tokens_to_ids(x) for x in token_src], [
            self.tokenizer.convert_tokens_to_ids(x) for x in token_tgt]

        ids_src, ids_tgt = self.tokenizer.prepare_for_model(list(itertools.chain(*wid_src)), return_tensors='pt',
                                                       max_length=512)['input_ids'], \
                           self.tokenizer.prepare_for_model(list(itertools.chain(*wid_tgt)), return_tensors='pt',
                                                       max_length=512)['input_ids']


        bpe2word_map_src = []
        for i, word_list in enumerate(token_src):
            bpe2word_map_src += [i for x in word_list]
        bpe2word_map_tgt = []
        for i, word_list in enumerate(token_tgt):
            bpe2word_map_tgt += [i for x in word_list]
        return (ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt, sent_src, sent_tgt)

    def __iter__(self):

        f_src = open(self.file_path_src, encoding="utf-8")
        f_tgt = open(self.file_path_tgt, encoding="utf-8")
        i = 0
        for line_src, line_tgt in zip(f_src, f_tgt):
            i = i+1
            if line_src and line_tgt:

                processed = self.process_line(line_src, line_tgt)
                if processed is None:
                    print(
                        f'Line "{line_src.strip()}" (offset in bytes: {f_src.tell()}) is not in the correct format. Skipping...')
                    empty_tensor = torch.tensor([self.tokenizer.cls_token_id, 999, self.tokenizer.sep_token_id])
                    empty_sent = ''
                    yield (empty_tensor, empty_tensor, [-1], [-1], empty_sent, empty_sent)
                else:
                    yield processed






def word_align(align_layer, device, keep_layer ,extraction, softmax_threshold, per_gpu_train_batch_size, tokenizer, model, folder_path, src_path, tgt_path,
               output_prob=True):

    device = torch.device(device)
    def collate(examples):
        ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt, sents_src, sents_tgt = zip(*examples)
        ids_src = pad_sequence(ids_src, batch_first=True, padding_value=tokenizer.pad_token_id)
        ids_tgt = pad_sequence(ids_tgt, batch_first=True, padding_value=tokenizer.pad_token_id)
        return ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt, sents_src, sents_tgt


    dataset = LineByLineTextDataset(tokenizer, file_path_src=src_path, file_path_tgt=tgt_path)
    dataloader = DataLoader(
        dataset, batch_size=per_gpu_train_batch_size, collate_fn=collate
    )

    tqdm_iterator = trange(0, desc="Extracting")
    model_sentence = SentenceAligner_word(softmax_threshold=softmax_threshold, model=model)


    word_aligns_list_all_layer_dic = {}
    model.eval()
    for batch in tqdm(dataloader):
        with torch.no_grad():
            ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt, sents_src, sents_tgt = batch


            ids_src, ids_tgt = ids_src.to(device), ids_tgt.to(device)
            word_aligns_list_all_layer_dic_one_batch = model_sentence.get_aligned_word(align_layer,
                                                                                       extraction,
                                                                                       softmax_threshold,
                                                                                       ids_src,
                                                                                       ids_tgt,
                                                                                       bpe2word_map_src,
                                                                                       bpe2word_map_tgt,
                                                                                       tokenizer.pad_token_id,
                                                                                       tokenizer.cls_token_id,
                                                                                       tokenizer.sep_token_id,
                                                                                       output_prob = True)
            for layer_id in word_aligns_list_all_layer_dic_one_batch:
                if layer_id not in word_aligns_list_all_layer_dic:
                    word_aligns_list_all_layer_dic[layer_id] = word_aligns_list_all_layer_dic_one_batch[layer_id]
                else:
                    word_aligns_list_all_layer_dic[layer_id] = word_aligns_list_all_layer_dic[layer_id] + word_aligns_list_all_layer_dic_one_batch[layer_id]


            # On extrait toutes les couches pour le txt mais on retourne seulement celle visée.
            for layer_id in word_aligns_list_all_layer_dic:
                with open(os.path.join(folder_path, f'{"XX2XX.align"}.{str(layer_id)}'), 'w', encoding='utf-8') as writers:
                    for word_aligns in word_aligns_list_all_layer_dic[layer_id]:
                        output_str = []
                        if output_prob:
                            # word_aligns est un dict {(src, tgt): prob}
                            for (src_idx, tgt_idx), prob in word_aligns.items():
                                if src_idx != -1:
                                    output_str.append(f'{src_idx}-{tgt_idx}:{prob:.4f}')
                        else:
                            # word_aligns est un set de tuples (src, tgt)
                            for word_align in word_aligns:
                                if word_align[0] != -1:
                                    output_str.append(f'{word_align[0]}-{word_align[1]}')
                        writers.write(' '.join(output_str) + '\n')
                if str(keep_layer) == str(layer_id):
                    corresponding_dict = word_aligns_list_all_layer_dic[layer_id]
    return corresponding_dict


from aligner.word_align import ComponentAligner

def word_align_components(args, tokenizer, model, folder_path, src_path, tgt_path):
    """
    Extraction d'alignements par composantes connexes.
    Format de sortie : mots_src ||| mots_tgt ||| score ||| type
    où type = '1:1', 'n:1', '1:m', 'n:m'
    """
    device = torch.device(args.device)

    def collate(examples):
        ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt, sents_src, sents_tgt = zip(*examples)
        ids_src = pad_sequence(ids_src, batch_first=True, padding_value=tokenizer.pad_token_id)
        ids_tgt = pad_sequence(ids_tgt, batch_first=True, padding_value=tokenizer.pad_token_id)
        return ids_src, ids_tgt, bpe2word_map_src, bpe2word_map_tgt, sents_src, sents_tgt

    dataset = LineByLineTextDataset(tokenizer, file_path_src=src_path, file_path_tgt=tgt_path)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate)

    aligner = ComponentAligner(args, model)
    model.eval()


    os.makedirs(folder_path, exist_ok=True)
    output_file = os.path.join(folder_path, 'XX2XX.components_align')

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for batch in tqdm(dataloader, desc="Component alignment"):
            with torch.no_grad():
                ids_src, ids_tgt, b2w_src_b, b2w_tgt_b, sents_src, sents_tgt = batch
                ids_src = ids_src.to(device)
                ids_tgt = ids_tgt.to(device)

                # batch_size=1, donc on prend l'index 0
                b2w_src = b2w_src_b[0]
                b2w_tgt = b2w_tgt_b[0]
                sent_src = sents_src[0]
                sent_tgt = sents_tgt[0]

                # Lancer l'alignement
                blocks = aligner.align_sentence_pair(
                    args, ids_src[0], ids_tgt[0], b2w_src, b2w_tgt,
                    tokenizer.pad_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id
                )

                # Écrire les résultats
                for src_words, tgt_words, score in blocks:
                    # Récupérer les vrais mots
                    src_str = ' '.join(sent_src[w] for w in src_words if w < len(sent_src))
                    tgt_str = ' '.join(sent_tgt[w] for w in tgt_words if w < len(sent_tgt))

                    # Type d'alignement
                    n, m = len(src_words), len(tgt_words)
                    if n == 1 and m == 1:
                        align_type = '1:1'
                    elif n == 1:
                        align_type = '1:m'
                    elif m == 1:
                        align_type = 'n:1'
                    else:
                        align_type = 'n:m'
                    f_out.write(f"{src_str} ||| {tgt_str} ||| {score:.4f} ||| {align_type}\n")