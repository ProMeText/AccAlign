# coding=utf-8

import os
import logging
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from transformers import BertModel, BertTokenizer, XLMModel, XLMTokenizer, RobertaModel, RobertaTokenizer, \
    XLMRobertaModel, XLMRobertaTokenizer, AutoConfig, AutoModel, AutoTokenizer
import torch.nn as nn
from train_utils import get_logger

LOG = get_logger(__name__)


def return_extended_attention_mask(attention_mask, dtype):
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(
            "Wrong shape for input_ids or attention_mask"
        )
    extended_attention_mask = extended_attention_mask.to(dtype=dtype)
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask



class SentenceAligner_word(object):
    def __init__(self, softmax_threshold, model):

        self.guide = None
        self.softmax_threshold = softmax_threshold
        self.embed_loader = model

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (1, x.size(-1))
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def get_subword_matrix(self, align_layer, extraction, softmax_threshold, inputs_src, inputs_tgt, PAD_ID, CLS_ID,
                           SEP_ID, output_prob=False):

        output_src, output_tgt = self.embed_loader(
            inputs_src=inputs_src, inputs_tgt=inputs_tgt,
            attention_mask_src=(inputs_src != PAD_ID).long(),
            attention_mask_tgt=(inputs_tgt != PAD_ID).long(),
            guide=None, align_layer=align_layer,
            extraction=extraction, softmax_threshold=softmax_threshold,
            do_infer=True,
        )

        align_matrix_all_layers = {}
        align_prob_all_layers = {}  # ← AJOUTÉ

        for layer_id in range(1, len(output_src.hidden_states)):

            hidden_states_src = output_src.hidden_states[layer_id]
            hidden_states_tgt = output_tgt.hidden_states[layer_id]

            attention_mask_src = ((inputs_src == PAD_ID) + (inputs_src == CLS_ID) + (inputs_src == SEP_ID)).float()
            attention_mask_tgt = ((inputs_tgt == PAD_ID) + (inputs_tgt == CLS_ID) + (inputs_tgt == SEP_ID)).float()
            len_src = torch.sum(1 - attention_mask_src, -1)
            len_tgt = torch.sum(1 - attention_mask_tgt, -1)
            attention_mask_src = return_extended_attention_mask(1 - attention_mask_src, hidden_states_src.dtype)
            attention_mask_tgt = return_extended_attention_mask(1 - attention_mask_tgt, hidden_states_tgt.dtype)

            query_src = self.transpose_for_scores(hidden_states_src)
            query_tgt = self.transpose_for_scores(hidden_states_tgt)

            attention_scores = torch.matmul(query_src, query_tgt.transpose(-1, -2))
            attention_scores_src = attention_scores + attention_mask_tgt
            attention_scores_tgt = attention_scores + attention_mask_src.transpose(-1, -2)

            attention_probs_src = nn.Softmax(dim=-1)(attention_scores_src)
            attention_probs_tgt = nn.Softmax(dim=-2)(attention_scores_tgt)

            if self.guide is None:
                threshold = self.softmax_threshold
                align_matrix = (attention_probs_src > threshold) * (attention_probs_tgt > threshold)

                if output_prob:
                    # Probabilité d'alignement = moyenne des deux directions
                    # (ou tu peux utiliser la moyenne harmonique comme dans le code original commenté)
                    align_prob = (attention_probs_src + attention_probs_tgt) / 2.0
                    align_matrix_all_layers[layer_id] = align_matrix
                    align_prob_all_layers[layer_id] = align_prob  # ← AJOUTÉ
                else:
                    align_matrix_all_layers[layer_id] = align_matrix

        if output_prob:
            return align_matrix_all_layers, align_prob_all_layers  # ← AJOUTÉ
        return align_matrix_all_layers

    def get_aligned_word(self, align_layer, extraction, softmax_threshold, inputs_src, inputs_tgt,
                         bpe2word_map_src, bpe2word_map_tgt, PAD_ID, CLS_ID, SEP_ID, output_prob=False):

        if output_prob:
            attention_probs_inter_all_layers, alignment_probs_all_layers = self.get_subword_matrix(
                align_layer, extraction, softmax_threshold, inputs_src, inputs_tgt, PAD_ID, CLS_ID, SEP_ID, output_prob
            )
        else:
            attention_probs_inter_all_layers = self.get_subword_matrix(
                align_layer, extraction, softmax_threshold, inputs_src, inputs_tgt, PAD_ID, CLS_ID, SEP_ID, output_prob
            )

        word_aligns_all_layers = {}

        for layer_id in attention_probs_inter_all_layers:

            attention_probs_inter = attention_probs_inter_all_layers[layer_id].float()
            word_aligns = []
            attention_probs_inter = attention_probs_inter[:, 0, 1:-1, 1:-1]

            if output_prob:
                # Récupère les probas pour cette couche et retire CLS/SEP
                alignment_probs = alignment_probs_all_layers[layer_id][:, 0, 1:-1, 1:-1]

            for idx, (attention, b2w_src, b2w_tgt) in enumerate(
                    zip(attention_probs_inter, bpe2word_map_src, bpe2word_map_tgt)):

                aligns = set() if not output_prob else dict()
                non_zeros = torch.nonzero(attention)

                for i, j in non_zeros:
                    word_pair = (b2w_src[i], b2w_tgt[j])
                    if output_prob:
                        prob = alignment_probs[idx, i, j].item()
                        if word_pair not in aligns:
                            aligns[word_pair] = prob
                        else:
                            aligns[word_pair] = max(aligns[word_pair], prob)
                    else:
                        aligns.add(word_pair)

                word_aligns.append(aligns)

            word_aligns_all_layers[layer_id] = word_aligns

        return word_aligns_all_layers


class ComponentAligner:
    def __init__(self, args, model):
        self.embed_loader = model
        self.min_edge = getattr(args, 'component_min_edge')
        self.block_threshold = getattr(args, 'component_threshold')
        self.max_ngram = getattr(args, 'max_ngram')
        self.device = args.device

    def get_prob_matrix(self, args, inputs_src, inputs_tgt, PAD_ID, CLS_ID, SEP_ID, device=None):
        """Calcule la matrice de probabilités softmax (float, pas booléen)."""
        if device is None:
            device = inputs_src.device

        # Contourner BertForSO pour forcer output_hidden_states=True
        # self.embed_loader est BertForSO, .model est le LaBSE sous-jacent
        base_model = self.embed_loader.model if hasattr(self.embed_loader, 'model') else self.embed_loader

        output_src = base_model(
            inputs_src,
            attention_mask=(inputs_src != PAD_ID).long(),
            output_hidden_states=True,  # ← FORCER ICI
        )
        output_tgt = base_model(
            inputs_tgt,
            attention_mask=(inputs_tgt != PAD_ID).long(),
            output_hidden_states=True,  # ← FORCER ICI
        )

        # Vérifications de sécurité
        if output_src.hidden_states is None:
            raise ValueError("Le modèle ne retourne pas hidden_states. Vérifiez que c'est bien un modèle compatible.")

        n_layers = len(output_src.hidden_states)
        if args.align_layer >= n_layers:
            raise ValueError(
                f"align_layer={args.align_layer} hors limites. Le modèle a {n_layers} couches (0 à {n_layers - 1}).")

        hidden_states_src = output_src.hidden_states[args.align_layer].to(device)
        hidden_states_tgt = output_tgt.hidden_states[args.align_layer].to(device)

        # Masques
        attention_mask_src = ((inputs_src == PAD_ID) + (inputs_src == CLS_ID) + (inputs_src == SEP_ID)).float()
        attention_mask_tgt = ((inputs_tgt == PAD_ID) + (inputs_tgt == CLS_ID) + (inputs_tgt == SEP_ID)).float()
        attention_mask_src = return_extended_attention_mask(1 - attention_mask_src, hidden_states_src.dtype)
        attention_mask_tgt = return_extended_attention_mask(1 - attention_mask_tgt, hidden_states_tgt.dtype)

        def transpose_for_scores(x):
            new_x_shape = x.size()[:-1] + (1, x.size(-1))
            return x.view(*new_x_shape).permute(0, 2, 1, 3)

        query_src = transpose_for_scores(hidden_states_src)
        query_tgt = transpose_for_scores(hidden_states_tgt)

        attention_scores = torch.matmul(query_src, query_tgt.transpose(-1, -2))
        attention_scores_src = attention_scores + attention_mask_tgt
        attention_scores_tgt = attention_scores + attention_mask_src.transpose(-1, -2)

        probs_src = nn.Softmax(dim=-1)(attention_scores_src)
        probs_tgt = nn.Softmax(dim=-2)(attention_scores_tgt)
        probs = (probs_src + probs_tgt) / 2.0

        # Retirer CLS/SEP et retourner sur CPU pour l'agrégation
        return probs[:, 0, 1:-1, 1:-1].detach().cpu()

    def aggregate_subwords_to_words(self, prob_matrix, b2w_src, b2w_tgt):
        """Agrège les probabilités sous-mots → mots (max pooling)."""
        n_words_src = max(b2w_src) + 1 if b2w_src else 0
        n_words_tgt = max(b2w_tgt) + 1 if b2w_tgt else 0

        word_probs = torch.zeros(n_words_src, n_words_tgt, dtype=torch.float32)

        # Construire les index sous-mots par mot
        src_word_to_subwords = {}
        for idx, w in enumerate(b2w_src):
            if w != -1:
                src_word_to_subwords.setdefault(w, []).append(idx)

        tgt_word_to_subwords = {}
        for idx, w in enumerate(b2w_tgt):
            if w != -1:
                tgt_word_to_subwords.setdefault(w, []).append(idx)

        # Max pooling
        for wi in range(n_words_src):
            for wj in range(n_words_tgt):
                if wi in src_word_to_subwords and wj in tgt_word_to_subwords:
                    # Convertir en tenseurs et utiliser torch.ix_ pour indexation croisée
                    src_idx = torch.tensor(src_word_to_subwords[wi], dtype=torch.long)
                    tgt_idx = torch.tensor(tgt_word_to_subwords[wj], dtype=torch.long)
                    # torch.ix_ crée une grille 2D : toutes les combinaisons src×tgt
                    subword_probs = prob_matrix[src_idx][:, tgt_idx]
                    word_probs[wi, wj] = subword_probs.float().max()

        return word_probs


    def align_sentence_pair(self, args, ids_src, ids_tgt, b2w_src, b2w_tgt,
                            PAD_ID, CLS_ID, SEP_ID):
        """Pipeline principal : retourne les blocs n:m avec leurs scores."""
        # 1. Matrice de probabilités (sous-mots)

        probs = self.get_prob_matrix(args, ids_src.unsqueeze(0), ids_tgt.unsqueeze(0),
                                      PAD_ID, CLS_ID, SEP_ID)
        prob_matrix = probs[0].detach().cpu()  # [len_src_sub, len_tgt_sub], float

        # 2. Agréger sous-mots → mots (toujours float)
        word_probs = self.aggregate_subwords_to_words(prob_matrix, b2w_src, b2w_tgt)
        n_words_src, n_words_tgt = word_probs.shape

        # 3. Construire le graphe avec union-find (pas d'argmax !)
        parent = list(range(n_words_src + n_words_tgt))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        # Ajouter les arêtes au-dessus du seuil minimal
        edges = []
        for i in range(n_words_src):
            for j in range(n_words_tgt):
                p = word_probs[i, j].item()
                if p > self.min_edge:
                    edges.append((i, j, p))
                    union(i, n_words_src + j)

        # 4. Regrouper par composante connexe
        components = {}
        for i, j, p in edges:
            root = find(i)
            if root not in components:
                components[root] = {'src': set(), 'tgt': set(), 'edges': []}
            components[root]['src'].add(i)
            components[root]['tgt'].add(j)
            components[root]['edges'].append(p)

        # 5. Calculer les scores des blocs et filtrer
        results = []
        for comp in components.values():
            if not comp['src'] or not comp['tgt']:
                continue

            src_words = sorted(comp['src'])
            tgt_words = sorted(comp['tgt'])
            n, m = len(src_words), len(tgt_words)

            # Limiter la taille
            if n > self.max_ngram or m > self.max_ngram:
                continue

            # Score du bloc = moyenne des probabilités des arêtes
            block_score = sum(comp['edges']) / len(comp['edges'])

            # Filtrage : garder seulement si score > seuil du bloc
            if block_score > self.block_threshold:
                results.append((src_words, tgt_words, block_score))

        # Trier par score décroissant
        results.sort(key=lambda x: x[2], reverse=True)
        return results