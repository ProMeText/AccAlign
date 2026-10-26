# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
# Modifications copyright (C) 2020, Zi-Yi Dou
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
import glob
import json
import logging
import os
import random
import re
from typing import Dict, List, Tuple
from tqdm import tqdm
import copy

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from self_training_modeling_adapter import BertForSO
import adapters
from adapters import DoubleSeqBnConfig
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoConfig, AutoModel, get_linear_schedule_with_warmup
from train_utils import _sorted_checkpoints, _rotate_checkpoints, WEIGHTS_NAME

logger = logging.getLogger(__name__)

import itertools
from aligner.sent_aligner import word_align





def set_seed(seed):
	if seed >= 0:
		random.seed(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)



def align(align_layer,
		  extraction='softmax',
		  eval_data_file_src='',
		  keep_layer=6,
		  infer_data_file_src='',
		  infer_data_file_tgt='',
		  output_dir_adapter='',
		  # Connexion des composants
		  use_components=True,
		  component_min_edge=0.01,
		  component_threshold=0.08,
		  component_exclude_1to1=False,
		  component_output='infer_output/components/',
		  infer_path='infer_output/',
		  model_name_or_path=None,
		  adapter_path=None,
		  tokenizer_name=None,
		  cache_dir=None,
		  softmax_threshold=0.1,
		  block_size=-1,
		  max_len=512,
		  do_eval=False,
		  do_test=False,
		  per_gpu_eval_batch_size=32,
		  no_cuda=False,
		  overwrite_output_dir=False,
		  seed=42,
		  fp16=False,
		  local_rank=0
		  ):

	if eval_data_file_src is None and do_eval:
		raise ValueError(
			"Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
			"or remove the --do_eval argument."
		)


	if (
			os.path.exists(output_dir_adapter)
			and os.listdir(output_dir_adapter)
			and not overwrite_output_dir
	):
		raise ValueError(
			"Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
				output_dir_adapter
			)
		)

	# Setup CUDA, GPU & distributed training
	if local_rank == -1 or no_cuda:
		device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
		n_gpu = 0 if no_cuda else torch.cuda.device_count()
	else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
		torch.cuda.set_device(local_rank)
		device = torch.device("cuda", local_rank)
		torch.distributed.init_process_group(backend="nccl")
		n_gpu = 1
	device = device

	# Setup logging
	logging.basicConfig(
		format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
		datefmt="%m/%d/%Y %H:%M:%S",
		level=logging.INFO if local_rank in [-1, 0] else logging.WARN,
	)
	logger.warning(
		"Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
		local_rank,
		device,
		n_gpu,
		bool(local_rank != -1),
		fp16,
	)

	# Set seed
	set_seed(seed)

	# Load pretrained model and tokenizer
	if local_rank not in [-1, 0]:
		torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

	# config_class, model_class, tokenizer_class = BertConfig, BertForSO, BertTokenizer
	modelforALING, tokenizer_class = BertForSO, AutoTokenizer
	config = AutoConfig.from_pretrained(model_name_or_path, attn_implementation="eager")

	if tokenizer_name:
		tokenizer = tokenizer_class.from_pretrained(tokenizer_name, cache_dir=cache_dir)
	elif model_name_or_path:
		tokenizer = tokenizer_class.from_pretrained(model_name_or_path, cache_dir=cache_dir)
	else:
		raise ValueError(
			"You are instantiating a new {} tokenizer. This is not supported, but you can do it from another script, save it,"
			"and load it from here, using --tokenizer_name".format(tokenizer_class.__name__)
		)

	if block_size <= 0:
		block_size = max_len
		# Our input block size will be the max possible for the model
	else:
		block_size = min(block_size, max_len)

	labse_model = AutoModel.from_pretrained(model_name_or_path, output_hidden_states=True)
	adapters.init(labse_model)  # requis: oriente add_adapter/load_adapter vers 'adapters', pas PEFT


	if local_rank == 0:
		torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab


	

	if do_test:
		# extract word alignment for all layers
		if adapter_path:
			adapter_name = labse_model.load_adapter(adapter_path)
			labse_model.set_active_adapters(adapter_name)
		model = modelforALING(config, labse_model)
		model.to(device)

		folder_path = infer_path
		if not os.path.exists(folder_path):
			os.makedirs(folder_path)
		if use_components:
			# Créer un namespace args pour ComponentAligner
			class Args:
				pass

			comp_args = Args()
			comp_args.align_layer = align_layer
			comp_args.extraction = extraction
			comp_args.device = device
			comp_args.component_min_edge = component_min_edge
			comp_args.component_min_edge = 0.35
			comp_args.component_threshold = component_threshold
			comp_args.component_threshold = 0.0
			comp_args.max_ngram = 6
			comp_args.component_exclude_1to1 = component_exclude_1to1
			comp_args.device = device

			from aligner.sent_aligner import word_align_components
			word_align_components(
				comp_args, tokenizer, model,
				component_output if component_output else folder_path,
				infer_data_file_src, infer_data_file_tgt
			)
		else:
			alignments = word_align(align_layer=align_layer, device=device, keep_layer=keep_layer, extraction=extraction, softmax_threshold=softmax_threshold,
					   per_gpu_train_batch_size=per_gpu_eval_batch_size, tokenizer=tokenizer, model=model, folder_path=folder_path,
					   src_path=infer_data_file_src, tgt_path=infer_data_file_tgt, output_prob=True)
		return alignments


def visualize(alignments, src, tgt, output_file):
	with open(src, 'r', encoding='utf-8') as f:
		src_sentences = [normalize_text(line.strip()) for line in f.readlines()]

	with open(tgt, 'r', encoding='utf-8') as f:
		tgt_sentences = [normalize_text(line.strip()) for line in f.readlines()]

	with open(output_file, "w") as output_file:
		for i, (src_sent, tgt_sent) in tqdm(enumerate(zip(src_sentences, tgt_sentences)), total=len(src_sentences)):
			output_file.write("\n\n---\n")
			output_file.write(f"Src: {src_sent}\n")
			output_file.write(f"Tgt: {tgt_sent}\n")
			if i < len(alignments):
				# Convertir les alignements en paires de mots
				align_pairs = alignments[i]
				for pair, probs in align_pairs.items():
					src_idx, tgt_idx = pair
					try:
						src_word = src_sent.split()[int(src_idx)]
						tgt_word = tgt_sent.split()[int(tgt_idx)]
						# Écrire chaque paire d'alignement sur une ligne séparée
						output_file.write(f"{src_word} <=> {tgt_word} ({probs})\n")
					except (IndexError, ValueError):
						continue  # Skip invalid indices


def normalize_text(text):
	"""Normalize text by removing special characters and normalizing spaces"""
	text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
	text = re.sub(r'\s+', ' ', text).strip()  # Normalize spaces
	return text

def json_to_txt():
	input_dir = "/home/mgl/Bureau/Travail/projets/alignement/alignement_multilingue/medieval-multilingual-parallel-corpus/data/*/*.json"
	all_pairs = []
	for file in glob.glob(input_dir):
		print(file)
		json_file = json.load(open(file, 'r'))
		for _, value in json_file['alignement_id'].items():
			alignments = value[0]
			all_possible_pairs = list(itertools.combinations(alignments.values(), 2))
			all_pairs.extend(all_possible_pairs)

	with open('data/lancelot/source.txt', 'w') as f:
		for pair in all_pairs:
			f.write(pair[0] + '\n')
	with open('data/lancelot/target.txt', 'w') as f:
		for pair in all_pairs:
			f.write(pair[1] + '\n')

if __name__ == "__main__":
	json_to_txt()
	src = 'data/tests/source.txt'
	tgt = 'data/tests/target.txt'
	# src = 'data/lancelot/source.txt'
	# tgt = 'data/lancelot/target.txt'
	for threshold in [0.0001]:
	# for threshold in [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.5]:
	#for threshold in [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0]:
		alignments = align(
			eval_data_file_src='',
			infer_data_file_src=src,
			infer_data_file_tgt=tgt,
			output_dir_adapter='',
			infer_path='infer_output/',
			model_name_or_path="models/model_labse_bibles_clean/",
			# model_name_or_path="LaBSE/",
			# adapter_path="checkpoint-adapter/",
			# adapter_path=None,
			use_components=False,
			tokenizer_name=None,
			cache_dir=None,
			keep_layer=6,
			block_size=-1,
			max_len=512,
			softmax_threshold=threshold,
			do_eval=False,
			do_test=True,
			per_gpu_eval_batch_size=32,
			no_cuda=True,
			overwrite_output_dir=False,
			seed=42,
			fp16=False,
			local_rank=-1,
			align_layer=12
		)
		print(alignments)
		visualize(alignments=alignments, src=src, tgt=tgt, output_file='results/result_' + str(threshold) + '.txt')