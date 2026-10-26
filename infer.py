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



def main(align_layer=6,
		 extraction='softmax',
		eval_data_file_src='',
		infer_data_file_src='',
		infer_data_file_tgt='',
		output_dir_adapter='',
		infer_path='infer_output/',
		model_name_or_path="models/model_labse_bibles_clean/",
		adapter_path="checkpoint-adapter/",
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

		# folder_path= os.path.join(infer_path, f'{}2{}' + '_steps')
		folder_path = infer_path
		if not os.path.exists(folder_path):
			os.makedirs(folder_path)
		word_align(align_layer, extraction, softmax_threshold, per_gpu_eval_batch_size, tokenizer, model, folder_path, infer_data_file_src,
				   infer_data_file_tgt)


if __name__ == "__main__":
	main(
		eval_data_file_src='',
		infer_data_file_src='data/lancelot/source.txt',
		infer_data_file_tgt='data/lancelot/target.txt',
		output_dir_adapter='',
		infer_path='infer_output/',
		model_name_or_path="models/model_labse_bibles_clean/",
		adapter_path="checkpoint-adapter/",
		tokenizer_name=None,
		cache_dir=None,
		block_size=-1,
		max_len=512,
		softmax_threshold=0.1,
		do_eval=False,
		do_test=True,
		per_gpu_eval_batch_size=32,
		no_cuda=False,
		overwrite_output_dir=False,
		seed=42,
		fp16=False,
		local_rank=-1
	)		