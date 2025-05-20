# Create dataclasses for parsing arguments

from inspect import getargs
import os
import json
import argparse
from typing import Optional, Literal, List, Dict, Any

import pydantic
from transformers import set_seed

# from . import log_utils
from .utils import common_utils, log_utils


_SEED = 124124

GOAL_OPTIONS = ['source', 'target']


class DebugConfig(pydantic.BaseModel):
	train_num_entries: int = 40
	test_num_entries: int = 20
	train_batch_size: int = 2
	test_batch_size: int = 2


class TrainingConfig(pydantic.BaseModel):
	epochs: int = 1
	learning_rate: float = 5e-05
	warmup_ratio: float = 0.0
	train_batch_size: int = 4
	test_batch_size: int = 4


class DataConfig(pydantic.BaseModel):
	goal: Literal['source', 'target']
	train_dataset: Optional[str] = None
	test_dataset: str = 'fixed_original_test'
	input_turn_masking: Literal['none', 'assistant', 'all']
	input_task_instruction: Literal['system', 'user']
	input_unify_repair: bool = True
	""" either output a single bbox, or a bbox and a repair """
	output_repair: bool
	output_repair_form: Literal['word', 'sentence']

	class Config:
		use_enum_values = True


class Config(pydantic.BaseModel):
	base_model: str = "HuggingFaceM4/idefics2-8b"
	load_model: Literal['lora', 'qlora', ''] = 'qlora'
	task: Literal['sft', 'dpo', 'eval', 'convert'] = 'sft'
	do_train: bool = False
	do_eval: bool = True
	run_name: str
	goal: str
	seed: int = _SEED
	deterministic: bool = True
	output_dir_model: str = None
	output_dir_results: str = None
	use_evaluation_cache: bool = True
	debug: Optional[DebugConfig] = None
	training: TrainingConfig = TrainingConfig()
	data: DataConfig

	def to_dict(self) -> dict:
		return args_to_dict(self)


def _add_config(parser, model):
	# Add Pydantic model to an ArgumentParser
	fields = model.__fields__
	for name, field in fields.items():
		parser.add_argument(
			f"--{name}",
			dest=name,
			type=field.type_,
			default=field.default,
			help=field.field_info.description,
		)


def _parse_args(*, required: List[str] = None, defaults: Dict[str,Any] = None, description: str = None) -> argparse.Namespace:
	# Parse and get arguments
	required = required or []
	defaults = defaults or {}

	parser = argparse.ArgumentParser(description=description)

	# Model configuration
	parser.add_argument("--base_model", type=str, default="HuggingFaceM4/idefics2-8b")
	parser.add_argument("--load_model", type=str, default="qlora", choices=['qlora', 'lora'])
	parser.add_argument("--task", type=str, choices=['sft', 'dpo', 'eval', 'convert'],
				   default=defaults.get('task', 'eval'))

	# Dataset configuration
	parser.add_argument("--train_dataset", type=str, required='train_dataset' in required, default=defaults.get('train_dataset', None))
	parser.add_argument("--test_dataset", type=str, required='test_dataset' in required, default=defaults.get('test_dataset', None))
	parser.add_argument("--goal", type=str, choices=['source', 'target'], required='goal' in required, default=defaults.get('goal', None))
	parser.add_argument("--turn_masking", type=str, choices=['none', 'assistant', 'all'], default=defaults.get('turn_masking', 'none'))
	parser.add_argument("--prompt_task_instruction", type=str, choices=['system', 'user'], default=defaults.get('prompt_task_instruction', 'system'))

	# Training parameters
	parser.add_argument("--train_epochs", type=int, default=1)
	parser.add_argument("--train_learning_rate", type=float, default=5e-05)
	parser.add_argument("--train_warmup_ratio", type=float, default=0.0)
	parser.add_argument("--train_batch_size", type=float, default=4)

	# System configuration
	parser.add_argument("--seed", type=int, default=_SEED)
	parser.add_argument("--deterministic", type=bool, default=False)
	parser.add_argument("--output_dir_model", type=str, default=None)
	parser.add_argument("--output_dir_results", type=str, default=None)
	parser.add_argument("--use_evaluation_cache", type=bool, default=True)
	parser.add_argument('--debug', default=False, action='store_true')
	parser.add_argument('--debug_num_entries', type=int, default=40)

	args = parser.parse_args()
	return args


def setup_configs(args: argparse.Namespace) -> Config:
	args.run_name = _get_run_name(args)

	assert args.test_dataset == 'fixed_original_test', f"test_dataset must be fixed_original_test, got {args.test_dataset}"

	do_train = args.task not in ['eval']
	do_eval = True
	assert do_train or do_eval		# eval is always true anyway

	# Create DataConfig
	data_config = DataConfig(
		goal=args.goal,
		train_dataset=args.train_dataset if do_train else None,
		test_dataset=args.test_dataset,
		input_turn_masking=args.turn_masking,
		input_task_instruction=args.prompt_task_instruction,
		output_repair=True,
		output_repair_form='sentence',
	)

	# Create DebugConfig if debug mode is enabled
	debug_config = DebugConfig(
		train_num_entries=args.debug_num_entries,
		test_num_entries=args.debug_num_entries//2
	) if args.debug else None

	# Create TrainingConfig
	training_config = TrainingConfig(
		epochs=args.train_epochs if getattr(args, 'train_epochs', None) else TrainingConfig.model_fields['epochs'].default,
		learning_rate=args.train_learning_rate if getattr(args, 'train_learning_rate', None) else TrainingConfig.model_fields['learning_rate'].default,
		warmup_ratio=args.train_warmup_ratio if getattr(args, 'train_warmup_ratio', None) else TrainingConfig.model_fields['warmup_ratio'].default,
		train_batch_size=args.train_batch_size if getattr(args, 'train_batch_size', None) else TrainingConfig.model_fields['train_batch_size'].default,
		test_batch_size=args.train_batch_size if getattr(args, 'train_batch_size', None) else TrainingConfig.model_fields['train_batch_size'].default,
	)

	# Set up output directories
	output_dir_model = _get_model_output_dir(args) if not args.task == 'convert' else args.output_dir_results
	output_dir_results = args.output_dir_results if getattr(args, 'output_dir_results', None) else os.path.join(output_dir_model, 'evals')

	os.makedirs(output_dir_model, exist_ok=True)
	os.makedirs(output_dir_results, exist_ok=True)

	# Create and return Config object
	config = Config(
		base_model=args.base_model,
		load_model=args.load_model,
		task=args.task,
		# Set train/eval flags - we always evaluate, but only train if sft or
		do_train=do_train,
		do_eval=do_eval,
		run_name=args.run_name,
		goal=args.goal,
		seed=args.seed,
		deterministic=getattr(args, 'deterministic', Config.model_fields['deterministic'].default),
		output_dir_model=output_dir_model,
		output_dir_results=output_dir_results,
		use_evaluation_cache=getattr(args, 'use_evaluation_cache', Config.model_fields['use_evaluation_cache'].default),
		debug=debug_config,
		training=training_config,
		data=data_config
	)

	# Log to file within model folder
	logger = log_utils.get_logger()
	log_utils.configure_file_logging(os.path.join(config.output_dir_model, 'log.txt'))
	logger.info(f"Run parameters: {config}")
	save_arguments(config)
	if args.task == 'convert':
		logger.info(f"Outputs will be saved in {config.output_dir_results}")
	else:
		logger.info(f"Checkpoints and results will be saved in {config.output_dir_model}")

	# if not config.do_train:
	# 	config.data

	# Set seed
	if config.deterministic:
		try:
			set_seed(config.seed, deterministic=config.deterministic)
		except TypeError as e:
			logger.warning(f"Failed to set seed: {e}")
			logger.warning("This is probably because you are using an older version of transformers. Set --deterministic=False to disable deterministic training and continue in older versions.")
			exit()
		os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ":4096:8")
	else:
		set_seed(config.seed)

	return config


def parse_configs_from_args(args: argparse.Namespace, mapping: Dict[str, Any] = None) -> Config:
	if mapping is None:
		mapping = {}

	for key, value in mapping.items():
		setattr(args, key, getattr(args, value, None))

	return setup_configs(args)


def parse_configs(required: List[str] = None, defaults: Dict[str,Any] = None, *, description: str = None) -> Config:
	args = _parse_args(required=required, defaults=defaults, description=description)
	return setup_configs(args)


def _get_run_name(args: argparse.Namespace) -> str:
	if common_utils.is_finetuned_model(args.base_model) and args.task == 'eval':
		tmp_str = args.base_model.replace('/checkpoint-best', '').lstrip('/').split('/')[-1]
		if args.task == 'eval':
			tmp_str += '_eval'
		if args.debug and 'debug' not in tmp_str:
			tmp_str += '_debug'

		return tmp_str
	else:
		# default run name
		return f"{common_utils.get_datetime_prefix()}_{common_utils.get_short_model_name(args.base_model)}_{args.task}{'_debug' if args.debug else ''}"


def save_arguments(arguments):
	# save arguments inside the model folder for future reference
	with open(os.path.join(arguments.output_dir_model, 'config.json'), 'w') as out_f:
		json.dump(args_to_dict(arguments), out_f, indent=4)


def args_to_dict(arguments) -> dict:
	result = {}
	for key, value in arguments.__dict__.items():
		if isinstance(value, pydantic.BaseModel):
			result[key] = value.model_dump()
		elif isinstance(value, (str, int, float, bool, list, dict)):
			result[key] = value
		else:
			# For any other types, you might want to add specific handling
			# or use a default serialization method
			result[key] = str(value)
	return result


def _get_model_output_dir(args):
	"""Helper function to determine the model output directory"""
	if getattr(args, 'output_dir_model', None):
		return args.output_dir_model

	if common_utils.is_finetuned_model(args.base_model) and args.task == 'eval':
		return args.base_model.replace('/checkpoint-best', '')

	return os.path.join(
		os.environ['OUTPUT_DIR'].replace('output', f"checkpoints_{args.task}"),
		args.run_name
	)
