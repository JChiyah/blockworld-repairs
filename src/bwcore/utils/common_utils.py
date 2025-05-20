
import datetime
from typing import Optional


def get_datetime_prefix() -> str:
	return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


_MODEL_NAMES_SHORT = {
	'HuggingFaceM4/idefics2-8b': 'idefics2-8b',
	'HuggingFaceM4/idefics2-8b-chatty': 'idefics2-8b-chatty',
	'liuhaotian/llava-v1.5-7b': 'llava-v1.5-7b',
	'gpt-4o': 'gpt-4o',
}


def get_short_model_name(model_name: str) -> Optional[str]:
	# check if model short name is in model_name
	# for short_name in _MODEL_NAMES_SHORT.values():
	# 	if short_name in model_name:
	# 		return short_name
	return _MODEL_NAMES_SHORT.get(model_name, None)


def is_finetuned_model(model_name: str) -> bool:
	# todo: this is a quick hack for now
	return get_short_model_name(model_name) is None


def get_short_model_task(model_task: str) -> str:
	if model_task == 'zeroshot':
		return 'zs'
	elif model_task == 'finetune':
		return 'ft'
	else:
		return model_task


def get_short_dataset_name(dataset_name):
	if dataset_name == 'fixed_original_test_instructions':
		return 'instructions'
	elif dataset_name == 'fixed_original_test_corrections':
		return 'corrections'
	elif dataset_name == 'fixed_original_test':
		return 'full'
	else:
		return dataset_name
