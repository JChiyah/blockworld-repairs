"""
Author: Javier Chiyah-Garcia
Script to convert the BlockWorld-Repairs dataset into LLaVA format for training and evaluation.

The script processes Block World entries and converts them into a format compatible with LLaVA. It handles both source (block) and target (location) bounding boxes, and supports different conversation masking strategies.

Usage:
    python convert_dataset_to_llava_format.py --output_dir_results /path/to/output --train_dataset dataset_name

Use it within the run_llava1.5.sh script to convert the dataset to LLaVA format automatically before eval/fine-tuning.
"""

import os
import json
from typing import Dict, List, Optional

import bwcore
bwcore.configure_logging(logger_level=bwcore.log_utils.DEBUG)
logger = bwcore.get_logger()

from bwcore.utils import unity_utils


def _get_instruction_and_prompts(goal: str) -> tuple[str, callable, callable]:
	"""
	Returns the appropriate instruction and prompt generation functions based on the goal.

	Args:
		goal: Either 'source' (block) or 'target' (location)

	Returns:
		tuple containing:
		- instruction text
		- function to generate true bounding box prompt
		- function to generate candidate bounding box prompt
	"""
	if goal == 'source':
		instruction = "What is the bounding box of the block mentioned in the following command?"
		generation_prompt = lambda x: \
			f"block bounding box {unity_utils.bbox_as_str(bwcore.data.normalise_bbox(x.unity.true_source_bbox))}"
		cand_generation_prompt = lambda x: \
			f"block bounding box {unity_utils.bbox_as_str(bwcore.data.normalise_bbox(x.cand_source_bbox))}"
	elif goal == 'target':
		instruction = "What is the bounding box of the location mentioned in the following command?"
		generation_prompt = lambda x: \
			f"location bounding box {unity_utils.bbox_as_str(bwcore.data.normalise_bbox(x.unity.true_target_bbox))}"
		cand_generation_prompt = lambda x: \
			f"location bounding box {unity_utils.bbox_as_str(bwcore.data.normalise_bbox(x.cand_target_bbox))}"
	else:
		raise ValueError(f"Invalid goal: {goal}")

	return instruction, generation_prompt, cand_generation_prompt


def _create_conversation(
		entry: bwcore.data.BWEntry,
		initial_text: str,
		generation_prompt: Optional[callable],
		candidate_generation_prompt: Optional[callable],
		turn_masking: Optional[str]) -> List[Dict]:
	"""
	Creates a conversation in LLaVA format from a Block World entry.

	Args:
		entry: Block World entry containing the conversation
		initial_text: initial text to prepend to the first message
		generation_prompt: expected response prompt callable
		candidate_generation_prompt: candidate response prompt callable
		turn_masking: Strategy for masking turns ('assistant', 'all', or None)

	Returns:
		List of conversation turns in LLaVA format
	"""
	conversation = []
	if initial_text:
		initial_text = f"\n{initial_text}"

	for msg in entry.messages:
		# Handle first user message specially to include image
		if msg.role == 'user' and len(conversation) == 0:
			text = f"<image>{initial_text}"
		else:
			text = ''

		# Process message content
		for cnt in msg.content:
			if cnt.type != 'image':
				text = f"{text}{' ' if len(text) > 0 else ''}{cnt.value}"

		# Replace placeholder with candidate bounding box if present
		if '<img_current>' in text:
			text = text.replace('<img_current>', candidate_generation_prompt(entry) + ' ')

		# Determine if this turn should be masked during training
		mask = False
		if turn_masking:
			if (turn_masking == 'assistant' and msg.role == 'assistant') or (turn_masking == 'all'):
				mask = True

		conversation.append({
			'from': 'human' if msg.role == 'user' else 'gpt',
			'value': text.strip(),
			'mask_during_train': mask
		})

	# Add the generation prompt as the final assistant message
	if generation_prompt:
		conversation.append({
			'from': 'gpt',
			'value': generation_prompt(entry),
			'mask_during_train': False
		})

	return conversation

def convert_to_llava_json(
		entries: List[bwcore.data.BWEntry],
		goal: str,
		turn_masking: str = None) -> List[dict]:
	"""
	Converts a list of Block World entries to LLaVA JSON format.

	Args:
		entries: List of Block World entries to convert
		goal: Either 'source' (block) or 'target' (location)
		turn_masking: Strategy for masking turns ('assistant', 'all', or None)

	Returns:
		List of entries in LLaVA JSON format
	"""
	if turn_masking == 'none':
		turn_masking = None

	instruction, generation_prompt, candidate_generation_prompt = _get_instruction_and_prompts(goal)

	final_data = []
	max_num_entries = config.debug.test_num_entries if config.debug else None
	for entry in entries:
		# Skip entries that don't match the goal type
		if (not entry.has_source and goal == 'source') or (not entry.has_target and goal == 'target'):
			continue

		final_data.append({
			'id': entry.entry_idx,
			'image': entry.image_with_resolution,
			'conversations': _create_conversation(
				entry, "", generation_prompt, candidate_generation_prompt, turn_masking),
			'answer': generation_prompt(entry)
		})
		if max_num_entries and len(final_data) >= max_num_entries:
			break

	return final_data

def convert_to_llava_format(
		output_dir: str, dataset_name: str, turn_masking: str = None):
	"""
	Converts a Block World dataset to LLaVA format and saves it to disk.

	Args:
		output_dir: Directory to save the converted data
		dataset_name: Name of the dataset to convert
		turn_masking: Strategy for masking turns ('assistant', 'all', or None)
	"""
	all_entries = bwcore.data.get_bw_dataset(dataset_name)

	for goal in bwcore.configs.GOAL_OPTIONS:
		for split in ['test', 'train']:
			entries = getattr(all_entries, f"_{split}")
			llava_json = convert_to_llava_json(entries, goal, turn_masking=turn_masking)

			os.makedirs(output_dir, exist_ok=True)
			output_file = os.path.join(output_dir, f"{dataset_name}-{goal}-{split}.json")
			with open(output_file, 'w') as out_f:
				json.dump(llava_json, out_f, indent=4)

			logger.debug(f"Data saved in '{output_file}'")

if __name__ == "__main__":
	# Get CLI arguments into the Config object to use throughout the script
	config: bwcore.configs.Config = bwcore.configs.parse_configs(
		description='Converts data into a format that can be used by the LLaVA model',
		required=['train_dataset', 'output_dir_results'],
		defaults={
			'task': 'convert',
			'goal': 'source',  # this is ignored anyway, as we generate both
			'test_dataset': 'fixed_original_test', # also ignored
			'turn_masking': 'none',
			'prompt_task_instruction': 'system',
		}
	)

	logger.info(f"{'=' * 10} Starting convert_dataset_to_llava_format.py {'=' * 10}")

	assert config.data.input_task_instruction == 'system', "Only system instruction is supported!"

	if 'llava' not in config.output_dir_results:
		config.output_dir_results = os.path.join(config.output_dir_results, 'llava')

	config.output_dir_results = os.path.join(config.output_dir_results, config.data.input_turn_masking)

	convert_to_llava_format(config.output_dir_results, config.data.train_dataset, config.data.input_turn_masking)

	logger.info(f"Data converted to LLAVA format and saved to '{config.output_dir_results}'")
	logger.info(f"{'=' * 10} Finished convert_dataset_to_llava_format.py {'=' * 10}")
