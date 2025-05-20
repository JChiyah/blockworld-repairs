# https://colab.research.google.com/drive/1rm3AGquGEYXfeeizE40bbDtcWh5S4Nlq?usp=sharing#scrollTo=yePZRNBTK0Ux
# Install:
# conda env create --file ../env/env3.yaml --prefix .envs/idefics2
# conda activate /users/fjc3/block-world-training/.envs/idefics2
# conda env config vars set HF_HOME='/users/fjc3/sharedscratch/hf_cache'
#
# pip install git+https://github.com/huggingface/transformers.git
# pip install accelerate datasets peft bitsandbytes Levenshtein pillow==10.3.0 colorlog pydantic
# pip install -U datasets # ensure above version 2.19 to avoid bug in load_dataset
#
# Run: sbatch train_scrum.sh source original_entries zero-shot

import json
from typing import List, Dict, Any

import torch
from tqdm import tqdm
import datasets
import bwcore.configs
from transformers import TrainingArguments, Trainer, Seq2SeqTrainingArguments
# from datasets import load_dataset


# add all helper packages, used or not used here
# sys.path.append('../../../block-world-research/')
# sys.path.append('../../../block-world-research/bw-correction-dialogues')
# sys.path.append('../../../block-world-research/bw-correction-annotations')

# import evaluate_annotations
# evaluate_annotations.evaluate_predictions(predictions)
# sys.path.append('../../')
# sys.path.append('../../rlhf')
import bwcore
logger = bwcore.get_logger(bwcore.log_utils.DEBUG)

from bwcore.utils import debug_utils, file_utils
import bwcore.modelling
import bwcore.modelling.idefics2 as idefics2
import bwcore.modelling.conversation_processor

# from bwcore import evaluation, debug_utils

# from bw_modelling import evaluation, debug_utils, data_modelling, idefics2
# import evaluate_bw_results
# import evaluate_annotations
# from rlhf_utils import parse_arguments, convert_to_hf_dataset, save_arguments, args_to_dict


DEVICE = "cuda:0"


# _generated_data_path = os.path.join(os.environ['UNITY_DATA_DIR'], '576p')
# _block_world_data_path = os.path.join(os.environ['UNITY_DATA_DIR'].replace(
# 	'generated_data', 'datasets'), 'BlockWorld-Random')


# role_tokens = [('system', 'System'), ('user', 'User'), ('assistant', 'Assistant')]


def check_pipeline_integrity(
		model: torch.nn.Module,
		dataset: datasets.Dataset,
		entry_sample: bwcore.data.BWEntry
	) -> None:
	# do a quick eval to check that the model works and the input/output formats are correct
	logger.debug('Checking pipeline integrity')
	logger.debug(f"Sample: {json.dumps(entry_sample, indent=4, default=str)}")

	text = text_processor.apply_chat_template(
		entry_sample['messages'], add_generation_prompt=True)
	inputs = processor(text=[text.strip()], images=[entry_sample['image']], return_tensors="pt", padding=True)

	if model:
		model.eval()
		inputs = inputs.to(model.device)

	labels = bwcore.modelling.conversation_processor.get_masked_labels(
		inputs['input_ids'], [entry_sample], processor,
		role_tokens=idefics2.ROLE_TOKENS)

	# bug in old versions of transformers does not like decoding images, so needs the following code
	# img_token = processor.tokenizer.additional_special_tokens_ids[processor.tokenizer.additional_special_tokens.index("<image>")]
	# image_index_start = inputs['input_ids'][0].tolist().index(img_token)
	# image_index_end = inputs['input_ids'][0][image_index_start+1:].tolist().index(img_token)
	# logger.info(f"Input prompt: \n```{processor.decode(inputs['input_ids'][0][:image_index_start], skip_special_tokens=True)}<image>{processor.decode(inputs['input_ids'][0][image_index_start+1:][image_index_end:], skip_special_tokens=True)}```")

	logger.debug(f"Input prompt: \n```{processor.decode(inputs['input_ids'][0], skip_special_tokens=False)}```")
	logger.debug(f"Labels: \n```{debug_utils.decode_label_ids(labels[0], processor=processor)}```")

	if model is not None:
		generated_ids = model.generate(**inputs, max_new_tokens=64)
		generated_texts = processor.batch_decode(generated_ids[:, inputs["input_ids"].size(1):], skip_special_tokens=True)
		logger.debug(f"Generated text: \n```{generated_texts[0]}```")
		predictions = bwcore.data.Idefics2Prediction.from_model_batch_output(
			[dataset.get_entry_by_idx(x['entry_idx']) for x in [entry_sample]], generated_texts, config.goal)
		logger.debug("Evaluation:")
		bwcore.evaluation.print_evaluation_table(bwcore.evaluation.evaluate_predictions(predictions), remove_columns=['seconds_per_entry_std'])
	else:
		logger.debug("Model not loaded, skipping forward pass checks")



def convert_to_hf_dataset(dataset: bwcore.data.BWEntrySet, split: str, config: bwcore.configs.Config) -> datasets.Dataset:
	# generated_data, _, _ = bwcore.data.get_data_paths()

	hf_dataset = dataset.to_hf_dataset(
		config.data, split,
		max_num_entries=getattr(config.debug, f"{split}_num_entries") if config.debug else None)
	return hf_dataset


def main(config: bwcore.configs.Config):
	if config.do_train:
		train_dataset = bwcore.data.get_bw_dataset(config.data.train_dataset)
		train_hf_dataset = convert_to_hf_dataset(train_dataset, 'train', config)

	else:
		train_dataset, train_hf_dataset = None, None

	test_dataset = bwcore.data.get_bw_dataset(config.data.test_dataset)
	test_hf_dataset = convert_to_hf_dataset(test_dataset, 'test', config)

	# Check pipeline integrity (input/output) before and after loading model
	check_pipeline_integrity(None, test_dataset, test_hf_dataset[-1])
	model = idefics2.load_model(config.base_model, config.load_model)
	check_pipeline_integrity(model, test_dataset, test_hf_dataset[-1])
	# The last integrity check does a forward pass to check everything is working

	if config.do_eval:
		# Do an evaluation before training
		evaluate_model(model, processor, test_hf_dataset, test_dataset, config, epoch=0)

	if config.do_train:
		# Train then evaluate after
		train_model(model, processor, train_hf_dataset, train_dataset, config)
		# temporary: we assume that it trained for all the epochs during .train()
		evaluate_model(model, processor, test_hf_dataset, test_dataset, config, epoch=config.training.epochs)


def train_model(
		model: torch.nn.Module,
		processor: Any,
		train_hf_dataset: datasets.Dataset,
		train_dataset: bwcore.data.BWEntrySet,
		config: bwcore.configs.Config
	) -> None:
	"""Train model and save checkpoints"""
	logger.info("** Starting Model Training **")

	# Set up model and training args
	model.config.use_cache = False
	training_args = idefics2.get_training_args(config, is_eval=False)

	trainer = Trainer(
		model=model,
		args=training_args,
		data_collator=idefics2.CustomDataCollator(processor),
		train_dataset=train_hf_dataset,
	)

	try:
		model.train()
		trainer.train()
		file_utils.save_model_checkpoint(trainer, config.output_dir_model)

	except KeyboardInterrupt:
		logger.warning("Training interrupted by user. Attempting to save checkpoint...")
		file_utils.save_model_checkpoint(trainer, config.output_dir_model)
		raise

	except Exception as e:
		logger.error(f"Training failed: {str(e)}")
		raise

def evaluate_model(
		model: torch.nn.Module,
		processor: Any,
		test_hf_dataset: datasets.Dataset,
		test_dataset: bwcore.data.BWEntrySet,
		config: bwcore.configs.Config,
		epoch: int = 0
	) -> None:
	"""Evaluate model on test dataset and save results"""
	logger.info("** Starting Model Evaluation **")

	try:
		# Clean GPU memory and set model to eval
		model.eval()
		torch.cuda.empty_cache()

		# Configure trainer
		training_args = idefics2.get_training_args(config, is_eval=True)

		trainer = idefics2.Idefics2PredictionTrainer(
			goal=config.goal,
			generation_max_length=64,
			model=model,
			args=training_args,
			data_collator=idefics2.CustomDataCollator(processor),
			tokenizer=processor  # Added for decoding
		)

		# Get predictions
		predictions, all_responses = trainer.get_predictions_with_progress(
			test_hf_dataset,
			test_dataset
		)

		if not predictions:
			raise RuntimeError("No predictions generated")

		# Log and save results
		_save_evaluation_results(
			predictions=predictions,
			responses=all_responses,
			config=config,
			test_dataset=test_dataset,
			epoch=epoch
		)

	except Exception as e:
		logger.error(f"Evaluation failed: {str(e)}")
		raise

def _save_evaluation_results(
		predictions: List[bwcore.data.Idefics2Prediction],
		responses: List[Dict[str, Any]],
		config: bwcore.configs.Config,
		test_dataset: bwcore.data.BWEntrySet,
		epoch: int
	) -> None:
	"""Save evaluation results and print metrics."""
	logger.info("Final evaluation results")
	bwcore.evaluation.print_evaluation_table(
		bwcore.evaluation.evaluate_predictions(predictions),
		remove_columns=['seconds_per_entry_std']
	)

	output_file = file_utils.save_model_outputs(config, responses, epoch)
	bwcore.evaluation.eval_output_file(output_file, test_dataset)


if __name__ == '__main__':
	# Get CLI arguments into the Config object to use throughout the script
	config: bwcore.configs.Config = bwcore.configs.parse_configs(
		required=['goal', 'task'],
		defaults={
			'base_model': 'HuggingFaceM4/idefics2-8b',
			'load_model': 'qlora',
			'turn_masking': 'none',
			'prompt_task_instruction': 'system',
			'train_learning_rate': 1e-4
		}
	)

	logger.info(f"{'=' * 10} Starting idefics2_main.py {'=' * 10}")

	assert 'idefics2' in config.base_model

	processor = idefics2.get_processor(config.base_model)
	# Set processor and chat template
	text_processor = processor
	text_processor.chat_template = text_processor.chat_template.split(
		"{% if add_generation_prompt %}")[0] + "{% if add_generation_prompt %}{{ 'Assistant: " + ('block' if config.goal == 'source' else 'location') + " bounding box' }}{% endif %}"

	main(config)

	logger.info(f"{'=' * 10} Finished idefics2_main.py {'=' * 10}")
