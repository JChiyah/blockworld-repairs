
import time
from typing import List, Dict, Any, Tuple

import peft
import torch
import transformers
from transformers import (
	Idefics2ForConditionalGeneration, AutoModelForVision2Seq,
	BitsAndBytesConfig, Idefics2Processor, Trainer, TrainingArguments
)
from tqdm import tqdm

from . import conversation_processor
from .. import get_logger
from ..utils import debug_utils, common_utils
from ..data import Idefics2Prediction
from ..evaluation import get_training_evaluation_str
from ..configs import Config


ROLE_TOKENS = [('system', 'System'), ('user', 'User'), ('assistant', 'Assistant')]


logger = get_logger()


# 2 things:
# - load pretrained hf model (to eval or to fine-tune) - add adapters
# - load fine-tuned model (to eval or to further fine-tune) - make sure it already has the prev adapters
def load_model(model_name: str, model_load_mode: str, is_trainable: bool = True) -> peft.PeftModel:
	is_finetuned = common_utils.is_finetuned_model(model_name)

	logger.debug("Loading model...")

	# load model with the adapters
	model = _load_finetuned_model(model_name, model_load_mode, is_trainable) \
		if is_finetuned else _load_hf_model(model_name, model_load_mode)

	logger.info(f"Model loaded: {'fine-tuned' if is_finetuned else 'pre-trained'} '{model_name}' with {get_parameter_info_str(model)}")

	assert not is_trainable or get_parameter_info(model)['trainable'] > 0, "Model has no trainable parameters"

	return model


# def is_finetuned_model(model_name: str) -> bool:
# 	# todo: maybe this should check for a "ft" in the model name
# 	return common_utils.get_short_model_name(model_name) is None


def _load_finetuned_model(model_path: str, model_load: str = None, is_trainable: bool = True) -> peft.PeftModel:
	config = peft.PeftConfig.from_pretrained(model_path)

	model = _idefics2_from_pretrained(config.base_model_name_or_path, model_load=model_load)
	model = peft.PeftModel.from_pretrained(model, model_path, is_trainable=is_trainable)
	return model


def _load_hf_model(model_base: str, model_load: str = None, add_lora_adapters: bool = True) -> peft.PeftModel:
	# Three options for training, from the lowest precision training to the highest precision training:
	# - QLora
	# - Standard Lora
	# - Full fine-tuning
	# if is_finetuned_model(model_base):
	# 	# automatically turn off the lora adapters, probably already there if finetuned and saved
	# 	add_lora_adapters = False

	USE_QLORA = True if model_load == 'qlora' else False
	USE_LORA = True if model_load == 'lora' else False
	if model_load in ['qlora', 'lora']:
		if model_load == 'qlora':
			bnb_config = BitsAndBytesConfig(
				load_in_4bit=True,
				bnb_4bit_quant_type="nf4",
				bnb_4bit_compute_dtype=torch.float16
			)
		# model = Idefics2ForConditionalGeneration.from_pretrained(
		# 	model_base,
		# 	torch_dtype=torch.float16,
		# 	quantization_config=bnb_config if USE_QLORA else None,
		# )
		model = _idefics2_from_pretrained(model_base, model_load=model_load)
		# model = Idefics2ForConditionalGeneration.from_pretrained(
		# 	model_base,
		# 	torch_dtype=torch.float16,
		# 	# attn_implementation="flash_attention_2",
		# 	device_map={"": torch.cuda.current_device()} if torch.cuda.is_available() else None,
		# 	quantization_config=bnb_config if USE_QLORA else None,
		# )

		# if add_lora_adapters:   # only run in training
		lora_config = peft.LoraConfig(
			r=8,
			lora_alpha=8,
			lora_dropout=0.1,
			target_modules='.*(text_model|modality_projection|perceiver_resampler).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$',
			use_dora=False if USE_QLORA else True,
			init_lora_weights="gaussian"
		)

		# making the model a PeftModel, as huggingface trainer does not
		# have the same checks for a model with adapters when saving and loading
		model = peft.prepare_model_for_kbit_training(model)
		model = peft.get_peft_model(model, lora_config)
		# model.add_adapter(lora_config)
		# model.enable_adapters()

	# model = model.to(DEVICE)
	# logger.info(f"Loading model using {'QLora' if USE_QLORA else 'Lora'}")
	else:
		raise NotImplementedError("Model too large to load without QLoRa or LoRa")
		# logger.info(f"Loading full model")
		# model = Idefics2ForConditionalGeneration.from_pretrained(
		# 	model_base,
		# 	torch_dtype=torch.float16,
		# 	# _attn_implementation="flash_attention_2", # Only available on A100 or H100
		# ).to(DEVICE)

		model = AutoModelForVision2Seq.from_pretrained(
			model_base,
			device_map={"": torch.cuda.current_device()} if torch.cuda.is_available() else None
		)  # .to(DEFAULT_DEVICE)

	return model


def _idefics2_from_pretrained(model_name: str, *, model_load: str) -> transformers.PreTrainedModel:
	if model_load == 'qlora':
		bnb_config = BitsAndBytesConfig(
			load_in_4bit=True,
			bnb_4bit_quant_type="nf4",
			bnb_4bit_compute_dtype=torch.float16
		)

	return Idefics2ForConditionalGeneration.from_pretrained(
		model_name,
		torch_dtype=torch.float16,
		# attn_implementation="flash_attention_2",
		device_map={"": torch.cuda.current_device()} if torch.cuda.is_available() else None,
		quantization_config=bnb_config if model_load == 'qlora' else None,
	)


def get_processor(model_base: str = 'HuggingFaceM4/idefics2-8b'):
	return Idefics2Processor.from_pretrained(
		model_base,
		do_image_splitting=False,
	)


def get_parameter_info(model) -> dict:
	"""
	Returns the number of trainable parameters in the model.
	"""
	trainable_params = 0
	all_param = 0
	for _, param in model.named_parameters():
		all_param += param.numel()
		if param.requires_grad:
			trainable_params += param.numel()
	return {
		"all": all_param,
		"trainable": trainable_params,
		"trainable_percent": 100 * trainable_params / all_param
	}


def get_parameter_info_str(model) -> str:
	"""
	Returns a string with info about the number of trainable parameters in a model.
	"""
	info = get_parameter_info(model)
	return f"all params: {info['all']:,} || trainable params: {info['trainable']:,} || trainable%: {info['trainable_percent']:.2f}%"


class CustomDataCollator:

	def __init__(self, processor, logger_callable = None):
		self.processor = processor
		self.text_processor = processor
		self.logger_callable = logger_callable or logger.debug
		self._printed_examples = False

		self.image_token_id = self.processor.tokenizer.additional_special_tokens_ids[
			self.processor.tokenizer.additional_special_tokens.index("<image>")
		]

	def __call__(self, examples, mode: str = 'train'):
		texts = []
		images = []
		for example in examples:
			# messages = [
			# 	{
			# 		"role": "user",
			# 		"content": [
			# 			{"type": "text", "text": "Answer briefly."},
			# 			{"type": "image"},
			# 			{"type": "text", "text": question}
			# 		]
			# 	},
			# 	{
			# 		"role": "assistant",
			# 		"content": [
			# 			{"type": "text", "text": answer}
			# 		]
			# 	}
			# ]
			# only add the additional assistant message when training
			if mode == 'train':
				answer_msg = {
					"role": "assistant",
					"content": [
						{"type": "text", "text": example['answer']}
					]
				}
				conversation = example['messages'] + [answer_msg]
			else:
				# eval mode
				conversation = example['messages']

			# apply chat template, and if eval then add the generation prompt
			text = self.text_processor.apply_chat_template(conversation, add_generation_prompt=mode != 'train')
			texts.append(text.strip())
			images.append([example["image"]])

		batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
		# input(f"Text: {texts[0]}, is length the same? {len(texts[0])}, {len(batch['input_ids'][0])}")

		batch["labels"] = conversation_processor.get_masked_labels(
			batch["input_ids"], examples, self.processor, role_tokens=ROLE_TOKENS,
			token_ids_to_mask=[self.processor.tokenizer.pad_token_id, self.image_token_id])

		if not self._printed_examples and self.logger_callable:
			# Print additional info when first using the data collator
			self._printed_examples = True
			# self.logger_callable(f"input_ids: {batch['input_ids'][0]}")
			# self.logger_callable(
			# 	f"Decoded input ids: {debug_utils.decode_input_ids(batch['input_ids'][0], self.processor, handle_image_token=self.image_token_id)}")
			# self.logger_callable(f"Decoded labels: {debug_utils.decode_label_ids(batch['labels'][0], processor=self.processor)}")
			# self.logger_callable(f"labels: {batch['labels'][0]}")

			debug_utils.explore_processed_batch(
				self.processor, self.logger_callable, batch, examples, texts, token_ids={
					'image_token_id': self.image_token_id,
					'pad_token_id': self.processor.tokenizer.pad_token_id})

		return batch


def get_training_args(config: Config, is_eval: bool = False) -> TrainingArguments:
	"""Create training arguments based on config and mode."""
	base_args = {
		'output_dir': config.output_dir_model,
		'per_device_eval_batch_size': config.training.test_batch_size,
		'remove_unused_columns': False,
		'fp16': True,
		'report_to': "none",
	}

	if not is_eval:
		base_args.update({
			'num_train_epochs': config.training.epochs,
			'per_device_train_batch_size': config.training.train_batch_size,
			'gradient_accumulation_steps': 32//config.training.train_batch_size,
			'gradient_checkpointing_kwargs': {'use_reentrant': True},
			'warmup_steps': 50 if not config.debug else 0,
			'learning_rate': config.training.learning_rate,
			'weight_decay': 0.01,
			'logging_steps': 25 if not config.debug else 5,
			'save_strategy': "steps",
			'save_steps': 250 if not config.debug else 10,
			'save_total_limit': 1,
		})

	return TrainingArguments(**base_args)


class Idefics2PredictionTrainer(Trainer):
	def __init__(self, goal: str, log_interval: int = 30, generation_max_length: int = 64, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.goal = goal
		self.log_interval = log_interval  # seconds, set to 0 to disable logging
		self._last_log_time = 0
		self.generation_max_length = generation_max_length	# should use the Seq2SeqTrainingArguments instead

	@property
	def should_log_progress(self) -> bool:
		return self.log_interval > 0 and time.time() - self._last_log_time >= self.log_interval

	def log_progress(self, eval_str: str):
		self._last_log_time = time.time()
		logger.info(f"Evaluation {eval_str}")

	def prediction_step(self, model, inputs, prediction_loss_only=False):
		"""Override prediction_step to handle generation instead of prediction"""
		inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v
				 for k, v in inputs.items()}

		try:
			generated_ids = model.generate(
				**inputs,
				max_new_tokens=self.generation_max_length,
				# **self.args.generation_config
			)
			return None, None, generated_ids[:, inputs["input_ids"].size(1):]
		except Exception as e:
			logger.error(f"Generation failed for batch: {str(e)}")
			return None, None, None

	def get_predictions_with_progress(
			self,
			test_hf_dataset,
			test_dataset
		) -> Tuple[List[Idefics2Prediction], List[Dict[str, Any]]]:
		"""Get predictions with progress tracking"""
		dataloader = self.get_test_dataloader(test_hf_dataset)
		model = self.model.eval()

		predictions = []
		all_responses = []

		progress_bar = tqdm(
			enumerate(dataloader),
			total=len(dataloader),
			desc="Generating predictions"
		)

		for batch_idx, inputs in progress_bar:
			try:
				# Generate for current batch
				inputs = self._prepare_inputs(inputs)
				_, _, generated_ids = self.prediction_step(model, inputs)
				if generated_ids is None:
					continue

				# Decode generated text
				generated_texts = self.tokenizer.batch_decode(
					generated_ids,
					skip_special_tokens=True
				)

				# Get original examples for this batch
				batch_start = batch_idx * self.args.per_device_eval_batch_size
				batch_end = min(batch_start + len(generated_texts), len(test_hf_dataset))
				batch_examples = [test_hf_dataset[i] for i in range(batch_start, batch_end)]  # Get actual examples

				# Record responses for this batch
				batch_responses = [
					{
						'entry_idx': example['entry_idx'],
						'image_name': example['image_name'],
						'input_messages': example['messages'],
						'input_prompt': self.tokenizer.decode(
							inputs['input_ids'][i],
							skip_special_tokens=True
						),
						'input_labels': debug_utils.decode_label_ids(
							inputs['labels'][i],
							processor=self.tokenizer
						),
						'model_output_string': gen_text,
						'true_string': example['answer']
					}
					for i, (example, gen_text) in enumerate(zip(batch_examples, generated_texts))
				]

				# Create predictions for this batch
				batch_predictions = Idefics2Prediction.from_model_batch_output(
					[test_dataset.get_entry_by_idx(x['entry_idx']) for x in batch_examples],
					generated_texts,
					self.goal
				)

				# Extend our collections
				predictions.extend(batch_predictions)
				all_responses.extend(batch_responses)

				# Update progress with current metrics
				if len(predictions) > 0:
					eval_str = get_training_evaluation_str(
						predictions,
						total_entries=len(test_hf_dataset)
					)
					progress_bar.set_postfix_str(eval_str)

					# Periodically log progress
					if self.should_log_progress:
						self.log_progress(eval_str)

			except Exception as e:
				logger.error(f"Error processing batch {batch_idx}: {str(e)}")
				raise e

		return predictions, all_responses
