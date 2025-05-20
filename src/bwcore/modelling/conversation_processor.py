

from typing import List, Dict

import transformers


MASK_TOKEN_ID = -100


def get_masked_labels(batch_input_ids, batch_conversations: List[Dict[str, List[dict]]], tokenizer, role_tokens, token_ids_to_mask: List[str] = None):
	labels = batch_input_ids.clone()
	# there are two fine-tuning tutorial colabs:
	# https://colab.research.google.com/drive/1NtcTgRbSBKN7pYD3Vdx1j9m8pt3fhFDB?usp=sharing
	# uses the following:
	# labels[labels == processor.tokenizer.pad_token_id] = self.image_token_id

	# https://colab.research.google.com/drive/1rm3AGquGEYXfeeizE40bbDtcWh5S4Nlq?usp=sharing#scrollTo=yePZRNBTK0Ux
	# uses the following instead, but this crashes with a CUDA device-side assertion error:
	# labels[labels == processor.tokenizer.pad_token_id] = MASK_TOKEN_ID
	# labels[labels == self.image_token_id] = MASK_TOKEN_ID
	# see this: https://github.com/huggingface/transformers/pull/30898
	# after discussing with Alessandro, we keep the -100. We had to update to the latest transformers version from github, as it was new code
	# this takes the previous part as we usually pass pad_token_id and image_token_id
	if token_ids_to_mask:
		for token_id in token_ids_to_mask:
			# mask these tokens for now, usually the image tokens, etc
			labels[labels == token_id] = MASK_TOKEN_ID

	# now we get the overall structure of the conversations, and see if we need to mask anything else
	turn_indices = get_turn_start_end_indices_batch(labels, tokenizer, role_tokens)

	for i, conv in enumerate(batch_conversations):

		if 'messages' not in conv:      # we are using LLaVA
			# consume the system turn as it is not part of the conversation but the instructions
			role, start, end = turn_indices[i][0]
			labels[i][start:end] = MASK_TOKEN_ID
			turn_indices[i] = turn_indices[i][1:]
		else:
			# consume the last turn of the indices if assistant, as that is the target
			if turn_indices[i][-1][0] == 'assistant':
				turn_indices[i] = turn_indices[i][:-1]
			conv = conv['messages']

		assert len(conv) == len(turn_indices[i]), \
			f"Conversation and turn indices do not match! {len(conv)} vs {len(turn_indices[i])} - check the tokenizer" \
			f"\n  conv: {conv}\n  turns: {turn_indices[i]}\n{get_turn_start_end_indices(labels[i], tokenizer, role_tokens, debug=True)}"

		# now check which turn need to be masked
		for turn_i, turn in enumerate(conv):
			# assert turn['from'] == turn_indices[i][turn_i][0], f"Turn role does not match! {turn['from']} vs {turn_indices[i][turn_i][0]}"
			# logger.debug(f"Turn {turn_i}: {turn}")
			if turn['mask_during_train']:                   # mask this turn!
				role, start, end = turn_indices[i][turn_i]
				# print(f"Masking turn {turn_i} ({role}) from {start} to {end}")
				labels[i][start:end] = MASK_TOKEN_ID

	return labels


def get_turn_start_end_indices(input_ids, tokenizer, role_tokens: List[str], *, debug: bool = False) -> List[tuple]:
	# models that tokenize like: ASS ISTANT (idefics2)
	check_extra_tokens = 2
	try:
		if isinstance(tokenizer, transformers.models.llama.tokenization_llama.LlamaTokenizer):
			# these models tokenize like: A SS IST ANT (LLaVA)
			check_extra_tokens = 4

	except AttributeError:
		# idefics2 tokenizer
		pass

	turn_indices = []       # (role, start)
	end_indices = []
	for current_index in range(len(input_ids)):
		if input_ids[current_index] < 0:
			# ignore negative ids, as that is probably to do with masking
			continue

		# we want to check for any of the keywords in the turn_roles list
		# user and system are a single id, but assistant becomes ass istant, so we need to do +2
		try:
			text = tokenizer.decode(input_ids[current_index], skip_special_tokens=True)
		except IndexError as ex:
			print(f"Issue trying to decode the following input_id: {input_ids[current_index]} ({ex})\nInput_ids: {input_ids}")
			raise ex

		if 'Ass' in text:
			text = tokenizer.decode(
				input_ids[current_index:current_index+check_extra_tokens],
				skip_special_tokens=True)
		elif text.strip().startswith('A'):
			# assistant turns, decode a few more depending on model type
			text = tokenizer.decode(
				input_ids[current_index:current_index+check_extra_tokens],
				skip_special_tokens=True)

		if debug: print(f"decoded text: {text}")
		if text == 'US':
			if debug: print(f"before: {text}")
			# user turns, decode a few more if with LLaVA as it does US ER
			text = tokenizer.decode(
				input_ids[current_index:current_index+2],
				skip_special_tokens=True)

		for role, role_token in role_tokens:
			if debug: print(f"Checking {text} vs {role_token}")
			if text == role_token:
				if len(turn_indices) > 0:
					end_indices.append(current_index)
				turn_indices.append((role, current_index))
				if debug: print(f"turn found for {role} at {current_index} with '{text}' !")
				break

	end_indices.append(len(input_ids))

	# (role, start, end)
	turn_indices = [(turn[0], turn[1], e) for turn, e in zip(turn_indices, end_indices)]
	# print(f"turn_indices: {turn_indices}")

	# ensure it is correct
	# for r, s, e in turn_indices:
	# 	logger.debug(f"Turn decoded={r} `{processor.decode(input_ids[s:e], skip_special_tokens=True)}`")

	return turn_indices


def get_turn_start_end_indices_batch(all_input_ids, tokenizer, turn_roles: List[str]) -> List[List[tuple]]:
	all_indices = []
	# iterate over the all_input_ids numpy array
	for row in all_input_ids:
		all_indices.append(get_turn_start_end_indices(row, tokenizer, turn_roles))

	return all_indices


class MaskedLMDataCollator:

	def __init__(self, processor):
		self.processor = processor
		self.text_processor = processor
		if 'idefics2' in args.model_base:
			self.image_token_id = self.processor.tokenizer.additional_special_tokens_ids[
				self.processor.tokenizer.additional_special_tokens.index("<image>")
			]
		else:
			# tried it for LLava, but never worked
			# self.image_token_id = None
			raise NotImplementedError

	def __call__(self, examples, mode: str = 'train', logger_callable=None):
		texts = []
		images = []
		for example in examples:
			image = example["image"]
			# question = example["query"]["en"]
			# answer = random.choice(example["answers"])
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
				# eval mode probs
				conversation = example['messages']

			# apply chat template, and if eval then add the generation prompt
			text = self.text_processor.apply_chat_template(conversation, add_generation_prompt=mode == 'train')
			texts.append(text.strip())
			images.append(fewshot_images + [image])

		batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
		# input(f"Text: {texts[0]}, is length the same? {len(texts[0])}, {len(batch['input_ids'][0])}")

		batch["labels"] = get_masked_labels(
			batch["input_ids"], examples,
			token_ids_to_mask=[self.processor.tokenizer.pad_token_id, self.image_token_id])

		# if args.debug:
		if logger_callable:
			debug_utils.explore_processed_batch(
				self.processor, logger_callable, batch, examples, texts, token_ids={
					'image_token_id': self.image_token_id,
					'pad_token_id': self.processor.tokenizer.pad_token_id})
		# input("\nPress enter to continue")

		return batch


