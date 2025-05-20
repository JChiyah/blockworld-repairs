
from typing import Optional


MASK_TOKEN_ID = -100
MASK_TOKEN = '<mask>'


def decode_label_ids(label_ids, *, processor=None, tokenizer=None) -> str:
	if tokenizer is None:
		tokenizer = processor.tokenizer
	# add special token if not in the special ids just for this conversion
	if MASK_TOKEN not in tokenizer.additional_special_tokens:
		tokenizer.add_special_tokens({'additional_special_tokens': tokenizer.additional_special_tokens + [MASK_TOKEN]})

	label_clone = label_ids.clone()
	label_clone[label_clone == MASK_TOKEN_ID] = tokenizer.additional_special_tokens_ids[tokenizer.additional_special_tokens.index(MASK_TOKEN)]

	if processor:
		return processor.decode(label_clone, skip_special_tokens=False)
	else:
		return tokenizer.decode(label_clone, skip_special_tokens=False)


def decode_input_ids(input_ids, processor, *, handle_image_token: Optional[int] = None) -> str:
	# give in handle_image_token the index of the image token
	if handle_image_token:
		try:
			img_token_index = input_ids.tolist().index(handle_image_token)
		except ValueError:
			print(f"Image token {handle_image_token} not found in entry with input_ids: {input_ids}")
			decoded_input = 'ERROR! ' + processor.decode(input_ids)
		else:
			decoded_input = \
				f"{processor.decode(input_ids[:img_token_index])}<image>" \
				f"{processor.decode(input_ids[img_token_index + 1:])}"
	else:
		decoded_input = processor.decode(input_ids)

	return decoded_input


def explore_processed_batch(processor, logger_callable, processed_batch, entries, texts, token_ids: dict = None, index: int = 0):
	# take the first entry, print some info
	entry = entries[index]
	logger_callable(f"Exploring processed batch\n\tEntry info: {entry['entry_idx']}")

	logger_callable(f"\tBatch keys: {processed_batch.keys()}")
	logger_callable(f"\tBatch size: {len(processed_batch['input_ids'])}, {processed_batch['input_ids'].shape}")
	if token_ids:
		logger_callable(f"\tToken ids: {token_ids.items()}")

	logger_callable(f"\ttext: {texts[index]}")
	logger_callable(f"\tinput_ids: {processed_batch['input_ids'][index]}")
	logger_callable(f"\tinput_ids decoded: \n```{decode_input_ids(processed_batch['input_ids'][index], processor=processor)}```")
	logger_callable(f"\tattention_mask: {processed_batch['attention_mask'][index]}")
	logger_callable(f"\tlabels: {processed_batch['labels'][index]}")
	logger_callable(f"\tlabels decoded: \n```{decode_label_ids(processed_batch['labels'][index], processor=processor)}```")

	logger_callable(f"\tis labels same as input_ids? {processed_batch['labels'][index] == processed_batch['input_ids'][index]}")
