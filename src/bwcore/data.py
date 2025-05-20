import os
import abc
import csv
import sys
import json
import enum
import random
import collections
from typing import *

import datasets
import pydantic
import numpy as np
from PIL import Image
# from shapely.geometry import box
import torch
import torchvision

# from src import log_utils
# logger = log_utils.get_logger(log_utils.INFO)
from . import get_logger
logger = get_logger()

# sys.path.append('../../block-world-training')
from .utils.load_utils import load_entries, calculate_bw_block_distance, block_length, _calculate_midpoint
from .utils import unity_utils
from .utils.removeserver import get_dialogue_entries, format_dialogue_for_annotations, entry_selection_config

# from server import get_dialogue_entries, format_dialogue_for_annotations, entry_selection_config

# sys.path.append('../../block-world-training')
from . import configs, parsing


IMAGE_SIZES = {
	'576p': (1024, 576),
}
DEFAULT_IMAGE_SIZE = IMAGE_SIZES['576p']
PLACEHOLDER_INVALID_VALUE = -999

# PARTICIPANTS = ['J-105', 'W-255'] Q092
PARTICIPANTS = ['K-301', 'H-356', 'N-979', 'P-200', 'Q-848', 'D-497', 'B-263', 'R-147', 'M-823', 'B-141', 'C-011', 'R-347', 'X-534', 'S-443', 'L-642', 'V-670', 'M-634', 'H-147', 'J-997', 'J-105', 'W-255']
PARTICIPANT_OUTLIERS = ['H-356']


def num_with_perc(num: int, total: int) -> str:
	if total == 0:
		return f"0 (0%)"
	return f"{num} ({num/total*100:.1f}%)"


def read_jsonl(filename: str) -> list:
	entries = []
	with open(filename, 'r') as file:
		for line in file:
			# Parse the JSON object in each line
			entries.append(json.loads(line))

	return entries


# enum for entry type
class BWEntryType(enum.Enum):
	INSTRUCTION_OG = 'instruction_og'
	INSTRUCTION = 'instruction'
	CORRECTION_SOURCE = 'correction_source'
	CORRECTION_TARGET = 'correction_target'

	@property
	def has_source(self) -> bool:
		return self in [BWEntryType.INSTRUCTION_OG, BWEntryType.INSTRUCTION, BWEntryType.CORRECTION_SOURCE]

	@property
	def has_target(self) -> bool:
		return self in [BWEntryType.INSTRUCTION_OG, BWEntryType.INSTRUCTION, BWEntryType.CORRECTION_TARGET]

	@property
	def has_candidate(self) -> bool:
		return self in [BWEntryType.CORRECTION_SOURCE, BWEntryType.CORRECTION_TARGET]

	@property
	def is_correction(self) -> bool:
		return self in [BWEntryType.CORRECTION_SOURCE, BWEntryType.CORRECTION_TARGET]

	@property
	def is_instruction(self) -> bool:
		return self in [BWEntryType.INSTRUCTION_OG, BWEntryType.INSTRUCTION]

	def __str__(self):
		return self.value

	@staticmethod
	def custom_order(value):
		order = ['instruction_og', 'instruction', 'correction', 'correction_source', 'correction_target']
		return order.index(str(value))


class MessageContent(pydantic.BaseModel):
	type: str       # image or text
	value: str      # image, text, etc

	@pydantic.field_validator('type', mode='before')
	def validate_type(cls, value):
		if value not in ['image', 'text']:
			raise ValueError(f"type must be either 'image' or 'text', got {value}")
		return value

	def __str__(self):
		if self.type == 'image':
			return f"<image: {self.value}>"
		else:
			return self.value


class Message(pydantic.BaseModel):
	role: str       # user/assistant
	content: List[MessageContent]

	@pydantic.field_validator('role', mode='before')
	def validate_role(cls, value):
		if value == 'agent':            # quick fix
			value = 'assistant'
		if value not in ['user', 'assistant']:
			raise ValueError(f"role must be either 'user' or 'assistant', got {value}")
		return value

	def __str__(self):
		return f"{self.role}: {' '.join([str(x) for x in self.content])}"


class BWEntry(pydantic.BaseModel):
	entry_idx: str
	type: BWEntryType
	messages: List[Message] = []
	blocks_coords: List[List[float]]
	unity: unity_utils.UnityData
	image: str
	raw_data: dict

	# sort things below
	# dialogue_idx: str
	# user_utterances: List[str]
	# origin_idx: str
	# origin_img: str
	# img_final: str
	# unity_data: dict
	# # user data collection
	# mturk_scenario_id: str
	# previous_dialogues: List[str]       # number of prev dialogues that the user has had
	# user_idx: str
	# is_training: bool               # todo: maybe fix?
	# low_quality: bool               # todo: maybe fix?
	# elapsed_seconds: float
	# survey_answers: Dict[str, str]
	# events: List[Dict[str, str]]  # You may need to specify the inner dictionary structure

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.__trim_raw_data()

	def __trim_raw_data(self):
		# helps to avoid duplication by removing keys that are already defined in BWEntry
		keys_to_remove = [key for key in self.raw_data.keys() if hasattr(self, key)]
		for key in keys_to_remove:
			del self.raw_data[key]

	@pydantic.field_validator('blocks_coords', mode='after')
	def validate_blocks_coords(cls, value):
		# ensure 10 blocks and 3 coords per block (xyz)
		if len(value) != 10 or any([len(x) != 3 for x in value]):
			raise ValueError(f"Invalid blocks_coords (they should be 10x3): {value}")
		return value

	@pydantic.field_validator('image', mode='after')
	def validate_image(cls, value):
		if not value.endswith('.png'):
			raise ValueError(f"Invalid image path: {value}")
		return value

	@property
	def dpo_type(self):
		if self.type == BWEntryType.INSTRUCTION_OG: # or (entry.type.is_instruction and eval_result.predictions and eval_result.source.accuracy > 0):
			return DPOEntryType.INSTRUCTION_CLEAN
		elif self.type.is_correction:
			return DPOEntryType.INSTRUCTION_WITH_TPR
		elif self.type == BWEntryType.INSTRUCTION:
			# instructions here
			return DPOEntryType.INSTRUCTION_TO_REPAIR
		else:
			return DPOEntryType.NULL

	# @property
	# def image_name(self) -> str:
	# 	input(self.raw_data['origin_img'])
	# 	return self.raw_data['origin_img']

	@property
	def image_with_resolution(self) -> str:
		return self.image.replace('.png', '-576p.png')

	@property
	def has_source(self) -> bool:
		return self.type.has_source

	@property
	def has_target(self) -> bool:
		return self.type.has_target

	@property
	def has_candidate(self) -> bool:
		return self.type.has_candidate

	@property
	def is_correction(self) -> bool:
		return self.type.is_correction

	@property
	def is_instruction(self) -> bool:
		return self.type.is_instruction

	@property
	def true_source_index(self) -> int:
		return self.unity.true_source_index

	@property
	def true_source_coords(self) -> List[float]:
		return self.blocks_coords[self.true_source_index]

	@property
	def true_target_coords(self) -> List[float]:
		return self.unity.true_target_coords

	@property
	def is_training_scenario(self) -> bool:
		return self.raw_data['is_training']

	@property # state_index, world_name
	def world_name(self) -> str:
		return '_'.join(self.image.split('_')[:-1])

	@property
	def state_order(self) -> str:
		return self.image.split('_')[-1].replace('.png', '')

	@property
	def state_name(self) -> str:
		return self.image.replace('.png', '')

	@property
	def is_original(self) -> bool:
		return isinstance(self, OriginalBWEntry)

	@property
	def is_collected(self) -> bool:
		return isinstance(self, CollectedBWEntry)

	def messages_as_dict(self, *, max_images: int = 1, user_instruction: str = None, system_instruction: str = None, turn_masking: str = None, repair_form: Literal['word', 'sentence', None] = None) -> List[dict]:
		if turn_masking == 'none': turn_masking = None          # assistant, all, None
		assert turn_masking in [None, 'assistant', 'all']
		final_messages = []
		for message in self.messages:
			content = []
			# content = [{
		    #        'type': content.type,
		    #        'text': content.value
	        #     } if content.type == 'text' else {
			# 		'type': content.type,
			# 	} for content in message.content]
			for _content in message.content:
				if _content.type != 'text':
					continue
				_tmp = {
					'type': _content.type,
					'text': _content.value
				}
				if repair_form:
					repair_new_txt = (parsing.REPAIRS_SOURCE[0] if self.has_source else parsing.REPAIRS_TARGET[0]) if repair_form == 'sentence' else 'repair'
					# we want to unify the input so the repair sentence is always the same
					for repair_str in parsing.REPAIRS_SOURCE[1:] if self.has_source else parsing.REPAIRS_TARGET[1:]:
						# input(f"is {repair_str} in {_tmp['text']}? {repair_str in _tmp['text']}")
						if repair_str in _tmp['text']:
							_tmp['text'] = _tmp['text'].replace(repair_str, repair_new_txt)
							break

				if _tmp['type'] == 'text' and '<img_current>' in _tmp['text']:
					# TODO: FIX THIS ERROR, THIS WOULD CAUSE ISSUES AS DEFAULTS TO HAS_SOURCE EVEN ON INSTRUCTIONS!!
					# we have a candidate text!
					if self.has_source:
						cand_text = f"block bounding box {unity_utils.bbox_as_str(normalise_bbox(self.cand_source_bbox))}"
					elif self.has_target:
						cand_text = f"location bounding box {unity_utils.bbox_as_str(normalise_bbox(self.cand_target_bbox))}"
					else:
						raise ValueError

					content.append({
						'type': 'text',
						'text': cand_text + '\n'
					})

					_tmp['text'] = _tmp['text'].replace('<img_current>', '')
				content.append(_tmp)

			if len(final_messages) == 0:
				# if len(content) == 1 and content[0]['type'] == 'text':
					# we are missing the image, fix!
					# content.append({'type': 'image'})
				if user_instruction:
					content.insert(0, {'type': 'text', 'text': user_instruction})
				content.insert(0, {'type': 'image'})

			# check if any text in content has "<img_current>"
			# for cnt in content:
			# 	if cnt['type'] == 'text' and '<img_current>' in cnt['text']:
			# 		cnt['text'] = cnt['text'].replace('<img_current>', '')

			mask = False
			if turn_masking:
				if turn_masking == 'assistant' and message.role == 'assistant':
					mask = True
				elif turn_masking == 'all':
					mask = True

			final_messages.append({
				'role': message.role,
				'content': content,
				'mask_during_train': mask
			})

		# remove any images above max_images
		for message in reversed(final_messages):
			for cnt in message['content']:
				if cnt['type'] == 'image':
					if max_images > 0:
						max_images -= 1
						continue
					else:
						# print(f"before: {message['content']}")
						message['content'].remove(cnt)
						# print(f"after: {message['content']}")

		# if self.is_instruction:
		# input(json.dumps(final_messages, indent=2))
		# print('---')

		if max_images > 0:
			# should never get here!
			print(final_messages)
			raise ValueError(f"Number of images is not 1 (this will crash later)")

		if system_instruction:
			final_messages.insert(0, {
				'role': 'system',
				'mask_during_train': turn_masking == 'all',
				'content': [{
					'type': 'text',
					'text': system_instruction
				}]
			})

		# return [{
		# 	'role': message.role,
		# 	'content': [{
		# 		'type': content.type,
		# 		'text': content.value
		# 	} if content.type == 'text' else {
		# 		'type': content.type,
		# 	} for content in message.content]
		# } for message in self.messages]
		return final_messages

	@property
	def dialogue_str(self):
		return ' | '.join([str(x) for x in self.messages])

# true_xy = unity_utils.convert_coords_to_xy(entry['events'][-1]['true_source_coords']),
# true_coords = entry['events'][-1]['true_source_coords'],
# true_index = entry['events'][-1]['true_source_index'],
# cand_xy = entry['events'][-2].get('cand_source_xy'),
# cand_coords = entry['events'][-2].get('cand_source_coords'),


class OriginalBWEntry(BWEntry):

	@classmethod
	def from_utterance(
		cls, entry_idx: str, utterance: str, unity_dict: dict,
		blocks_coords: List[List[float]], image: str) -> 'OriginalBWEntry':
		return cls(
			entry_idx=entry_idx,
			type=BWEntryType('instruction_og'),
			messages=cls.messages_from_utterance(utterance, image),
			unity=unity_utils.UnityData.from_dict(unity_dict),
			blocks_coords=blocks_coords,
			image=image,
			raw_data={'is_training': False}
		)

	@staticmethod
	def messages_from_utterance(utterance: str, image: str) -> List[Message]:
		return [Message(role='user', content=[
			MessageContent(type='text', value=utterance),
			MessageContent(type='image', value=image)])]


class CollectedBWEntry(BWEntry):

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		if self.has_source:
			assert self.unity.true_source_index == self.raw_data['events'][-1]['true_source_index']
			assert self.unity.true_source_coords == self.raw_data['events'][-1]['true_source_coords']
		if self.has_target:
			assert self.unity.true_target_coords == self.raw_data['events'][-1]['true_target_coords']

	@classmethod
	def from_split_entry(cls, entry_dict: dict) -> 'CollectedBWEntry':
		# print(entry_dict)
		# input(entry_dict.get('entry_type'))
		return cls(
			entry_idx=entry_dict['entry_idx'],
			type=entry_dict.get('entry_type') or BWEntryType(entry_dict['type']),
			messages=cls.messages_from_events(entry_dict['events']),
			unity=unity_utils.UnityData.from_dict(entry_dict['unity']),
			blocks_coords=json.loads(entry_dict['events'][-1]['block_coords']),
			image=entry_dict['origin_img'],
			raw_data=entry_dict
		)

	@staticmethod
	def messages_from_events(events: List[dict]) -> List[Message]:
		messages_list = []
		for event in events:
			if event['event_type'] == 'agent_message':
				messages_list.append(Message(role='assistant', content=[
					MessageContent(type='text', value=event['utterance']),
					MessageContent(type='image', value=event['img_current'])]))
			elif event['event_type'] == 'user_message':
				messages_list.append(Message(role='user', content=[
					MessageContent(type='text', value=event['utterance'])]))

		# input(events[0])
		return messages_list

	@property
	def cand_source_coords(self) -> List[float]:
		# will raise error if no candidate
		return self.blocks_coords[self.raw_data['events'][-2]['cand_source_index']]

	@property
	def cand_target_coords(self) -> List[float]:
		# will raise error if no candidate
		return self.raw_data['events'][-2]['cand_target_coords']

	@property
	def cand_source_xy(self):
		# return self.unity.convert_xy_to_coords(self.raw_data['events'][-2]['cand_source_xy'])
		return self.raw_data['events'][-2]['cand_source_xy']

	@property
	def cand_source_index(self) -> int:
		return self.raw_data['events'][-2]['cand_source_index']

	@property
	def cand_target_xy(self):
		return self.raw_data['events'][-2]['cand_target_xy']

	@property
	def cand_source_bbox(self) -> List[float]:
		# if 'cand_source_index' not in self.raw_data['events'][-2]:
		# 	input(self.raw_data['events'][-2])
		return self.unity.blocks_bboxes[self.raw_data['events'][-2]['cand_source_index']]

	@property
	def cand_target_bbox(self) -> List[float]:
		temporary_bbox = self.raw_data['events'][-2]['cand_target_xy']
		temporary_bbox = [temporary_bbox[0], temporary_bbox[1], temporary_bbox[0] + 10, temporary_bbox[1] + 10]
		return temporary_bbox


# todo: handle this: right now, we have 3 images: original, bw_collection and unity (also used for annotations)

class PredictionItem(pydantic.BaseModel):
	entry_idx: str
	entry: BWEntry
	source: Optional[Union['PredictedSource']] = None
	target: Optional[Union['PredictedTarget']] = None

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		# self._entry = get_entry_by_idx(self.entry_idx)  # if self._entry is None else self._entry
		# input(self._entry.keys())
		assert self.entry.entry_idx == self.entry_idx           # todo: change to validator?
		assert self.has_source or self.has_target, f"Invalid entry {self.entry_idx}: {self.type} but no source or target"
		assert self.source or self.target, f"Invalid entry {self.entry_idx}: {self.type} and both predictions are None"

	@classmethod
	def from_baseline(cls, entry: BWEntry, baseline: 'BaselineBase') -> 'PredictionItem':
		try:
			return cls(
				entry_idx=entry.entry_idx,
				entry=entry,
				source=PredictedSource.from_baseline(entry, baseline) if entry.has_source else None,
				target=PredictedTarget.from_baseline(entry, baseline) if entry.has_target else None,
			)
		except ValueError as e:
			if 'baseline does not apply' in str(e):
				pass            # ignore this msg
			else:
				logger.exception(e)
			return None

	@classmethod
	def all_from_baseline(cls, baseline: 'BaselineBase', all_entries: 'BWEntrySet', filter_func: Callable = None) -> List['PredictionItem']:
		all_baselines = [cls.from_baseline(x, baseline) for x in all_entries]
		if filter_func:
			all_baselines = [x for x in all_baselines if x and filter_func(x)]

		return all_baselines

	@property
	def type(self) -> BWEntryType:
		return self.entry.type

	@property
	def has_source(self) -> bool:
		return self.type.has_source and self.source

	@property
	def has_target(self) -> bool:
		return self.type.has_target and self.target

	@property
	def has_candidate(self) -> bool:
		return self.type.has_candidate

	@property
	def is_correction(self) -> bool:
		return self.type.is_correction

	@property
	def is_instruction(self) -> bool:
		return self.type.is_instruction

	@property
	def is_annotation(self) -> bool:
		return isinstance(self, AnnotatedPrediction)

	@property
	def is_training_scenario(self) -> bool:
		return self.entry.is_training_scenario


class AnnotatedPrediction(PredictionItem):
	participant_idx: str
	elapsed_seconds: float
	image: str

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.participant_idx = self.participant_idx.replace('-', '')
		assert len(self.participant_idx) == 4, f"Invalid participant_idx {self.participant_idx} for {self.entry_idx}"
		# self._entry = get_entry_by_idx(self.entry_idx)
		# assert self._entry['entry_idx'] == self.entry_idx           # todo: change to validator?
		# self.__prepare_annotations(kwargs['annotations'])
		if self.has_candidate:
			assert (self.source and self.source.cand_coords) \
			       or (self.target and self.target.cand_coords), f"Missing candidate coords for {self.entry_idx}"

	@classmethod
	def from_annotation_collection(cls, entry: BWEntry, **kwargs) -> 'AnnotatedPrediction':
		# entry = get_entry_by_idx(kwargs['entry_idx'])
		# entry_type = BWEntryType(kwargs['entry_type'])
		if 'elapsed_seconds' in kwargs: del kwargs['elapsed_seconds']       # keep calculated val only
		if 'image' in kwargs: del kwargs['image']       # keep calculated val only

		return cls(
			**kwargs,
			entry=entry,
			participant_idx=kwargs['annotations'].get('participant_idx') or kwargs['user_idx'],
			elapsed_seconds=kwargs['annotations']['timestamp_end'] - kwargs['annotations']['timestamp_start'],
			image=kwargs['annotations'].get('image') or kwargs['dialogue_with_annotations'][-2]['image'],
			source=PredictedSource.from_annotation(data=kwargs['annotations'], entry=entry) if entry.has_source else None,
			target=PredictedTarget.from_annotation(data=kwargs['annotations'], entry=entry) if entry.has_target else None,
		)


class PredictedLocation(pydantic.BaseModel):
	is_valid: bool = True
	_type: str
	pred_coords: List[float]    # xyz in BW
	pred_bbox: Optional[List[float]] = None
	pred_repair: bool = False

	true_coords: List[float]    # xyz in BW
	true_bbox: Optional[List[float]] = None
	true_repair: bool = False

	cand_coords: Optional[List[float]] = None       # xyz in BW

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		assert self._type in ['source', 'target'], f"Invalid type {self._type} for LocationPrediction"
		self._block_distance = calculate_bw_block_distance(self.pred_coords, self.true_coords)

	@property
	def block_distance(self) -> float:
		"""
		:return: Distance from pred to true coords in Block measurements
		"""
		return self._block_distance

	@property
	def iou_score(self) -> float:
		"""
		:return: Intersection over Union score between pred_bbox and true_bbox
		"""
		if not self.pred_bbox or not self.true_bbox:
			return np.NaN

		# Convert bounding boxes to torch tensors
		# Expected format is [x_min, y_min, x_max, y_max]
		pred_box_tensor = torch.tensor([self.pred_bbox], dtype=torch.float)
		true_box_tensor = torch.tensor([self.true_bbox], dtype=torch.float)

		# Calculate IoU using torchvision.ops.box_iou
		# box_iou returns a tensor of shape (N, M) where N is number of pred_boxes and M is number of true_boxes
		iou_matrix = torchvision.ops.box_iou(pred_box_tensor, true_box_tensor)

		# Since we have one prediction and one true box, the result is a 1x1 tensor.
		iou = iou_matrix[0, 0].item()

		return iou


class PredictedSource(PredictedLocation):
	_type: str = 'source'
	pred_xy: Optional[List[float]] = None     # xy in image pixels
	pred_index: int

	true_xy: Optional[List[float]] = None     # xy in image pixels
	true_index: int

	cand_xy: Optional[List[float]] = None     # xy in image pixels
	cand_index: Optional[int] = None

	@classmethod
	def from_annotation(cls, data: dict, entry: BWEntry) -> 'PredictedSource':
		if 'annotation_source_xy' not in data:
			raise ValueError(f"Source annotation missing for {entry.entry_idx} - (probably from pressing Continue twice)")

		if data['annotation_source_xy'][0] == PLACEHOLDER_INVALID_VALUE:
			raise ValueError(f"Source annotation invalid for {entry.entry_idx}: {data['annotation_source_xy']}")

		return cls.from_xy_image(data['annotation_source_xy'], entry)

	@classmethod
	def from_xy_image(cls, pred_xy: List[float], entry: BWEntry) -> 'PredictedSource':
		pred_coords = unity_utils.convert_xy_to_coords(pred_xy, DEFAULT_IMAGE_SIZE)
		try:
			pred_index = unity_utils.get_closest_block_index(pred_coords, entry.unity.blocks_bboxes)
			# print(f"\t{pred_index=} vs {entry.unity.true_source_index=}")
			# if using selected block coords for the distance calculations
			pred_coords = entry.blocks_coords[pred_index]

		except ValueError as e:
			logger.exception(e)
			# show_annotations_in_image(
			# 	entry['annotations'].get('image') or entry['image'], entry)
			exit()

		return cls._default_class(entry, pred_xy=pred_xy, pred_coords=pred_coords, pred_index=pred_index)
		# return cls(
		# 	pred_xy=pred_xy,
		# 	pred_coords=pred_coords,
		# 	pred_index=pred_index,
		# 	true_xy=entry.unity.true_source_xy,
		# 	true_coords=entry.true_source_coords,
		# 	true_index=entry.true_source_index,
		# 	cand_xy=entry.cand_source_xy if entry.has_candidate else None,
		# 	cand_coords=entry.cand_source_coords if entry.has_candidate else None,
		# )

	@classmethod
	def from_bbox(cls, pred_bbox: List[float], contains_repair: bool, entry: BWEntry) -> 'PredictedSource':
		pred_xy = _calculate_midpoint(pred_bbox)
		pred_coords = unity_utils.convert_xy_to_coords(pred_xy, DEFAULT_IMAGE_SIZE)
		try:
			pred_index = unity_utils.get_closest_block_index(pred_coords, entry.unity.blocks_bboxes)
			# print(f"\t{pred_index=} vs {entry.unity.true_source_index=}")
			# if using selected block coords for the distance calculations
			pred_coords = entry.blocks_coords[pred_index]

		except ValueError as e:
			logger.exception(e)
			# show_annotations_in_image(
			# 	entry['annotations'].get('image') or entry['image'], entry)
			exit()

		return cls._default_class(entry, pred_bbox=pred_bbox, pred_xy=pred_xy, pred_coords=pred_coords, pred_index=pred_index, pred_repair=contains_repair)

	@classmethod
	def from_baseline(cls, entry: BWEntry, baseline: 'BaselineBase') -> 'PredictedSource':
		# pred = baseline_func({**entry['events'][-1], **{'blocks': entry.block_coords}, **entry['unity']})
		pred = baseline(entry)

		if pred['coords'] == PLACEHOLDER_INVALID_VALUE:
			# return None
			raise ValueError(f"Ignoring this entry as the baseline does not apply")

		# print(baseline.name)
		return cls._default_class(entry, pred_coords=pred['coords'], pred_index=pred['index'], pred_repair=False)
		# return cls(
		# 	pred_coords=pred['coords'],
		# 	pred_index=pred['index'] if 'index' in pred else 10,
		# 	true_coords=entry['events'][-1]['true_source_coords'],
		# 	true_index=entry['events'][-1]['true_source_index'],
		# 	cand_xy=entry['events'][-2].get('cand_source_xy'),
		# 	cand_coords=entry['events'][-2].get('cand_source_coords'),
		# )

	@classmethod
	def _default_class(cls, entry: BWEntry, **kwargs) -> 'PredictedSource':
		return cls(
			**kwargs,
			true_xy=entry.unity.true_source_xy,
			true_coords=entry.true_source_coords,
			true_index=entry.true_source_index,
			true_bbox=entry.unity.true_source_bbox,
			true_repair=entry.dpo_type == DPOEntryType.INSTRUCTION_TO_REPAIR,
			cand_xy=entry.cand_source_xy if entry.has_candidate else None,
			cand_coords=entry.cand_source_coords if entry.has_candidate else None,
			cand_index=entry.cand_source_index if entry.has_candidate else None,
		)

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self._accuracy = 1 if self.pred_index == self.true_index else 0

	@property
	def accuracy(self) -> int:
		"""
		:return: 1 if pred_index == true_index, 0 otherwise
		"""
		return self._accuracy


class PredictedTarget(PredictedLocation):
	_type: str = 'target'
	pred_xy: Optional[List[float]] = None     # xy in image pixels
	pred_bbox: Optional[List[float]] = None

	# pred_coords: List[float]    # xyz in BW
	true_xy: Optional[List[float]] = None     # xy in image pixels
	# true_coords: List[float]    # xyz in BW
	true_bbox: Optional[List[float]] = None

	cand_xy: Optional[List[float]] = None     # xy in image pixels
	# cand_coords: Optional[List[float]] = None      # xyz in BW

	@classmethod
	def from_annotation(cls, data: dict, entry: BWEntry) -> 'PredictedTarget':
		if 'annotation_target_xy' not in data:
			raise ValueError(f"Target annotation missing for {entry.entry_idx} - (probably from pressing Continue twice)")

		if data['annotation_target_xy'][0] == PLACEHOLDER_INVALID_VALUE:
			raise ValueError(f"Target annotation invalid for {entry.entry_idx}: {data['annotation_target_xy']}")

		return cls.from_xy_image(data['annotation_target_xy'], entry)

	@classmethod
	def from_baseline(cls, entry: BWEntry, baseline: 'BaselineBase') -> 'PredictedTarget':
		pred = baseline(entry)

		if pred['coords'] == PLACEHOLDER_INVALID_VALUE:
			# return None
			raise ValueError(f"Ignoring this entry as the baseline does not apply")

		return cls._default_class(entry, pred_coords=pred['coords'], pred_repair=False)

	@classmethod
	def from_xy_image(cls, pred_xy: List[float], entry: BWEntry) -> 'PredictedTarget':
		pred_coords = unity_utils.convert_xy_to_coords(pred_xy, DEFAULT_IMAGE_SIZE)

		return cls._default_class(
			entry,
			pred_xy=pred_xy,
			pred_coords=pred_coords,
			# true_xy=unity_utils.convert_coords_to_xy(entry['events'][-1]['true_target_coords']),
			# true_coords=entry['events'][-1]['true_target_coords'],
			# cand_xy=entry['events'][-2].get('cand_target_xy'),
			# cand_coords=entry['events'][-2].get('cand_target_coords'),
		)

	@classmethod
	def from_bbox(cls, pred_bbox: List[float], contains_repair: bool, entry: BWEntry) -> 'PredictedTarget':
		pred_xy = _calculate_midpoint(pred_bbox)
		pred_coords = unity_utils.convert_xy_to_coords(pred_xy, DEFAULT_IMAGE_SIZE)

		return cls._default_class(
			entry,
			pred_bbox=pred_bbox,
			pred_xy=pred_xy,
			pred_coords=pred_coords,
			pred_repair=contains_repair,
			# true_xy=unity_utils.convert_coords_to_xy(entry['events'][-1]['true_target_coords']),
			# true_coords=entry['events'][-1]['true_target_coords'],
			# cand_xy=entry['events'][-2].get('cand_target_xy'),
			# cand_coords=entry['events'][-2].get('cand_target_coords'),
		)


	@classmethod
	def _default_class(cls, entry: BWEntry, **kwargs) -> 'PredictedTarget':
		return cls(
			**kwargs,
			true_xy=entry.unity.true_target_xy,
			true_coords=entry.true_target_coords,
			true_bbox=entry.unity.true_target_bbox,
			true_repair=entry.dpo_type == DPOEntryType.INSTRUCTION_TO_REPAIR,
			cand_xy=entry.cand_target_xy if entry.has_candidate else None,
			cand_coords=entry.cand_target_coords if entry.has_candidate else None,
		)

	@property
	def radius_1_accuracy(self) -> int:
		"""
		:return: 1 if pred_coords is within 1 block radius of true_coords, 0 otherwise
		"""
		return self.get_radius_accuracy(1)

	@property
	def radius_2_accuracy(self) -> int:
		"""
		:return: 1 if pred_coords is within 1 block radius of true_coords, 0 otherwise
		"""
		return self.get_radius_accuracy(2)

	def get_radius_accuracy(self, radius: int = 1, distance_measure: float = 1) -> int:
		"""
		Uses distance_measure as 1 as that is the block distance,
		but you may want to use another value to mean 1, such as 3.22 etc

		:return: 1 if pred_coords is within <radius> blocks of true_coords, 0 otherwise
		"""
		return 1 if self.block_distance < radius * distance_measure else 0


class InvalidPredictedSource(PredictedSource):
	# could prob add some reason here, but let's leave it empty for now
	is_valid: bool = False


class InvalidPredictedTarget(PredictedTarget):
	# could prob add some reason here, but let's leave it empty for now
	is_valid: bool = False


class DatasetBase(pydantic.BaseModel):
	name: str
	_items: List[Union[BWEntry, AnnotatedPrediction]]

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self._items = kwargs['_items']      # need this as pydantic ignores _items as a field

	# the following methods make this behave like a list
	def __getitem__(self, index) -> Union[BWEntry, AnnotatedPrediction]:
		return self._items[index]

	def __setitem__(self, index, value):
		self._items[index] = value

	def __delitem__(self, index):
		del self._items[index]

	def __iter__(self) -> Iterator[Union[BWEntry, AnnotatedPrediction]]:
		return iter(self._items)

	def __len__(self):
		return len(self._items)

	def get_summary_stats(self) -> Dict[str, Union[int, float, str]]:
		return {
			'name': self.name,
			'n_entries': len(self),
		}


# todo: move this away from here
class DPOEntryType(enum.Enum):
	NULL = 0  # invalid? not sure how to handle this
	INSTRUCTION_CLEAN = 'instruction_clean'  # at least 1 human got it right OR Bisk's original data
	INSTRUCTION_WITH_TPR = 'instruction_with_tpr'  # full TPR with correction
	INSTRUCTION_TO_REPAIR = 'instruction_to_repair'  # TPR without the upcoming correction (wrong action)


class BWEntrySet(DatasetBase, pydantic.BaseModel):
	_items: List[BWEntry]
	_train: List[BWEntry] = []
	_dev: List[BWEntry] = []
	_test: List[BWEntry] = []
	_split_sets: bool = False

	def __init__(self, **kwargs):
		if '_items' not in kwargs:
			# two ways of initialising, either from a list of entries, or from splits
			kwargs['_items'] = kwargs['_train'] + kwargs['_dev'] + kwargs['_test']

		super().__init__(**kwargs)
		logger.info(f"BWEntrySet: {len(self)} {type(self._items[0]).__name__ if len(self._items) > 0 else 'entries'} {self.name} loaded")

		self._train = kwargs['_train'] if '_train' in kwargs else []
		self._dev = kwargs['_dev'] if '_dev' in kwargs else []
		self._test = kwargs['_test'] if '_test' in kwargs else []
		if len(self._train) + len(self._dev) + len(self._test) > 0:
			self._split_sets = True
			# logger.info(f"Split sizes: train={len(self._train)}, dev={len(self._dev)}, test={len(self._test)}")
			assert len(self._items) == len(self._train) + len(self._dev) + len(self._test), "Invalid sizes"

		logger.info(f"   summary: {self.get_summary_stats()}")

	@classmethod
	def from_collected_entries(cls, split_sets=True, splits=None, generated_data_path=None, annotated_data_path=None) -> 'BWEntrySet':
		# takes the raw dict entries and converts them to BWEntry objects
		keep_splits = splits if splits is not None else ['train', 'dev', 'test']
		splits = ['train', 'dev', 'test']
		entries = {split: [] for split in splits}

		def remove_filler_messages(_entry) -> dict:
			new_events = []
			for i, message in enumerate(_entry['events']):

				if 'utterance' not in message:
					new_events.append(message)
				elif 'I am helping you arrange blocks' in message['utterance']:
					continue
				elif 'Which block should I move and where to' in message['utterance']:
					continue
				elif 'Make sure you give me enough details' in message['utterance']:
					continue
				else:
					new_events.append(message)

			new_entry = _entry.copy()
			new_entry['events'] = new_events
			return new_entry

		all_entry_dicts = get_dialogue_entries('dialogues_all', remove_fully_annotated=False, annotations_data_path=annotated_data_path)

		# missing the unity data! Load it
		unity_entries = {split: load_entries(split, generated_data_path=generated_data_path) for split in splits}
		# unity_entries = {**load_entries('train'), **load_entries('dev'), **load_entries('test')}

		for entry in all_entry_dicts:
			key_name = entry['origin_img'].replace('.png', '-u0')
			for split in splits:
				if key_name in unity_entries[split]:
					break
			else:
				# logger.error(f"Unity data not found for {key_name}")
				raise ValueError(f"Unity data not found for {key_name}")

			entry['unity'] = unity_entries[split][key_name]
			# if entry['entry_type'] != 'instruction':
			# 	continue
			entry = remove_filler_messages(entry)
			entries[split].append(CollectedBWEntry.from_split_entry(entry))

		# return cls(_items=list(map(CollectedBWEntry.from_split_entry, all_entries)))

		for _sp in splits:
			if _sp not in keep_splits:
				entries[_sp] = []

		if split_sets:
			return cls(
				name='collected_entries',
				_train=entries['train'], _dev=entries['dev'], _test=entries['test'])
		else:
			return cls(
				name='collected_entries',
				_items=entries['train'] + entries['dev'] + entries['test'])

	@classmethod
	def from_original_entries(cls, splits=None, generated_data_path=None, block_world_path=None) -> 'BWEntrySet':
		if splits is None:
			splits = ['train', 'dev', 'test']
		if block_world_path is None:
			block_world_path = '../../BlockWorld-Random'

		entries = {split: [] for split in splits}
		# two ways of doing this: either have 1 set per split, or have an overall set and then inside them a train/dev/test
		unity_entries = {**{k: v for split in splits for k, v in load_entries(split, generated_data_path=generated_data_path).items()}}
		# unity_entries = {**load_entries('train'), **load_entries('dev'), **load_entries('test')}

		# read from the original entries, assuming these are in a BlockWorld/Random folder
		for split in splits:
			datum_list = read_jsonl(os.path.join(block_world_path, f"{split}set.json"))
			# each datum has: a list of states (usually 10), and each state has up to 8 paraphrases of the same instruction
			# we need each paraphrase to be its own entry
			for datum in datum_list:   # last state is ?? TODO fix
				for state_i, state in enumerate(datum['notes']):
					for instruction_i, instruction in enumerate(state['notes']):
						entries[split].append(OriginalBWEntry.from_utterance(
							entry_idx=f"{datum['images'][state_i].replace('.png', '')}_u{instruction_i}",
							utterance=instruction,
							unity_dict=unity_entries[datum['images'][state_i].replace('.png', '-u0')],
							blocks_coords=datum['states'][state_i],
							image=datum['images'][state_i]
						))

		return cls(
			name='original_entries',
			_train=entries['train'], _dev=entries['dev'], _test=entries['test'])

	def get_entry_by_idx(self, entry_idx: str) -> BWEntry:
		# todo: probably could be improved with an internal dict in __init__
		for entry in self._items:
			if entry.entry_idx == entry_idx:
				return entry
		raise ValueError(f"Entry {entry_idx} not found in dataset")

	@property
	def train_entries(self) -> List[BWEntry]:
		if self._split_sets: return self._train
		else: raise ValueError("No split entries")

	@property
	def dev_entries(self) -> List[BWEntry]:
		if self._split_sets: return self._dev
		else: raise ValueError("No split entries")

	@property
	def test_entries(self) -> List[BWEntry]:
		if self._split_sets: return self._test
		else: raise ValueError("No split entries")

	def get_summary_stats(self) -> Dict[str, Union[int, float, str]]:
		return {
			**super().get_summary_stats(),
			'num_train': num_with_perc(len(self._train), len(self)),
			'num_dev': num_with_perc(len(self._dev), len(self)),
			'num_test': num_with_perc(len(self._test), len(self)),
			'num_instruction': num_with_perc(
				sum([1 for entry in self if entry.is_instruction]), len(self)),
			'num_correction_source': num_with_perc(
				sum([1 for entry in self if entry.type == BWEntryType.CORRECTION_SOURCE]), len(self)),
			'num_correction_target': num_with_perc(
				sum([1 for entry in self if entry.type == BWEntryType.CORRECTION_TARGET]), len(self)),
			'unique_worlds_train': len(set([entry.world_name for entry in self._train])),
			'unique_worlds_dev': len(set([entry.world_name for entry in self._dev])),
			'unique_worlds_test': len(set([entry.world_name for entry in self._test])),
		}

	def get_stats(self, attribute_func, mode='count') -> dict:
		# for a particular attribute, calculate stats
		if mode == 'count':
			mode_func = lambda x: len(x)
		elif mode == 'unique':
			mode_func = lambda x: len(set(x))
		elif mode == 'sum':
			mode_func = lambda x: sum(x)
		else:
			raise NotImplementedError

		return {
			'train': mode_func([attribute_func(x) for x in self.train_entries]),
			'dev': mode_func([attribute_func(x) for x in self.dev_entries]),
			'test': mode_func([attribute_func(x) for x in self.test_entries]),
			'total': mode_func([attribute_func(x) for x in self])
		}

	def to_hf_dataset(self, data_config: configs.DataConfig, split: str, *, max_num_entries: int = None) -> 'datasets.Dataset':
		# todo: create a config object for the message formatting
		base_img_path, _, _ = get_data_paths()

		# assert data_config.goal in ['source', 'target'], f"Invalid goal: {data_config.goal}"
		# assert data_config.input_task_instruction in ['system', 'user'], f"Invalid prompt_task_instruction: {data_config.input_task_instruction}"

		if data_config.goal == 'source':
			instruction = "Answer only with the bounding box of the block mentioned."
			generation_prompt = lambda x: \
				f"block bounding box {unity_utils.bbox_as_str(normalise_bbox(x.unity.true_source_bbox))}"
		else:
			instruction = "Answer only with the final bounding box location mentioned."
			generation_prompt = lambda x: \
				f"location bounding box {unity_utils.bbox_as_str(normalise_bbox(x.unity.true_target_bbox))}"

		# include repair in those instances where a repair is needed
		if data_config.output_repair:
			# just use the first repair for now
			repair_str = parsing.REPAIRS[data_config.goal][0] if data_config.output_repair_form == 'sentence' else 'repair'
			repair = lambda x: repair_str if x.dpo_type == DPOEntryType.INSTRUCTION_TO_REPAIR else ''
		else:
			repair = lambda x: ''

		instruction = f"{instruction} The bounding box consists of 4 values between 0 and 1. Here is an example: [0.000, 0.123, 0.075, 0.204]. "

		extra_args = {f"{data_config.input_task_instruction}_instruction": instruction, "repair_form": data_config.output_repair_form if data_config.input_unify_repair else None}
		dataset = datasets.Dataset.from_list([
			{
				'entry_idx': entry.entry_idx,
				'messages': entry.messages_as_dict(turn_masking=data_config.input_turn_masking, **extra_args),
				'answer': f"{generation_prompt(entry)} {repair(entry) if data_config.output_repair else ''}".strip(),
				'image': os.path.join(base_img_path, entry.image_with_resolution),
				'image_name': entry.image_with_resolution,
			} for entry in getattr(self, f"_{split}")[:]
                if (data_config.goal == 'source' and entry.has_source) or (data_config.goal == 'target' and entry.has_target)][:max_num_entries]).cast_column("image", datasets.Image())

		logger.info(f"Dataset {self.name} split {split:5s} converted to HuggingFace dataset with {len(dataset)} entries")
		return dataset

	@classmethod
	def balanced_dataset(
		cls, original_set: 'BWEntrySet', collected_set: 'BWEntrySet',
		distribution: str, respect_original_splits: bool = False, split_sizes: str = '70/10/20') -> 'BWEntrySet':
		"""
		Takes the original and collected datasets and balances them according to the distribution specified, resulting a new, combined set.
		This creates a final BWEntrySet with the specified entries.

		:param original_set:
		:param collected_set:
		:param distribution: choose one from:
			- 'balanced': NOT IMPLEMENTED - equal number of entries from each dataset per split, so
			  50/50. Some entries from the larger dataset will be skipped.
			- 'combined': all entries from both datasets, does not respect split sizes
			- 'balanced_by_collected': we calculate (see Notion) a 70/10/20 split of the collected
			  dataset, and then add original dataset entries to match this split up to 50% of the
			  total. Some original entries are not used.

		:param respect_original_splits:
		:param split_sizes: floats for the train/dev/test split sizes, as % of the total
		:return:
		"""
		split_sizes = [float(x) / 100 for x in split_sizes.split('/')]
		assert sum(split_sizes) == 1, f"Invalid split sizes: {split_sizes}"
		_stats = {
			'distribution': distribution,
			'splits': split_sizes,
		}

		if distribution == 'balanced':
			raise NotImplementedError("Not implemented yet")

		elif distribution == 'combined':
			# final entries original/collected
			final_original = {
				'train': original_set.train_entries,
				'dev': original_set.dev_entries,
				'test': original_set.test_entries,
			}
			final_collected = {
				'train': collected_set.train_entries,
				'dev': collected_set.dev_entries,
				'test': collected_set.test_entries,
			}

		elif distribution == 'balanced_by_collected':
			assert not respect_original_splits, "Not compatible, read function description"
			# test = 8
			collected_test_world_names = ['final_4', 'final_24']
			original_test_world_names = ['final_5', 'final_6', 'final_7', 'final_8', 'final_9',
			                             'final_10']
			# dev = 4
			collected_dev_world_names = ['final_13']
			original_dev_world_names = ['final_0', 'final_1', 'final_2']

			final_original = {
				'train': [entry for entry in original_set if
				          entry.world_name not in collected_test_world_names + collected_dev_world_names + original_dev_world_names + original_test_world_names],
				'dev': [entry for entry in original_set if
				        entry.world_name in original_dev_world_names],
				'test': [entry for entry in original_set if
				         entry.world_name in original_test_world_names],
			}
			final_collected = {
				'train': [entry for entry in collected_set if
				          entry.world_name not in collected_test_world_names + collected_dev_world_names],
				'dev': [entry for entry in collected_set if
				        entry.world_name in collected_dev_world_names],
				'test': [entry for entry in collected_set if
				         entry.world_name in collected_test_world_names],
			}

			# now balance to have approx 50/50 in each split,
			# by removing the most common entry from the original dataset
			collected_data_threshold = 0.47  # only remove if the % of collected is below this
			for i in range(8, 0, -1):
				_tmp_original = final_original.copy()
				for split in ['train', 'dev', 'test']:
					if len(final_collected[split]) / len(final_original[split] + final_collected[
						split]) < collected_data_threshold:
						# remove!
						_tmp_original[split] = [
							entry for entry in final_original[split]
							if not entry.entry_idx.endswith('u' + str(i))]
						# keep track of the removed entries
						_stats[f"removed_u{i}_{split}"] = len(final_original[split]) - len(
							_tmp_original[split])
				final_original = _tmp_original
			# todo: there is a smarter way to do the above (know the amount needed to get to 50%, then remove only that)

		elif 'fixed_original_test' in distribution:
			assert not respect_original_splits, "Not compatible, read function description"
			if distribution == 'fixed_original_test':
				filter_func = lambda x: True
			elif 'prefer_corrections' in distribution:
				# this one has original instructions, but removes collected instructions as they will have corrections, as to not see dialogues twice
				filter_func = lambda x: x.is_original or x.is_correction
			elif 'instructions_to_repair' in distribution:
				filter_func = lambda x: x.is_instruction and x.is_collected
			elif 'instructions' in distribution:
				filter_func = lambda x: x.is_instruction
			elif 'corrections' in distribution:
				filter_func = lambda x: x.is_correction
			else:
				raise NotImplementedError(f"Invalid distribution: {distribution}")

			# list of worlds to move to test, then balance the rest - test original is always fixed
			world_names_to_move_to_test = ['final_27']

			# test = 8
			# collected_test_world_names = ['final_4', 'final_24']
			original_test_world_names = ['final_4', 'final_5', 'final_6', 'final_7', 'final_8', 'final_9', 'final_10', 'final_11']
			# dev = 4
			# collected_dev_world_names = ['final_13']
			original_dev_world_names = ['final_0', 'final_1', 'final_2', 'final_3']

			final_original = {
				'train': [entry for entry in original_set if
				          entry.world_name not in world_names_to_move_to_test + original_test_world_names + original_dev_world_names and filter_func(entry)],
				'dev': [entry for entry in original_set if
				        entry.world_name in original_dev_world_names and filter_func(entry)],
				'test': [entry for entry in original_set if
				         entry.world_name in original_test_world_names + world_names_to_move_to_test and filter_func(entry)],
			}
			final_collected = {
				'train': [entry for entry in collected_set if
				          entry.world_name not in world_names_to_move_to_test + original_test_world_names + original_dev_world_names and filter_func(entry)],
				'dev': [entry for entry in collected_set if
				        entry.world_name in original_dev_world_names and filter_func(entry)],
				'test': [entry for entry in collected_set if
				         entry.world_name in original_test_world_names + world_names_to_move_to_test and filter_func(entry)],
			}

			# now balance to have approx 50/50 in each split,
			# by removing the most common entry from the original dataset
			# collected_data_threshold = 0.47  # only remove if the % of collected is below this
			# for i in range(8, 0, -1):
			# 	_tmp_original = final_original.copy()
			# 	for split in ['train', 'dev', 'test']:
			# 		if len(final_collected[split]) / len(final_original[split] + final_collected[
			# 			split]) < collected_data_threshold:
			# 			# remove!
			# 			_tmp_original[split] = [
			# 				entry for entry in final_original[split]
			# 				if not entry.entry_idx.endswith('u' + str(i))]
			# 			# keep track of the removed entries
			# 			_stats[f"removed_u{i}_{split}"] = len(final_original[split]) - len(
			# 				_tmp_original[split])
			# 	final_original = _tmp_original

		else:
			raise NotImplementedError(f"Invalid distribution: {distribution}")

		result_set = cls(
			_train=final_original['train'] + final_collected['train'],
			_dev=final_original['dev'] + final_collected['dev'],
			_test=final_original['test'] + final_collected['test'],
			name=f"balanced_dataset-{distribution}"
		)

		# print out some key stats, such as the % of each
		_stats.update({
			'train_original': num_with_perc(len(
				final_original['train']), len(result_set.train_entries)),
			'train_collected': num_with_perc(len(
				final_collected['train']), len(result_set.train_entries)),
			'dev_original': num_with_perc(len(final_original['dev']), len(result_set.dev_entries)),
			'dev_collected': num_with_perc(len(
				final_collected['dev']), len(result_set.dev_entries)),
			'test_original': num_with_perc(len(
				final_original['test']), len(result_set.test_entries)),
			'test_collected': num_with_perc(len(
				final_collected['test']), len(result_set.test_entries)),
		})
		logger.info(f"Resulting dataset balance: {json.dumps(_stats, indent=4)}")
		result_set.ensure_unique_worlds_across_splits()

		return result_set

	def ensure_unique_worlds_across_splits(self):
		# checks if the worlds from the entries are unique across the splits
		train_worlds = set([entry.world_name for entry in self.train_entries])
		dev_worlds = set([entry.world_name for entry in self.dev_entries])
		test_worlds = set([entry.world_name for entry in self.test_entries])

		assert len(train_worlds.intersection(dev_worlds)) == 0, \
			f"Worlds overlap between train and dev: {train_worlds.intersection(dev_worlds)}"
		assert len(train_worlds.intersection(test_worlds)) == 0, \
			f"Worlds overlap between train and test: {train_worlds.intersection(test_worlds)}"
		assert len(dev_worlds.intersection(test_worlds)) == 0, \
			f"Worlds overlap between dev and test: {dev_worlds.intersection(test_worlds)}"
		logger.debug(f"Worlds are unique across splits")


def _get_annotation_files(annotations_dir: str) -> list:
	# get all the files in the annotations directory
	return [x for x in os.listdir(annotations_dir) if x.endswith('.jsonl')]


def iterate_annotations_sysenv(participant_ids: Optional[List[str]] = None) -> Iterable:
	# new method to iterate over annotations using a system variable, so it is better than loading server.py
	annotations_dir = os.environ.get('ANNOTATIONS_DIR', 'annotations')
	# get all files
	for file in _get_annotation_files(annotations_dir):
		if participant_ids and file.replace('.jsonl', '') not in participant_ids:
			# print(f"Skipping {file}")
			continue

		# read files with their annotations
		entries = read_jsonl(os.path.join(annotations_dir, file))
		for entry in entries:
			if 'entry_idx' not in entry:
				entry['entry_idx'] = entry['annotations']['entry_idx']
			if 'timestamp_start' not in entry['annotations']:
				entry['annotations']['timestamp_start'] = entry['timestamp_start']
				entry['annotations']['timestamp_end'] = entry['timestamp_end']

			yield entry


def get_annotated_idxs(participant_ids: Optional[List[str]] = None) -> collections.Counter:
	annotated_idxs = collections.Counter()
	for entry in iterate_annotations_sysenv(participant_ids):
		annotated_idxs[entry['entry_idx']] += 1

	# logger.info(f"   {sum(annotated_idxs.values())} annotated entries found")
	return annotated_idxs


class AnnotationSet(DatasetBase):
	_items: List[AnnotatedPrediction]
	_fully_annotated_predictions: Dict[str, List[AnnotatedPrediction]] = {}
	# fully_annotated_entries = collections.Counter()
	errored_annotations: int = 0
	incomplete_annotations: int = 0

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		# self._items = self._load_annotations()
		self._fully_annotated_predictions = self._get_fully_annotated_predictions()

	@classmethod
	def from_data_annotation(cls, bw_entries: BWEntrySet, participants: List[str] = None, exclude_participant_outliers: bool = True) -> 'AnnotationSet':
		if participants is None:
			participants = PARTICIPANTS

		if exclude_participant_outliers:
			participants = [p for p in participants if p not in PARTICIPANT_OUTLIERS]

		image_size = None
		errored_annotations = 0
		incomplete_annotations = 0
		annotations = {}

		for annot in iterate_annotations_sysenv(participants):
			# input(entry['annotations']['participant_idx'])
			try:
				entry = bw_entries.get_entry_by_idx(annot['entry_idx'])
			except ValueError as ex:
				# basically didn't find the entry, so skip
				continue

			if image_size is None:
				image_size = IMAGE_SIZES[
					(annot['annotations'].get('image') or annot['image']).split('-')[
						-1].replace('.png', '')]
				assert image_size == DEFAULT_IMAGE_SIZE, f"time to fix hard-coded value"

			try:
				_annotation = AnnotatedPrediction.from_annotation_collection(entry, **annot)
				# self.fully_annotated_entries[_annotation.entry_idx] += 1

				# make sure we don't have repeated entries
				assert f"{_annotation.entry_idx}_{_annotation.participant_idx}" not in annotations, f"Repeated entry_idx {annot['entry_idx']} in annotations"
				annotations[f"{_annotation.entry_idx}_{_annotation.participant_idx}"] = _annotation

			except (KeyError, ValueError) as ex:
				if str(PLACEHOLDER_INVALID_VALUE) in str(ex) or 'annotation missing' in str(ex):  # smaller warning, don't care much
					logger.warning(f"Ignoring incomplete annotation from {annot['annotations'].get('participant_idx') or annot['user_idx']} / {annot['entry_idx']}")
					incomplete_annotations += 1
				else:
					logger.exception(ex)
					logger.error(f"Issue with annotation from {annot['annotations'].get('participant_idx') or annot['user_idx']} / {annot['entry_idx']}\n{json.dumps(annot['annotations'], indent=4)}")
					errored_annotations += 1

		logger.info(f"{len(annotations)} {AnnotatedPrediction.__name__} from_data_annotation loaded")
		return cls(
			name='annotations',
			_items=list(annotations.values()),
			errored_annotations=errored_annotations,
			incomplete_annotations=incomplete_annotations,
		)

	# @property
	# def annotations(self) -> List[AnnotatedPrediction]:
	# 	return self._items

	@property
	def fully_annotated_predictions(self):
		return self._fully_annotated_predictions

	@property
	def valid_annotations(self) -> int:
		return len(self._items)

	@property
	def participant_ids(self) -> List[str]:
		return sorted(list(set([p.participant_idx for p in self._items])))

	def _get_fully_annotated_predictions(self) -> Dict[str, List[AnnotatedPrediction]]:
		"""
		Get a Dict of the entries fully annotated, where each is a dict with the
		annotations for each participant.

		:return: dict of entry_idx with list of annotations: {
			'entry_123': [AnnotationEntry, AnnotationEntry...],
			'entry_234': ... }
		"""
		annotated_idxs = get_annotated_idxs(PARTICIPANTS)
		_fully_annotated_entries = {}
		for annotation in self._items:
			# check if enough annotations for that entry
			if annotated_idxs[annotation.entry_idx] >= entry_selection_config.n_annotations_per_entry:
				# append to dict of lists
				if annotation.entry_idx not in _fully_annotated_entries:
					_fully_annotated_entries[annotation.entry_idx] = []
				_fully_annotated_entries[annotation.entry_idx].append(annotation)

		return _fully_annotated_entries

	def get_annotation_summary(self) -> dict:
		return {
			'valid_annotations': self.valid_annotations,
			'errored_annotations': self.errored_annotations,
			'incomplete_annotations': self.incomplete_annotations,
		}

	def get_summary_stats(self) -> Dict[str, Union[int, float]]:
		return {
			**super().get_summary_stats(),
			'unique_participants': len(self.participant_ids),
			**self.get_annotation_summary(),
		}

	def get_valid_entry_set(self) -> 'BWEntrySet':
		# get the entries that have valid annotations
		return BWEntrySet(
			name='annotated_entries',
			_items=[pred.entry for pred in self._items]
		)

	def get_filtered_subset(self, filter_func: Callable[[AnnotatedPrediction], bool], name: str = None) -> 'AnnotationSet':
		def _filter_wrapper(x):
			try:
				return filter_func(x)
			except Exception as ex:
				# logger.exception(ex)
				return False

		return AnnotationSet(
			name=f"{self.name}_filtered" if name is None else name,
			_items=[pred for pred in self._items if _filter_wrapper(pred)]
		)


class BaselineBase(abc.ABC):
	name: str

	@abc.abstractmethod
	def __call__(self, entry: BWEntry) -> dict:
		pass


class BaselineRandomBlock(BaselineBase):
	name: str = 'baseline random block'

	def __call__(self, entry: BWEntry) -> dict:
		index = np.random.randint(0, 10)
		return {
			'coords': entry.blocks_coords[index],
			'bbox': entry.unity.blocks_bboxes[index],
			'index': index,
		}


class BaselineRandomCoords(BaselineBase):
	name: str = 'baseline random coords'

	def __call__(self, entry: BWEntry) -> dict:
		return {
			'index': PLACEHOLDER_INVALID_VALUE,
			'coords': [random.uniform(-1, 1), unity_utils.BLOCK_LENGTH, random.uniform(-1, 1)],
		}


from .utils.load_utils import discrete_table
class BaselineRandomBoundingBox(BaselineBase):
	name: str = 'baseline random bounding box'

	def __call__(self, entry: BWEntry) -> dict:
		return {
			'index': PLACEHOLDER_INVALID_VALUE,
			'bbox': random.choice(discrete_table['cells']),
			'coords': PLACEHOLDER_INVALID_VALUE
		}


class BaselineOracleSourceBlock(BaselineBase):
	name: str = 'baseline oracle source block'

	def __call__(self, entry: BWEntry) -> dict:
		return {
			'index': entry.unity.true_source_index if entry.has_source else PLACEHOLDER_INVALID_VALUE,
			'bbox': entry.unity.blocks_bboxes[entry.unity.true_source_index] if entry.has_source else [],
			'coords': entry.unity.true_source_coords if entry.has_source else PLACEHOLDER_INVALID_VALUE,
		}


class BaselineOracleTargetCoords(BaselineBase):
	name: str = 'baseline oracle target coords'

	def __call__(self, entry: BWEntry) -> dict:
		return {
			'index': PLACEHOLDER_INVALID_VALUE,
			'bbox': entry.unity.true_target_bbox if entry.has_target else [],
			'coords': entry.unity.true_target_coords if entry.has_target else PLACEHOLDER_INVALID_VALUE,
		}

# def baseline_random_block(entry: dict) -> dict:
# 	blocks = json.loads(entry['block_coords'])
# 	pred_index = random.randint(0, len(blocks) - 1)
#
# 	return {
# 		'index': pred_index,
# 		'bbox': entry['blocksBBoxes'][pred_index],
# 		'coords': blocks[pred_index]
# 	}
#
#
# def baseline_random_coords(entry: dict) -> dict:
# 	return {
# 		'coords': [random.uniform(-1, 1), BLOCK_LENGTH, random.uniform(-1, 1)],
# 	}

# todo: oracle bbox? or oracle xy? however, issue is that baseline funcs get entries, but not if we are doing source/target pred

# def baseline_oracle_block(entry: dict) -> dict:
# 	return {
# 		'index': entry['sourceIndex'] if 'true_source_coords' in entry else PLACEHOLDER_INVALID_VALUE,
# 		'bbox': entry['blocksBBoxes'][entry['sourceIndex']] if 'true_source_coords' in entry else [],
# 		'coords': entry['true_source_coords'] if 'true_source_coords' in entry else PLACEHOLDER_INVALID_VALUE
# 	}
#
# def baseline_oracle_target_coords(entry: dict) -> dict:
# 	return {
# 		'index': PLACEHOLDER_INVALID_VALUE,
# 		'bbox': entry['targetBBox'] if 'true_target_coords' in entry else [],
# 		'coords': entry['true_target_coords'] if 'true_target_coords' in entry else PLACEHOLDER_INVALID_VALUE
# 	}

# def baseline_oracle_coords(entry: dict) -> dict:
# 	return {
# 		'index': entry['sourceIndex'] if goal == 'source' else -1,
# 		'bbox': entry['blocksBBoxes'][entry['sourceIndex']] if goal == 'source' else entry['targetBBox'],
# 		'coords': entry['blocks'][entry['sourceIndex']] if goal == 'source' else entry['targetLocation']
# 	}
# def baseline_oracle_bbox(entry: dict) -> dict:
# 	return {
# 		'index': entry['sourceIndex'] if goal == 'source' else -1,
# 		'bbox': entry['blocksBBoxes'][entry['sourceIndex']] if goal == 'source' else entry['targetBBox'],
# 		'coords': bbox_to_bw_coords(entry['blocksBBoxes'][entry['sourceIndex']] if goal == 'source' else entry['targetBBox'])
# 	}

class Idefics2Prediction(PredictionItem):
	contains_repair: bool

	def __init__(self, **kwargs):
		super().__init__(**kwargs)

	# @classmethod
	# def from_model_output(cls, entry_idx: str, generated_text: str) -> 'Idefics2Prediction':
	# 	# text to a bbox
	# 	generated_text = generated_text.strip().strip('[').strip(']')
	# 	bbox = [float(x) for x in generated_text.split(",")]
	# 	input(bbox)
	# 	entry = original_entries.get_entry_by_idx(entry_idx)
	# 	midpoint = evaluate_bw_results._calculate_midpoint(bbox)
	# 	return cls(
	# 		entry_idx=entry_idx,
	# 		entry=entry,
	# 		source=data_utils.PredictedSource.from_xy_image(midpoint, entry), # if entry.has_source else None,
	# 		target=None,
	# 	)

	@classmethod
	def from_model_output(cls, entry, generated_text: str, goal: str, new_parser=None) -> 'Idefics2Prediction':
		# TODO: remove new_parser arg
		# sample outputs:
		#   [0.0, 0.0, 0.0, 0.0]
		#   'The bounding box of the block mentioned is [1.0, 0.0]'
		# text to a bbox
		assert goal in ['source', 'target']
		# take the last bounding box in the text, as sometimes models generate more than 1
		parsed_text = f"a {generated_text}{']' if ']' not in generated_text else ''} a".split(']')[-2].split('[')[-1]
		parsed_text = parsed_text.strip().strip('.').strip('{').strip('}').strip('(').strip(')')
		try:
			bbox = [float(x.strip()) for x in parsed_text.split(",")]
			if len(bbox) == 4:
				if sum(bbox) <= 4:
					# bbox is in range 0-1, needs to scale it up to pixels
					bbox = scale_bbox_to_image(bbox)

				output = parsing.parse_generated_text(generated_text)
				# print(bbox)
				# input(output)
				if bbox != output.bbox:
					logger.warning(f"bbox mismatch: {bbox} vs {output} in text '{generated_text}'")

				# ignore
				# midpoint = _calculate_midpoint(bbox)
				# print(f"\t{generated_text=}; {parsed_text=}; {bbox=}; {midpoint=}")
				return cls(
					entry_idx=entry.entry_idx,
					entry=entry,
					source=PredictedSource.from_bbox(output.bbox, output.contains_repair, entry) if goal == 'source' else None,
					target=PredictedTarget.from_bbox(output.bbox, output.contains_repair, entry) if goal == 'target' else None,
					contains_repair=output.contains_repair
				)

			else:
				midpoint = None
				# todo: fix this, it ignores entries for now
				# raise ValueError(f"Invalid bbox: {bbox}")
				# logger.warning(f"Parsing issue in {entry.entry_idx}: 'invalid bbox {bbox}' | {generated_text=}; {parsed_text=}; {parsed_text.split(',')}")

		except ValueError as ex:
			logger.warning(f"Parsing issue in {entry.entry_idx}: '{ex}' | {generated_text=}; {parsed_text=}; {parsed_text.split(',')}")
			pass

		# if random_pred_on_issue:
		# 	# return random predictions instead
		# 	return cls(
		# 		entry_idx=entry.entry_idx,
		# 		entry=entry,
		# 		source=PredictedSource.from_baseline(entry, BaselineRandomBlock()) if goal == 'source' else None,
		# 		target=PredictedTarget.from_baseline(entry, BaselineRandomCoords()) if goal == 'target' else None,
		# 	)

		# this handles the way that invalid predictions count towards metrics
		# current code creates an invalid prediction (is_valid=False) but still uses baseline (random) values for metrics
		return cls(
			entry_idx=entry.entry_idx,
			entry=entry,
			source=InvalidPredictedSource.from_baseline(entry, BaselineRandomBlock()) if goal == 'source' else None,
			target=InvalidPredictedTarget.from_baseline(entry, BaselineRandomCoords()) if goal == 'target' else None,
			contains_repair=False       # todo: fix this
		)


	@classmethod
	def from_model_batch_output(cls, batch_entries, batch_outputs, goal: str, new_parser=None) -> List['Idefics2Prediction']:
		batch = [cls.from_model_output(entry_idx, generated_text, goal, new_parser=new_parser) for entry_idx, generated_text in zip(batch_entries, batch_outputs)]
		# print(f"\t{[type(x) for x in batch]}")
		# return [x for x in batch if x is not None]
		return batch


def scale_bbox_to_image(bbox: List[float]) -> List[float]:
	# scale the bbox from the 0-1 to the actual image size
	return [bbox[0] * DEFAULT_IMAGE_SIZE[0], bbox[1] * DEFAULT_IMAGE_SIZE[1], bbox[2] * DEFAULT_IMAGE_SIZE[0], bbox[3] * DEFAULT_IMAGE_SIZE[1]]


def normalise_bbox(bbox: List[float]) -> List[float]:
	# normalise the bbox to be within 0-1
	return [bbox[0] / DEFAULT_IMAGE_SIZE[0], bbox[1] / DEFAULT_IMAGE_SIZE[1], bbox[2] / DEFAULT_IMAGE_SIZE[0], bbox[3] / DEFAULT_IMAGE_SIZE[1]]


def get_dataset(dataset_name: str, generated_data_path: str, block_world_data_path: str) -> BWEntrySet:
	annotated_data = os.path.join(os.environ['UNITY_DATA_DIR'].replace(
		'generated_data', 'datasets'), 'bw-correction-dialogues', 'data')
	original_entries = BWEntrySet.from_original_entries(
		generated_data_path=generated_data_path,
		block_world_path=block_world_data_path)

	if dataset_name == 'original_entries':
		main_dataset = original_entries

	elif dataset_name == 'collected_entries':
		collected_entries = BWEntrySet.from_collected_entries(
			generated_data_path=generated_data_path,
			annotated_data_path=annotated_data)
		main_dataset = collected_entries

	else:
		collected_entries = BWEntrySet.from_collected_entries(
			generated_data_path=generated_data_path,
			annotated_data_path=annotated_data)

		balanced_entries = BWEntrySet.balanced_dataset(
			original_entries, collected_entries, dataset_name)
		main_dataset = balanced_entries
	# original_entries = data_utils.BWEntrySet.from_original_entries()

	# train_dataset = load_dataset("nielsr/docvqa_1200_examples", split="train")
	# train_dataset = train_dataset.remove_columns(['id', 'words', 'bounding_boxes', 'answer'])
	# print(train_dataset[10])
	# return main_dataset.to_hf_dataset(
	# 	split, _generated_data_path, args.goal, limit=24 if args.debug else None)

	# eval_dataset = load_dataset("nielsr/docvqa_1200_examples", split="test")
	# eval_dataset = eval_dataset.remove_columns(['id', 'words', 'bounding_boxes', 'answer'])
	# dev_dataset = main_dataset.to_hf_dataset(
	# 	'dev', _generated_data_path, args.goal, limit=12 if args.debug else None)
	# test_dataset = main_dataset.to_hf_dataset(
	# 	'test', _generated_data_path, args.goal, limit=24 if args.debug else None)

	return main_dataset


_cached_data = {}
def get_bw_dataset(dataset_name: str, use_cache: bool = True) -> BWEntrySet:
	# gets a dataset, may be cached for speeding up processing
	if use_cache and dataset_name in _cached_data:
		return _cached_data[dataset_name]
	else:
		generated_data, bw_data, _ = get_data_paths()
		_cached_data[dataset_name] = get_dataset(
			dataset_name, generated_data, bw_data)
		return get_bw_dataset(dataset_name)


def get_data_paths():
	# todo: refactor to another file
	generated_data_path = os.path.join(os.environ['UNITY_DATA_DIR'], '576p')
	bw_data_path = os.path.join(os.environ['UNITY_DATA_DIR'].replace(
		'generated_data', 'datasets'), 'BlockWorld-Random')
	annotated_data_path = os.path.join(os.environ['UNITY_DATA_DIR'].replace(
		'sharedscratch', 'block-world-research').replace(
		'generated_data', 'bw-correction-annotations'), 'annotations')
	os.environ['ANNOTATIONS_DIR'] = annotated_data_path

	return generated_data_path, bw_data_path, annotated_data_path
