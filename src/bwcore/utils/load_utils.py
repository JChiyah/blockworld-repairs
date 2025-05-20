# todo: move all this away from this file

import sys

import os
import csv
import json
import random
import logging

from typing import List, Dict, Callable, Tuple

import io
import cv2
import base64

import numpy as np
import torch
import colorlog
from tabulate import tabulate
from PIL import Image
from tqdm import tqdm

# sys.path.append('../bw-correction-dialogues')
from .. import get_logger
logger = get_logger()


def check_env_variables():
	for variable in ['UNITY_DATA_DIR', 'OUTPUT_DIR']:
		assert variable in os.environ, \
			f"Environment variable not set ({variable}): see block-world-training/README.md > Environment Variables"

		os.makedirs(os.environ[variable], exist_ok=True)

	# os.makedirs()

check_env_variables()

# Configure the logger
# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)-8s - %(name)s - [%(filename)s:%(lineno)-3d] %(message)s')

# Create a logger with colorlog
# logger = colorlog.getLogger()
# logger.setLevel(logging.INFO)
#
# # Create a handler with a colored log format
# log_format = '%(asctime)s - %(log_color)s%(levelname)-8s%(reset)s - [%(filename)12s:%(lineno)-3d] %(log_color)s%(message)s%(reset)s'
# formatter = colorlog.ColoredFormatter(
# 	log_format,
# 	datefmt='%Y-%m-%d %H:%M:%S',
# 	reset=True,
# 	log_colors={
# 		'DEBUG': 'cyan',
# 		'INFO': 'light_white',
# 		'WARNING': 'yellow',
# 		'ERROR': 'red',
# 		'CRITICAL': 'bold_red,bg_white',
# 	}
# )
#
# # only set up loggers once, so output does not repeat
# if not len(logger.handlers):
# 	# Create a StreamHandler and set the formatter
# 	ch = logging.StreamHandler()
# 	ch.setFormatter(formatter)
# 	logger.addHandler(ch)
#
#
# # Function to adjust the log level of a logger by name
# def set_logger_level(logger_name, log_level):
# 	logger_to_configure = logging.getLogger(logger_name)
# 	logger_to_configure.setLevel(log_level)
#
# # hide messages from PngImagePlugin
# set_logger_level('PIL.PngImagePlugin', logging.WARNING)


def load_predictions(results_path: str, data_path: str) -> Dict[str, dict]:
	# split the data path and get only filename
	data_file = data_path.split('/')[-1].replace('.tsv', '')

	with open(os.path.join(results_path, f"{data_file}_predict.json"), 'r') as file:
		_predictions = json.load(file)

	all_predictions = {}
	# make it same format as entries
	for pred in _predictions:
		pred['coords'] = bbox_to_bw_coords(pred['box'])
		pred['bbox'] = pred['box']
		del pred['box']

		all_predictions[pred['uniq_id']] = pred

	# we also add some configs
	if goal == 'source':
		all_predictions['accuracy'] = True
		all_predictions['distance'] = True
	elif goal == 'target':
		all_predictions['accuracy'] = False
		all_predictions['distance'] = True

	return all_predictions


def load_entries(data_path: str, _goal: str = None, generated_data_path=None) -> Dict[str, dict]:
	if generated_data_path is None:
		generated_data_path = os.path.join(os.environ['UNITY_DATA_DIR'], '576p')
	if 'bwdials' in data_path:
		# special case, just load all splits as we are evaluating the bwdials
		print('special!')
		return {**load_entries('train'), **load_entries('dev'), **load_entries('test')}
	elif 'dev' in data_path:
		jsonl_file = 'devset'
	elif 'test' in data_path:
		jsonl_file = 'testset'
	elif 'train' in data_path:
		jsonl_file = 'trainset'
	else:
		raise ValueError(f"'data' must be dev or test, got {data_path}")

	entries = {}
	with open(os.path.join(generated_data_path, jsonl_file + '.jsonl'), 'r') as file:
		for line in file:
			# Parse the JSON object in each line
			entry = json.loads(line)
			for i, utterance in enumerate(entry['utterances']):
				# fix a few entry field
				entry_idx = f"{entry['entryIdx']}-u{i}"     # fix entry_idx
				entry['entry_idx'] = entry_idx
				entry['true_index'] = entry['sourceIndex'] if _goal == 'source' else -1
				entry['true_bbox'] = entry['blocksBBoxes'][entry['sourceIndex']] if _goal == 'source' else entry['targetBBox']
				entry['true_coords'] = entry['blocks'][entry['sourceIndex']] if _goal == 'source' else entry['targetLocation']
				# make a copy and save it to the overall entry list
				entries[entry_idx] = entry.copy()

	return entries


bwdial_entries = {}

def evaluate(data_path: str, all_model_predictions: Dict[str, dict], baselines: List[Callable]) -> None:
	global bwdial_entries
	# we have the true entries, then we have the predictions
	# we want to calculate the metrics based on the predictions, as not all
	# entries have predictions (e.g., coming from bwdials, etc)

	mode = 'bbox'  # bbox
	entries = load_entries(data_path, goal)

	# so, first get the values from any dict in all_model_predictions
	_core_predictions = list(all_model_predictions.values())[0]
	# ensure all model predictions have the same length
	for _model_predictions in all_model_predictions.values():
		assert len(_model_predictions) == len(_core_predictions), f"All model predictions must have the same length, {len(_model_predictions)} != {len(_core_predictions)}"
	_core_predictions = {k: v for k, v in _core_predictions.items() if k not in ['accuracy', 'distance']}

	# we add the baseline callables to all_model_predictions, so we can iterate too
	for _baseline in baselines:
		all_model_predictions[_baseline.__name__] = {
			'accuracy': True if goal == 'source' else False,
			'distance': True,
			'callable': _baseline
		}

	# some initial empty dicts for later
	pred_entries = {}
	true_entries = {
		'entry_idx': [],
		'index': [],
		'bbox': [],
		'coords': []
	}

	# now iterate predictions
	for entry_idx in _core_predictions.keys():
		# input(entry_idx)
		# input(entries.keys())
		alternative_idx = None
		if entry_idx.startswith('di_'):
			# first time, load dict
			if not bwdial_entries:
				sys.path.append('/users/fjc3/block-world-research')
				from preprocess_bwdials import get_all_bwdial_entries, BWDIALS_DIR
				bwdial_entries = get_all_bwdial_entries(f"../../../{BWDIALS_DIR}")
				bwdial_entries = {elem['entry_idx']: elem for elem in bwdial_entries}

			alternative_idx = entry_idx
			entry_idx = bwdial_entries[entry_idx.replace('-u0', '')]['origin_img'].replace('.png', '-u0')
			# print(bwdial_entries.keys())
			# input(_core_predictions[entry_idx])

		# input(enr)
		t_entry = entries[entry_idx]
		true_entries['entry_idx'].append(entry_idx)
		true_entries['index'].append(t_entry['true_index'])
		true_entries['bbox'].append(t_entry['true_bbox'])
		true_entries['coords'].append(t_entry['true_coords'])

		# aaaand check each prediction, collect their results in lists in pred_entries
		for model_name, model_predictions in all_model_predictions.items():
			if model_name not in pred_entries:
				pred_entries[model_name] = {
					'bbox': [],
				}
				if model_predictions['accuracy']:
					pred_entries[model_name]['index'] = []
				if model_predictions['distance']:
					pred_entries[model_name]['coords'] = []

			if 'callable' in model_predictions: # we have a baseline, just get a pred
				pred_entry = model_predictions['callable'](t_entry)
			else:
				# print(entry_idx)
				# print(predictions.keys())
				# input(entry['image'])
				# # todo: fix this with ids
				# entry_idx = entry['image'].replace('_576p', '-u0')
				# print(entry_idx)
				# input(f"entry_idx: {entry_idx}, predictions: {predictions.keys()}")
				# entry_idx = 'final_4-4-u0'
				# input(predictions.keys())
				pred_entry = model_predictions[alternative_idx if alternative_idx else entry_idx]

			# finally add the pred by idx to the lists
			# input(pred['bbox'])
			if 'index' in pred_entries[model_name]:
				if 'index' not in pred_entry:
					pred_entry['index'] = pred_bbox_to_index(t_entry['blocksBBoxes'], pred_entry['bbox'])

				pred_entries[model_name]['index'].append(pred_entry['index'])

			pred_entries[model_name]['bbox'].append(pred_entry['bbox'])
			pred_entries[model_name]['coords'].append(pred_entry['coords'])

	logging.info(f"{data_path} -- Total entries: {len(true_entries['bbox'])}")
	logging.debug(f"calculating metrics using '{mode}'")

	# now I want to calculate the following metrics for each model:
	# accuracy, mean distance to gold, median distance to gold
	metrics = {}
	for model_name in pred_entries.keys():
		distance = metrics_distance(                # pred vs true
			pred_entries[model_name]['coords'], true_entries['coords'])
		_ap_scores = _calculate_ap_score(           # pred vs true
			torch.tensor(pred_entries[model_name]['bbox']),
			torch.tensor(true_entries['bbox']))
		ap_score = compute_score(
			{'_score_sum': _ap_scores.sum(), '_score_cnt': _ap_scores.size(0)})

		metrics[model_name] = {     # metrics to output in table, in order
			'ap_mean': ap_score,
			'accuracy_mean': 0.0,
			'distance_mean': distance['mean'],
			'distance_median': distance['median'],
		}
		# add accuracy if checking indexes too
		if 'index' in pred_entries[model_name]:
			accuracy = metrics_accuracy(pred_entries[model_name]['index'], true_entries['index'])
			metrics[model_name]['accuracy_mean'] = accuracy['mean']
		else:
			del metrics[model_name]['accuracy_mean']

	# sort dict alphabetically by key
	metrics = dict(sorted(metrics.items(), key=lambda item: item[0]))

	# Convert the dictionary to a list of dictionaries with metric names as keys
	table_data = [{'Model': source, **vals} for source, vals in metrics.items()]
	# print a nicely formatted table
	logger.info('\n' + tabulate(table_data, headers='keys', tablefmt='grid') + '\n')
	# logger.info('\n' + tabulate(table_data, headers='keys', tablefmt='tsv') + '\n')

	# now, take the last few entries and draw the bboxes
	test_model = f"ofa_huge_{goal}"
	for i, true_entry_idx in enumerate(true_entries['entry_idx'][-12:]):
		# if true_entry_idx != 'final_9-8-u8':
		# 	continue
		i = -i - 1      # access index
		pred_entry = all_model_predictions[test_model][true_entry_idx]
		_ap = _calculate_ap_score(torch.tensor(pred_entries[test_model]['bbox'][i:]), torch.tensor(true_entries['bbox'][i:]))
		pred_entry['metrics'] = {
			'ap': compute_score({'_score_sum': _ap.sum(), '_score_cnt': _ap.size(0)}),
			'accuracy': metrics_accuracy([pred_entry['index']], true_entries['index'][i:])['mean'] if 'index' in pred_entry else 'NA',
			'distance': metrics_distance([pred_entry['coords']], true_entries['coords'][i:])['mean'],
		}

		# short string with all metrics for the model
		metrics_str = f"ap: {metrics[test_model]['ap_mean']}, ac: {metrics[test_model]['accuracy_mean'] if goal == 'source' else 'NA'}, dis_mean: {metrics[test_model]['distance_mean']}, dis_med: {metrics[test_model]['distance_median']}"

		check_predictions_in_image(f"model: {test_model} ({metrics_str})", pred_entry, entries[true_entry_idx])


def print_table(headers, data):
	# Print headers
	header_row = "\t".join(headers)
	print(header_row)

	# Print data rows
	for row in data:
		data_row = "\t".join(map(str, row))
		print(data_row)


def metrics_accuracy(pred_list, true_list) -> Dict[str, float]:
	accuracy = [1 if x == y else 0 for x, y in zip(pred_list, true_list)]
	return _calc_metrics(accuracy)


def metrics_distance(pred_list, true_list) -> Dict[str, float]:
	distance = [calculate_bw_block_distance(x, y) for x, y in zip(pred_list, true_list)]
	return _calc_metrics(distance)


def _calc_metrics(comparison_list) -> Dict[str, float]:
	# calculate mean, median, std
	return {
		'mean': np.round(np.mean(comparison_list), 4),
		'median': np.round(np.median(comparison_list), 4),
		'std': np.round(np.std(comparison_list), 4),
		'count': len(comparison_list)
	}


def check_predictions_in_image(title, pred_entry, true_entry):
	# check an image for the last prediction only
	image_file = true_entry['image'] + '.png'  # 'final_2-0.png'
	image = Image.open(os.path.join(data_dir, image_file))
	image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGRA)
	# image.putalpha(128)

	logger.debug(f"blocks bboxes: {true_entry['blocksBBoxes']}")
	logger.debug(f"true bbox: {true_entry['true_bbox']}")
	logger.debug(f"true index: {true_entry['true_index']}\n")
	logger.debug(f"pred bbox: {pred_entry['bbox']}")
	if 'index' in pred_entry:
		logger.debug(f"pred index: {pred_entry['index']}")

	logger.debug(f"blocks: {true_entry['blocks']}")
	logger.debug(f"true block coords: {true_entry['true_coords']}")
	logger.debug(f"pred block coords: {pred_entry['coords']}")
	logger.debug(f"distance: {calculate_bw_block_distance(true_entry['true_coords'], pred_entry['coords'])}")

	image = add_text_to_image(image, title, font_scale=0.6)

	# draw the background
	image = draw_bboxes_in_image(image, discrete_table['cells'], (0, 255, 0, 255), 1, 0.2)

	# refactor this code to use a list of tuples with (bbox list, text, colour)
	bbox_drawings = [
		# draw the blocks
		(' blocks bounding boxes (Unity)', true_entry['blocksBBoxes'], (255, 0, 0, 255), 2, 0.5),
		# draw the true block
		(f" true block {true_entry['true_coords']}", [true_entry['true_bbox']], (100, 255, 200, 255), 2),
		# draw the pred block
		(f" pred block {pred_entry['coords']}", [pred_entry['bbox']], (0, 150, 255, 255), 2),
		# find the selected cell for the predicted block
		(' selected cell for predicted block', [find_closest_discrete_cell_to_point(_calculate_midpoint(pred_entry['bbox'], True))['bbox']], (0, 255, 255, 255), 2),
		# find the gold cell for the true block
		(f" gold cell for true block {true_entry['true_coords']}", [find_closest_discrete_cell_to_bw_coords(true_entry['true_coords'])['bbox']], (255, 0, 150, 255), 2),
		# find the selected cell for the true block
		(' selected cell for true block', [find_closest_discrete_cell_to_point(_calculate_midpoint(true_entry['true_bbox'], True))['bbox']], (0, 255, 150, 255), 1),
	]

	# draw everything from that list
	for item in bbox_drawings:
		try:
			image = draw_bboxes_in_image(image, item[1], *item[2:])
			image = add_text_to_image(image, item[0], colour=item[2])
		except Exception as ex:
			logger.warning(f"Error drawing bboxes: {item}")
			raise ex

	# now also print some metrics
	image = add_text_to_image(image, f" -> {pred_entry['metrics']}")

	out_path = os.path.join(
		'/users/fjc3/sharedscratch/output_img/',
		# the second replace makes sure that the utterance matches with entry_idx
		'pred-' + image_file.replace(true_entry['entryIdx'], true_entry['entry_idx']))
	logging.info(f"Saving image to '{out_path}'")

	# image.save(os.path.join('/users/fjc3/sharedscratch/output_img/', 'out_bboxes-' + image_file))
	cv2.imwrite(out_path, image)


def draw_bboxes_in_image(_img, bboxes: List[List[float]], colour: Tuple[int, int, int, int] = (255, 0, 0, 255), thickness = 2, alpha=1.0):
	overlay = _img.copy() if alpha < 1.0 else _img

	for bbox in bboxes:
		cv2.rectangle(
			overlay,
			(int(bbox[2]), int(bbox[3])),
			(int(bbox[0]), int(bbox[1])),
			colour, thickness)  # Draw filled rectangle on overlay

	if alpha < 1.0:
		cv2.addWeighted(overlay, alpha, _img, 1 - alpha, 0, _img)  # Blend overlay with the original image

	return _img


def add_text_to_image(image, text, font_scale=1.0, colour=(0, 0, 0, 255)):
	"""
	Add text at the end of an image, increasing the image's height accordingly.

	Parameters:
	- image: The input image (NumPy array).
	- text: The text to be added.
	- font: Font type (e.g., cv2.FONT_HERSHEY_SIMPLEX).
	- font_scale: Font scale factor.
	- color: Tuple (B, G, R) specifying the text color (0-255).

	Returns:
	- The image with the added text at the end, and its height increased accordingly.
	"""
	font = cv2.FONT_HERSHEY_SIMPLEX
	padding = 7
	if len(colour) == 3:
		colour = (colour[0], colour[1], colour[2], 255) # add alpha if not there

	# Get the dimensions of the image
	height, width = image.shape[:2]

	# Calculate the size of the text box
	text_size = cv2.getTextSize(text, font, font_scale, 1)[0]
	text_height = text_size[1]

	# Create a new image with increased height to accommodate the text
	new_height = height + text_height + (padding * 2)
	new_image = np.zeros((new_height, width, image.shape[2]), dtype=np.uint8)
	# make array be white colour
	new_image[:] = (255, 255, 255, 255)
	new_image[:height, :] = image  # Copy the original image to the top of the new image

	# Add the text to the new image at the bottom
	text_position = (padding, height + text_height + padding)  # Adjust position for padding
	cv2.putText(new_image, text, text_position, font, font_scale, colour, thickness=2)

	return new_image


def baseline_random_block(entry: dict) -> dict:
	pred_index = random.randint(0, len(entry['blocks']) - 1)

	return {
		'index': pred_index,
		'bbox': entry['blocksBBoxes'][pred_index],
		'coords': entry['blocks'][pred_index]
	}


def baseline_random_bbox(entry: dict) -> dict:
	pred_index = random.randint(0, len(entry['blocks']) - 1)

	return {
		'index': pred_index,
		'bbox': entry['blocksBBoxes'][pred_index],
		'coords': bbox_to_bw_coords(entry['blocksBBoxes'][pred_index])
	}


def baseline_random_position(entry: dict) -> dict:
	pass


def baseline_centre_position(entry: dict) -> dict:
	centre = [0, 0.0762, 0]
	return {
		'bbox': find_closest_discrete_cell_to_bw_coords([0, 0.0762, 0])['bbox'],
		'coords': centre,
	}


def baseline_oracle_block(entry: dict) -> dict:
	return {
		'index': entry['sourceIndex'] if goal == 'source' else -1,
		'bbox': entry['blocksBBoxes'][entry['sourceIndex']] if goal == 'source' else entry['targetBBox'],
		'coords': entry['blocks'][entry['sourceIndex']] if goal == 'source' else entry['targetLocation']
	}


def baseline_oracle_bbox(entry: dict) -> dict:
	return {
		'index': entry['sourceIndex'] if goal == 'source' else -1,
		'bbox': entry['blocksBBoxes'][entry['sourceIndex']] if goal == 'source' else entry['targetBBox'],
		'coords': bbox_to_bw_coords(entry['blocksBBoxes'][entry['sourceIndex']] if goal == 'source' else entry['targetBBox'])
	}


def calculate_bw_distance(bw_coords1, bw_coords2) -> float:
	x1, _, y1 = bw_coords1
	x2, _, y2 = bw_coords2
	x1 += 1
	x2 += 1
	y1 += 1
	y2 += 1
	return calculate_distance((x1, y1), (x2, y2))


block_length = 0.0762 * 2


def calculate_bw_block_distance(bw_block1, bw_block2) -> float:
	dist = calculate_bw_distance(bw_block1, bw_block2)
	return dist / block_length


def calculate_distance(coords1, coords2) -> float:
	x1, y1 = coords1
	x2, y2 = coords2
	return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

# def baseline_oracle_source(blocks: list) -> int:


def calculate_distance_to_gold(pred_box: List[float], gold_bw_coords: List[float]) -> float:
	# print()
	# print(pred_box)
	# print(gold_bw_coords)
	# first, we need to convert the pred_box to a bw coord
	pred_bw_coords = bbox_to_bw_coords(pred_box)
	# print(f"pred_box: {pred_box}, pred_bw_coords: {pred_bw_coords}")
	# now we calculate the distance between pred and gold coords
	# we use the euclidean distance formula
	# sqrt((x2 - x1)^2 + (y2 - y1)^2)
	x1, _, y1 = gold_bw_coords
	x2, _, y2 = pred_bw_coords
	# print(f"gold_bw_coords: {gold_bw_coords}, pred_bw_coords: {pred_bw_coords}")
	distance = calculate_bw_distance(gold_bw_coords, pred_bw_coords)
	# print(f"distance: {distance}")
	block_distance = distance / block_length
	# print(f"block_distance: {block_distance}")
	return block_distance


def find_closest_discrete_cell_to_point(point: List[float]) -> Dict[str, List[float]]:
	distances = []
	for i, cell in enumerate(discrete_table['cells']):
		cell_centre = [(cell[0] + cell[2]) / 2, (cell[1] + cell[3]) / 2] # x, y
		distances.append((i, calculate_distance(point, cell_centre)))

	# now sort by distance
	distances.sort(key=lambda x: x[1])
	# and return the index of the closest cell
	return {
		'bbox': discrete_table['cells'][distances[0][0]],
		'coords': discrete_table['coords'][distances[0][0]]
	}


def find_closest_discrete_cell_to_bw_coords(bw_coords: List[float]) -> Dict[str, List[float]]:
	# calculate the shortest distance between the bw_coords and the discrete_table
	# then return the bbox of that cell
	distances = []
	for i, cell in enumerate(discrete_table['coords']):
		distances.append((i, calculate_bw_distance(bw_coords, cell)))

	# now sort by distance
	distances.sort(key=lambda x: x[1])
	# and return the index of the closest cell
	return {
		'bbox': discrete_table['cells'][distances[0][0]],
		'coords': discrete_table['coords'][distances[0][0]]
	}


def _calculate_midpoint(bbox: List[float], bw_adjustment=False) -> List[float]:
	midpoint = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2] # x, y
	if bw_adjustment:       # bring it slightly down so it's closer to bw coords
		midpoint = [midpoint[0], int(midpoint[1] + ((bbox[3] - bbox[1]) / 3.15))]
	return midpoint


def bbox_to_bw_coords(bbox: List[float]) -> List[float]:
	# we take a bounding box and calculate the ap score for each cell in discrete_table
	# then we return the index of the highest ap score
	# UPDATE: this method does not work well as it usually needs a small threshold, and the AP result is (almost always either 0 or 1)
	# index = pred_box_to_index(discrete_table['cells'], bbox, thresh=0.2)
	# print(index)
	# return discrete_table['coords'][index]

	# instead, calculate the center of the bbox and return the closest cell
	centre_point = _calculate_midpoint(bbox, bw_adjustment=True) # x, y

	# find the cell closest to the point
	return find_closest_discrete_cell_to_point(centre_point)['coords']


# use same function as in OFA/tasks/mm_tasks/refcoco.py
def _calculate_ap_score(hyps, refs, thresh=0.5):
	# import tasks.mm_tasks.refcoco as refcoco
	# return refcoco._calculate_ap_score(hyps, refs, thresh)
	interacts = torch.cat(
		[torch.where(hyps[:, :2] < refs[:, :2], refs[:, :2], hyps[:, :2]),
		 torch.where(hyps[:, 2:] < refs[:, 2:], hyps[:, 2:], refs[:, 2:])],
		dim=1
	)
	area_predictions = (hyps[:, 2] - hyps[:, 0]) * (hyps[:, 3] - hyps[:, 1])
	area_targets = (refs[:, 2] - refs[:, 0]) * (refs[:, 3] - refs[:, 1])
	interacts_w = interacts[:, 2] - interacts[:, 0]
	interacts_h = interacts[:, 3] - interacts[:, 1]
	area_interacts = interacts_w * interacts_h
	ious = area_interacts / (area_predictions + area_targets - area_interacts + 1e-6)
	return ((ious >= thresh) & (interacts_w > 0) & (interacts_h > 0)).float()


def compute_score(meters):
	score = meters["_score_sum"] / meters["_score_cnt"]
	score = score if isinstance(score, float) else score.item()
	return round(score, 4)


def calc_ap(gold_bboxes: List[List[float]], pred_bbox: List[float], thresh=0.5) -> List[float]:
	# convert List of List to tensor
	gold_bboxes = torch.tensor(gold_bboxes)

	# convert single List of floats to tensor
	pred_bbox = torch.tensor(pred_bbox)
	# now repeat the value for each gold bbox
	pred_bboxes = pred_bbox.repeat(len(gold_bboxes), 1)

	# at this point, gold_bboxes and pred_bboxes should be the same shape
	assert gold_bboxes.shape == pred_bboxes.shape

	_ap_scores = _calculate_ap_score(pred_bboxes, gold_bboxes, thresh)
	# print(_ap_scores)
	# not needed
	# ap_score = compute_score({'_score_sum': _ap_scores.sum(), '_score_cnt': _ap_scores.size(0)})
	# print(ap_score)

	# convert tensor to List of floats
	return _ap_scores.tolist()


def pred_bbox_to_index(gold_bboxes: List[List[float]], pred_bbox: List[float], thresh=0.5) -> int:
	block_ap_scores = calc_ap(gold_bboxes, pred_bbox, thresh)
	# convert list to list of tuples with index
	block_ap_scores = list(enumerate(block_ap_scores))
	# print(block_ap_scores)
	# now sort by score
	# print(block_ap_scores)
	block_ap_scores.sort(key=lambda x: x[1], reverse=True)
	# todo: I should probs do this better
	if block_ap_scores[0][1] == 0:
		# select the closest block
		midpoint = _calculate_midpoint(pred_bbox)
		# find the closest gold block to this midpoint
		distances = []
		for i, cell in enumerate(gold_bboxes):
			cell_centre = [(cell[0] + cell[2]) / 2, (cell[1] + cell[3]) / 2]  # x, y
			distances.append((i, calculate_distance(midpoint, cell_centre)))

		# now sort by distance
		distances.sort(key=lambda x: x[1])
		# if distances[0][1] > 5.0:
		# 	logger.warning('no block found??')
		# 	logger.debug(f"distances: {distances}")

		return distances[0][0]

	# assert block_ap_scores[0][1] > 0
	# print(block_ap_scores)
	# now return the index of the highest score
	return block_ap_scores[0][0]


# data_dir = f"/Users/javiercg/workspace/Unity-BW-Datagen/generated_data/576p/"
# data_dir = f"/users/fjc3/sharedscratch/generated_data/576p/"
# results_dir = '/users/fjc3/sharedscratch/results/'

# load a json file called discrete10x10.json inside the data_dir
_discrete_data = json.load(open(os.path.join(os.environ['UNITY_DATA_DIR'], '576p', 'discrete70x70.json'), 'r'))
discrete_table = {
	'cells': [x['cellBBox'] for x in _discrete_data['cells']],
	'coords': [x['cellCentreBWCoords'] for x in _discrete_data['cells']]
}


if __name__ == "__main__":
	logging.info('start!')

	# take arguments results_path and data
	import argparse
	argparser = argparse.ArgumentParser()
	argparser.add_argument('--results_path', type=str, help='path to results folder, e.g., model_nmame/best_score_checkpoint ')
	argparser.add_argument('--data', type=str, help='data to evaluate with, eg., bwrel_unity_576p/bwrel_unity_576p_source_test.tsv')
	args = argparser.parse_args()
	logging.debug(args)

	# C:\jchiyah\workspace\NyroDockerNoetic\generated_data
	_resolution = args.data.split('p')[0].split('-')[-1]

	data_dir = f"/users/fjc3/sharedscratch/generated_data/{_resolution}p/"
	# results_dir = '/users/fjc3/sharedscratch/results/'

	# load a json file called discrete10x10.json inside the data_dir
	_discrete_data = json.load(open(os.path.join(data_dir, 'discret70x70.json'), 'r'))
	discrete_table = {
		'cells': [x['cellBBox'] for x in _discrete_data['cells']],
		'coords': [x['cellCentreBWCoords'] for x in _discrete_data['cells']]
	}

	goal = 'target' if 'target' in args.results_path else 'source'

	# input(calculate_bw_distance([0.45801, 0.0762, -0.773059], [0.6588888, 0.0762, -0.50111115]))
	# input(calculate_bw_distance([0.45801, 0.0762, -0.773059], [-0.7, 0.0762, -0.66]))

	predictions = load_predictions(args.results_path, args.data)
	evaluate(
		args.data, {f"ofa_huge_{goal}": predictions},
		[baseline_oracle_block, baseline_random_block, baseline_centre_position, baseline_random_bbox, baseline_oracle_bbox])
	# evaluate_results(args.results_path, args.data)
