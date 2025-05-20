# Background server for annotations
# deployed to DigitalOcean with https://medium.com/@hobegi/flask-app-on-digital-ocean-9b6db466079e


import os
import time
import json
import random
import shutil
import logging
import collections
import dataclasses
from typing import *

from flask import Flask, request, jsonify, make_response, send_from_directory, render_template

from .. import get_logger
logger = get_logger()


app = Flask(__name__)

BASE_PATH = '../../block-world-research/bw-correction-annotations/'
GENERATED_DATA = '/Users/javiercg/workspace/Unity-BW-Datagen/generated_data/576p/'
DIALOGUE_PATH = BASE_PATH + 'data'					# ../bw-correction-dialogues
OUTPUT_PATH = BASE_PATH + 'annotations'
MEDIA_PATH = BASE_PATH + 'media'

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

# only set up loggers once, so output does not repeat
# if not len(logger.handlers):
# 	# Create a StreamHandler and set the formatter
# 	ch = logging.StreamHandler()
# 	ch.setFormatter(formatter)
# 	logger.addHandler(ch)
#
# # Function to adjust the log level of a logger by name
# def set_logger_level(logger_name, log_level):
# 	logger_to_configure = logging.getLogger(logger_name)
# 	logger_to_configure.setLevel(log_level)

# hide messages from PngImagePlugin
# set_logger_level('PIL.PngImagePlugin', logging.WARNING)


@dataclasses.dataclass(frozen=True)
class EntrySelectionConfig:
	# how many annotations per entry to collect
	n_annotations_per_entry: int
	# number of training instructions (source + target)
	n_training_instructions: int
	# number of training corrections (source OR target), this is after n_training_instruction_annotations have been done
	n_training_corrections: int
	# select entries with different scenarios as last one if true, else no restriction
	alternate_scenarios: bool
	# require instructions before corrections
	instructions_before_corrections: bool
	# training entry pool
	training_entry_pool: str
	# general entry pool
	entry_pool: str

	@property
	def n_training_instructions_corrections(self) -> int:
		return self.n_training_instructions + self.n_training_corrections

	def __str__(self) -> str:
		return json.dumps(dataclasses.asdict(self), indent=4)

	def __post_init__(self):
		assert self.n_annotations_per_entry > 0
		assert 0 <= self.n_training_corrections < 3


entry_selection_config = EntrySelectionConfig(
	n_annotations_per_entry=2,
	n_training_instructions=2,
	n_training_corrections=2,
	alternate_scenarios=True,
	instructions_before_corrections=True,
	training_entry_pool='dialogues_training',
	entry_pool='dialogues_high_quality'
)
logger.info(f"Dialogue selection config: {entry_selection_config}")


def read_jsonl(filename: str) -> list:
	entries = []
	with open(filename, 'r') as file:
		for line in file:
			# Parse the JSON object in each line
			entries.append(json.loads(line))

	return entries


def get_dialogue_entries(
	split, remove_fully_annotated: bool = True, annotations_data_path: str = None) -> list:
	if annotations_data_path is None:
		annotations_data_path = DIALOGUE_PATH
	entries = read_jsonl(os.path.join(annotations_data_path, split + '.jsonl'))
	# if app.debug:
		# entries = entries[:10]

	logger.info(f"Loaded {len(entries)} dialogues from '{split}'")

	if remove_fully_annotated:
		_length = len(entries)
		entries = [x for x in entries if annotated_idxs[x['entry_idx']] < entry_selection_config.n_annotations_per_entry]
		logger.info(f"   {len(entries) - _length} fully annotated entries removed, {len(entries)} remaining")

	if len(entries) <= 5:
		logger.warning(f"Too few entries remaining to annotate in '{split}'")

	return entries


def get_annotation_files() -> list:
	# get all the files in the annotations directory
	return [x for x in os.listdir(OUTPUT_PATH) if x.endswith('.jsonl')]


def iterate_annotations(participant_ids: Optional[List[str]] = None) -> Iterable:
	# get all files
	for file in get_annotation_files():
		if participant_ids and file.replace('.jsonl', '') not in participant_ids:
			print(f"Skipping {file}")
			continue

		# read files with their annotations
		entries = read_jsonl(os.path.join(OUTPUT_PATH, file))
		for entry in entries:
			if 'entry_idx' not in entry:
				entry['entry_idx'] = entry['annotations']['entry_idx']
			if 'timestamp_start' not in entry['annotations']:
				entry['annotations']['timestamp_start'] = entry['timestamp_start']
				entry['annotations']['timestamp_end'] = entry['timestamp_end']

			yield entry


def get_annotated_idxs(participant_ids: Optional[List[str]] = None) -> collections.Counter:
	annotated_idxs = collections.Counter()
	for entry in iterate_annotations(participant_ids):
		annotated_idxs[entry['entry_idx']] += 1

	logger.info(f"   {sum(annotated_idxs.values())} annotated entries found in '{OUTPUT_PATH}/'")
	return annotated_idxs


def get_unix_timestamp() -> float:
	return time.time()


def get_request_data() -> dict:
	return request.get_json() if request.data else request.args


def get_entry_by_idx(entry_idx: str, include_all: bool = False) -> dict:
	if include_all:
		pool = get_dialogue_entries(entry_selection_config.training_entry_pool, False) + get_dialogue_entries(entry_selection_config.entry_pool, False)

	else:
		pool = training_entry_pool + entry_pool

	# get entry from pools
	for entry in pool:
		if entry['entry_idx'] == entry_idx:
			return entry
	raise ValueError(f"Entry {entry_idx} not found in pools")


def iterate_dialogue_events(dialogue: dict, event_types: Optional[List[str]] = None) -> iter:
	for event in dialogue['events']:
		if event_types is None or event['event_type'] in event_types:
			# input(event)
			yield event


def get_annotated_entries_by_user(user_idx: str) -> List[dict]:
	try:
		entries = read_jsonl(f"{OUTPUT_PATH}/{user_idx}.jsonl")
		return entries
	except FileNotFoundError:
		return []


def get_simple_dialogue(dialogue: dict) -> list:
	dial = []
	for event in iterate_dialogue_events(dialogue, ['user_message', 'agent_message']):
		if 'utterance' not in event:
			# print(event)
			continue
		dial.append({
			'sender': event['event_type'].replace('_message', ''),
			'utterance': event['utterance']})

	return dial


def pseudo_randomly_select_dialogue_to_annotate(user_idx: str) -> dict:
	# get a list of entry_idx that have not been annotated by this user
	previous_user_entries = get_annotated_entries_by_user(user_idx)
	previous_idxs = [x['entry_idx'] for x in previous_user_entries]
	# input(training_entry_pool[0])

	if len(previous_user_entries) < entry_selection_config.n_training_instructions:
		# select from training pool
		_dialogue_pool = [
			x for x in training_entry_pool
			if x['entry_idx'] not in previous_idxs and x['entry_type'] == 'instruction']

	elif len(previous_user_entries) < entry_selection_config.n_training_instructions_corrections:
		# still training but now we should get corrections instead
		# let's also make sure that we show a different entry_type
		_prev_types = [x['entry_type'] for x in previous_user_entries]
		_dialogue_pool = [
			x for x in training_entry_pool
			if x['entry_type'] not in _prev_types and x['entry_idx'] not in previous_idxs]

	else:
		# default case, not training
		_last_entry = previous_user_entries[-1] \
			if entry_selection_config.alternate_scenarios else {'mturk_scenario_id': 'dummy_value'}

		_dialogue_pool = [
			x for x in entry_pool
			if x['entry_idx'] not in previous_idxs
			and x['mturk_scenario_id'] != _last_entry['mturk_scenario_id']]

		if entry_selection_config.instructions_before_corrections:
			# we need to filter out any corrections whose instructions have not been done yet
			# two options: filter out corrections until instructions are done OR filter out instructions if corrections on that dial are done
			# actually, best to avoid the max annotations is to filter out instruction if correction done
			_dialogue_pool = [
				x for x in _dialogue_pool
				# if the entry is correction, then filter out that dialogue idx
				if x['entry_type'] == 'instruction'
				# this also forces a single correction per dialogue, so cannot do both
				or x['dialogue_idx'] not in [y['dialogue_idx'] for y in previous_user_entries]
			]

	# at this point, all filters have been applied and dialogues with enough annotations have been removed

	# use weighted random so partially annotated entries are more likely to be sampled
	_weights = [(annotated_idxs[x['entry_idx']] * 20) + 1 for x in _dialogue_pool]
	selected_dialogue = random.choices(_dialogue_pool, weights=_weights, k=1)[0]

	return {
		**format_dialogue_for_annotations(selected_dialogue),
		'n_entries_annotated': len(previous_user_entries),
		'global_timestamp_start': previous_user_entries[0]['annotations']['timestamp_start'] if len(previous_user_entries) > 0 else get_unix_timestamp()
	}


def format_dialogue_for_annotations(dialogue: dict) -> dict:
	# we interleave the dialogue with annotations as it is easier to also save/user later
	dial = []
	for event in iterate_dialogue_events(dialogue, ['user_message', 'agent_message']):
		# input(json.dumps(event, indent=4))
		# continue
		if 'utterance' not in event:
			print(f"Skipping event {event['event_idx']} because it has no utterance:\n\t{event}")
			raise ValueError('Event has no utterance')

		if 'I am helping you arrange blocks' in event['utterance']:
			continue		# skip this message
		elif 'Which block should I move and where' in event['utterance']:
			event['utterance'] = 'Which block should I move and where?'

		dial_event = {
			'event_idx': event['event_idx'],
			'sender': event['event_type'].replace('_message', ''),
			'utterance': event['utterance'],
			**({'image': dialogue['origin_img'].replace('.png', '-576p.png')} if 'agent' in event['event_type'] else {}),
			**({'cand_source_xy': event['cand_source_xy']} if 'cand_source_xy' in event else {}),
			**({'cand_target_xy': event['cand_target_xy']} if 'cand_target_xy' in event else {}),
		}

		dial.append(dial_event)

	if dialogue['entry_type'] in ['instruction', 'correction_source']:
		dial[-1]['annotation_source_xy'] = []
	if dialogue['entry_type'] in ['instruction', 'correction_target']:
		dial[-1]['annotation_target_xy'] = []

	# input(dialogue['origin_img'])

	return {
		'entry_idx': dialogue['entry_idx'],
		'entry_type': dialogue['entry_type'],
		'dialogue_idx': dialogue['dialogue_idx'],
		'mturk_scenario_id': dialogue['mturk_scenario_id'],
		'dialogue_with_annotations': dial,
		'image': dialogue['origin_img'].replace('.png', '-576p.png'),
		'timestamp_start': get_unix_timestamp(),
	}


def save_annotations(all_data: dict):
	# we want to save the annotations and append them to the dialogues (so data is together)
	annotations = all_data['annotations']
	annotations['participant_idx'] = all_data['user_idx']
	annotations['event_idx'] = all_data['dialogue_with_annotations'][-1]['event_idx']
	annotations['image'] = all_data['image']
	annotations['timestamp_start'] = all_data['timestamp_start']
	annotations['timestamp_end'] = get_unix_timestamp()

	# find entry from pools
	entry = get_entry_by_idx(all_data['entry_idx'])
	entry['annotations'] = annotations

	# save annotations to a file by participant_idx
	with open(f"{OUTPUT_PATH}/{annotations['participant_idx']}.jsonl", 'a') as out_file:
		# now export to jsonl in new lines
		out_file.write(json.dumps(entry) + '\n')

	logger.info(f"Saved annotations for {annotations['participant_idx']} and {annotations['entry_idx']}")
	del entry['annotations']


@app.route('/get_dialogue_to_annotate', methods=['POST'])
def get_dialogue_to_annotate():
	# Ensure the request is JSON
	if not request.is_json:
		return jsonify({"error": "Request must be JSON"}), 400

	# Parse the JSON request
	request_data = get_request_data()

	if 'entry_idx' in request_data:
		# get the entry by idx, we are debugging
		response_data = {
			**format_dialogue_for_annotations(get_entry_by_idx(request_data['entry_idx'], include_all=True))
		}

	else:
		# default branch
		response_data = {
			**pseudo_randomly_select_dialogue_to_annotate(request_data['user_idx'])
		}

	# Return a JSON response
	return jsonify(response_data)


@app.route('/save_annotations', methods=['POST'])
def save_annotation():
	# Ensure the request is JSON
	if not request.is_json:
		return jsonify({"error": "Request must be JSON"}), 400

	# Parse the JSON request
	request_data = get_request_data()
	assert 'entry_idx' in request_data, "Missing entry_idx"
	assert 'annotations' in request_data, "Missing annotations"
	assert 'dialogue_with_annotations' in request_data, "Missing dialogue_with_annotations"

	save_annotations(request_data)

	# Return a JSON response
	return jsonify('200 - OK')


@app.route('/end_study', methods=['POST'])
def end_study():
	request_data = get_request_data()

	# do any cleaning

	# refresh annotations & entry pools
	global annotated_idxs, training_entry_pool, entry_pool
	annotated_idxs = get_annotated_idxs()
	training_entry_pool = get_dialogue_entries(entry_selection_config.training_entry_pool)
	entry_pool = get_dialogue_entries(entry_selection_config.entry_pool)

	# Return a JSON response
	return jsonify('200 - OK')


@app.route('/media/<string:file>')
def media(file):
	# input(file)
	if '.' not in file:
		file += '.png'      # default extension for images

	# only run this in debug
	if app.debug and 'example' not in file:
		# check if file exists in the media path
		if file not in os.listdir(MEDIA_PATH):
			print(f"File {file} not found in {MEDIA_PATH}")
			# copy file to directory

			shutil.copy(os.path.join(GENERATED_DATA, file), os.path.join(MEDIA_PATH, file))

	return send_from_directory(MEDIA_PATH, file)


FREE_ACCESS_ENDPOINTS = ['index', 'media', 'acknowledge', 'favicon.ico']

@app.before_request
def before_request():
	if request.method == 'OPTIONS':
		# allow cross-domain requests
		return jsonify(True)

	# Ensure the request is JSON
	if not request.is_json:
		# warning: not fully tested this code
		response = make_response({
			'error': 'request is not in JSON format'
		}, 401)

	request_data = get_request_data()
	extra = ''
	response = None

	if 'user_idx' not in request_data and request.endpoint not in FREE_ACCESS_ENDPOINTS \
		and request.base_url.split('/')[-1] not in FREE_ACCESS_ENDPOINTS:
		extra = ' - rejected because of missing user_idx'
		response = make_response({
			'error': 'Missing user_idx'
		}, 401)
		print(f"Rejected request to {request.endpoint} because of missing user_idx: ({extra}) (and endpoint not in {FREE_ACCESS_ENDPOINTS})")

	# if request.endpoint not in FREE_ACCESS_ENDPOINTS \
	# 	and request.base_url.split('/')[-1] not in FREE_ACCESS_ENDPOINTS:
	# 	uid = log_utils.get_uid(request_data['worker_id'], request_data['assignment_id'])
	# 	# identity = request_data['worker_id'] if 'worker_id' in request_data else ''
	# 	# identity += f"-{request_data['assignment_id'] if 'assignment_id' in request_data else ''}"
	# 	pymturk.logger.info(f"{uid}: Request={request.endpoint}{extra}")

	if response:
		return response


@app.after_request
def add_header(response):
	# this allows for cross-domain requests
	response.headers['Access-Control-Max-Age'] = '1000'
	response.headers['Access-Control-Allow-Origin'] = '*'
	response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
	response.headers['Access-Control-Allow-Headers'] = 'Origin, Content-Type'
	return response


@app.route('/')
def acknowledge():
	return jsonify('200 - OK')


# create index route
@app.route('/index')
def index():
	with open('index.html') as file:
		contents = file.read()
	return contents


if __name__ == '__main__':
	annotated_idxs = get_annotated_idxs()
	training_entry_pool = get_dialogue_entries(entry_selection_config.training_entry_pool)
	entry_pool = get_dialogue_entries(entry_selection_config.entry_pool)

	app.run(port=27027, debug=True)
