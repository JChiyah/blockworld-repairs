
import os
import csv
import copy
import json
import warnings
from typing import *
from datetime import datetime

import pandas as pd
import pydantic
import numpy as np
from tabulate import tabulate

# todo: fix this
# from . import PROJECT_ROOT
# sys.path.append(PROJECT_ROOT + '/../../block-world-research/bw-correction-dialogues')
# import data_utils
# from data_utils import PredictionItem

from . import data, parsing, get_logger
from .data import PredictionItem
from .utils import common_utils

logger = get_logger()


def evaluate_predictions222(predictions: List[PredictionItem], print_table: bool = True) -> dict:
	results = [AnnotationAnalysis._analysis_row('all', predictions)]

	header_rows = [
		['split', 'entries', 'time_per_entry', '', 'source', 'source_block_distance', '', '',
		 '', 'target_block_distance', '', '', '', ''],
		['', '', '', '', 'accuracy', 'median', 'mean', 'std', '', 'median', 'mean', 'std',
		 'accuracy_r=1', 'accuracy_r=2']
	]

	# print(tabulate_cell_merger.tabulate(header_rows + [list(x.values()) for x in table_rows] + header_rows[::-1], colspan, {})) #, headers='keys', tablefmt="grid"))
	if print_table:
		if len(set([x.type for x in predictions])) > 1:
			# multiple types!
			for _type in set([x.type for x in predictions]):
				results.append(AnnotationAnalysis._analysis_row(
					str(_type), [x for x in predictions if x.type == _type]))

		print(tabulate(header_rows + [list(x.values()) for x in results], tablefmt="github"))

	return results[0]

_DEFAULT_VALUE = -1.0


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class EvaluationItemBase(pydantic.BaseModel):
	predictions: List[PredictionItem]
	_n_unique_entries: int = -1
	block_distance_median: Optional[float] = _DEFAULT_VALUE
	block_distance_mean: Optional[float] = _DEFAULT_VALUE
	block_distance_std: Optional[float] = _DEFAULT_VALUE
	iou_score_mean: Optional[float] = _DEFAULT_VALUE
	_include_invalid_in_metrics: bool = True   # if False, it will ignore invalid evaluations, else it will use the default baseline values
	repair_precision: float = _DEFAULT_VALUE
	repair_recall: float = _DEFAULT_VALUE
	repair_macro_f1: float = _DEFAULT_VALUE

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		block_distances = self._extract_metric('block_distance')
		self.block_distance_median = np.median(block_distances) #  if len(block_distances) > 0 else np.NaN
		self.block_distance_mean = np.mean(block_distances)
		self.block_distance_std = np.std(block_distances)
		self.iou_score_mean = np.mean(self._extract_metric('iou_score'))

		true_repairs = self._extract_metric('true_repair')
		pred_repairs = self._extract_metric('pred_repair')
		self.repair_precision = precision_score(true_repairs, pred_repairs, zero_division=0)
		self.repair_recall = recall_score(true_repairs, pred_repairs, zero_division=0)
		# f1 = f1_score(y_true, y_pred)
		# confusion_matrix(y_true, y_pred)
		self.repair_macro_f1 = f1_score(true_repairs, pred_repairs, average='macro')
		self._n_unique_entries = len(set([x.entry_idx for x in self.predictions]))

	@property
	def n_predictions(self) -> int:
		return len(self.predictions)

	@property
	def n_unique_entries(self) -> int:
		return self._n_unique_entries

	def _extract_metric(self, metric_name) -> list:
		results = [
			getattr(getattr(x, self.goal_str), metric_name)
			for x in self.predictions if getattr(getattr(x, self.goal_str), 'is_valid')]
		if self._include_invalid_in_metrics:
			results.extend([
				getattr(getattr(x, self.goal_str), metric_name)
				for x in self.predictions if not getattr(getattr(x, self.goal_str), 'is_valid')])
		return results

	@property
	def goal_str(self) -> str:
		return 'source' if isinstance(self, SourceEvaluation) else 'target'


class SourceEvaluation(EvaluationItemBase):
	_accuracy: float = _DEFAULT_VALUE

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self._accuracy = np.mean(self._extract_metric('accuracy'))

	@property
	def accuracy(self) -> float:
		return self._accuracy

	@property
	def n_valid_predictions(self) -> int:
		return len([x for x in self.predictions if x.source.is_valid])

	@property
	def n_invalid_predictions(self) -> int:
		return len([x for x in self.predictions if not x.source.is_valid])


class TargetEvaluation(EvaluationItemBase):
	_radius_1_accuracy: float = _DEFAULT_VALUE
	_radius_2_accuracy: float = _DEFAULT_VALUE
	_radius_3_accuracy: float = _DEFAULT_VALUE
	HUMAN_DIST_MEAN: float = 3.22

	def __init__(self, **kwargs):
		# kwargs['block_distances'] = [x.target.block_distance for x in kwargs['predictions'] if x.target.is_valid]
		# kwargs['iou_scores'] = [x.target.iou_score for x in kwargs['predictions'] if x.target.is_valid]
		super().__init__(**kwargs)
		# todo: refactor as in sourceEvaluation, calling _extract_metric
		self._radius_1_accuracy = np.mean([
			x.target.radius_1_accuracy for x in self.predictions if x.target.is_valid])
		self._radius_2_accuracy = np.mean([
			x.target.radius_2_accuracy for x in self.predictions if x.target.is_valid])
		self._radius_3_accuracy = np.mean([
			x.target.get_radius_accuracy(3) for x in self.predictions if x.target.is_valid])

	@property
	def radius_1_accuracy(self) -> float:
		return self._radius_1_accuracy

	@property
	def radius_2_accuracy(self) -> float:
		return self._radius_2_accuracy

	@property
	def radius_3_accuracy(self) -> float:
		return self._radius_3_accuracy

	@property
	def radius_1_human_accuracy(self) -> float:
		return np.mean([x.target.get_radius_accuracy(
			1, distance_measure=self.HUMAN_DIST_MEAN) for x in self.predictions if x.target.is_valid])

	@property
	def radius_2_human_accuracy(self) -> float:
		return np.mean([x.target.get_radius_accuracy(
			2, distance_measure=self.HUMAN_DIST_MEAN) for x in self.predictions if x.target.is_valid])

	@property
	def radius_3_human_accuracy(self) -> float:
		return np.mean([x.target.get_radius_accuracy(
			3, distance_measure=self.HUMAN_DIST_MEAN) for x in self.predictions if x.target.is_valid])

	@property
	def n_valid_predictions(self) -> int:
		return len([x for x in self.predictions if x.target.is_valid])

	@property
	def n_invalid_predictions(self) -> int:
		return len([x for x in self.predictions if not x.target.is_valid])


class EvaluationOutput(pydantic.BaseModel):
	predictions: List[PredictionItem]
	name: str = 'all'
	_source: SourceEvaluation = None
	_target: TargetEvaluation = None
	_n_entries: int = -1
	_n_unique_entries: int = -1

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.predictions = copy.deepcopy(self.predictions)
		with warnings.catch_warnings():
			warnings.simplefilter("ignore", category=RuntimeWarning)
			self._source = SourceEvaluation(
				predictions=[x for x in self.predictions if x.has_source])
			self._target = TargetEvaluation(
				predictions=[x for x in self.predictions if x.has_target])
			self._n_entries = len(self.predictions)
			self._n_unique_entries = len(set([x.entry_idx for x in self.predictions]))

	@property
	def n_entries(self) -> int:
		return self._n_entries

	@property
	def n_unique_entries(self) -> int:
		return self._n_unique_entries

	@property
	def source(self) -> SourceEvaluation:
		return self._source

	@property
	def target(self) -> TargetEvaluation:
		return self._target

	def get_entry_types(self) -> List[data.BWEntryType]:
		type_set = list(set([x.type for x in self.predictions]))
		return sorted(type_set, key=lambda x: data.BWEntryType.custom_order(x))

	def as_dict_for_table(self, limit_decimals: bool = True) -> dict:
		# includes separators
		return {
			'split': self.name,
			'n_predictions': self.n_entries,
			'n_unique_entries': self.n_unique_entries,
			# '|': '',
			'source_predictions': self.source.n_valid_predictions,
			'source_invalid_preds': self.source.n_invalid_predictions,
			'source_accuracy': f"{self.source.accuracy:.2f}"
				if limit_decimals else self.source.accuracy,
			'source_block_distance_median': f"{self.source.block_distance_median:.2f}"
				if limit_decimals else self.source.block_distance_median,
			'source_block_distance_mean': f"{self.source.block_distance_mean:.2f}"
				if limit_decimals else self.source.block_distance_mean,
			'source_block_distance_std': f"{self.source.block_distance_std:.2f}"
				if limit_decimals else self.source.block_distance_std,
			'source_iou_mean': f"{self.source.iou_score_mean:.2f}"
				if limit_decimals else self.source.iou_score_mean,
			'source_repair_precision': f"{self.source.repair_precision:.2f}"
				if limit_decimals else self.source.repair_precision,
			'source_repair_recall': f"{self.source.repair_recall:.2f}"
				if limit_decimals else self.source.repair_recall,
			'source_repair_macro_f1': f"{self.source.repair_macro_f1:.2f}"
				if limit_decimals else self.source.repair_macro_f1,
			# '||': '',
			'target_predictions': self.target.n_valid_predictions,
			'target_invalid_preds': self.target.n_invalid_predictions,
			'target_block_distance_median': f"{self.target.block_distance_median:.2f}"
				if limit_decimals else self.target.block_distance_median,
			'target_block_distance_mean': f"{self.target.block_distance_mean:.2f}"
				if limit_decimals else self.target.block_distance_mean,
			'target_block_distance_std': f"{self.target.block_distance_std:.2f}"
				if limit_decimals else self.target.block_distance_std,
			'target_radius_1_accuracy': f"{self.target.radius_1_accuracy:.2f}"
				if limit_decimals else self.target.radius_1_accuracy,
			'target_radius_2_accuracy': f"{self.target.radius_2_accuracy:.2f}"
				if limit_decimals else self.target.radius_2_accuracy,
			'target_radius_3_accuracy': f"{self.target.radius_3_accuracy:.2f}"
				if limit_decimals else self.target.radius_3_accuracy,
			'target_radius_1_human_accuracy': f"{self.target.radius_1_human_accuracy:.2f}"
				if limit_decimals else self.target.radius_1_human_accuracy,
			'target_radius_2_human_accuracy': f"{self.target.radius_2_human_accuracy:.2f}"
				if limit_decimals else self.target.radius_2_human_accuracy,
			'target_radius_3_human_accuracy': f"{self.target.radius_3_human_accuracy:.2f}"
				if limit_decimals else self.target.radius_3_human_accuracy,
			'target_iou_mean': f"{self.target.iou_score_mean:.2f}"
				if limit_decimals else self.target.iou_score_mean,
		}

	def get_table_column_names(self) -> List[str]:
		return [x for x in self.as_dict_for_table().keys() if '|' not in x]

	def evaluate_subset(self, name: str, filter_func) -> 'EvaluationOutput':
		return self.__class__(name=name,
			predictions=[x for x in self.predictions if filter_func(x)], extra_info=getattr(self, 'extra_info', None))


class EvaluationOutputWithModelInfo(EvaluationOutput):
	name: str = 'all'                # replaced by default in init
	extra_info: dict

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.name = get_short_model_name(self.extra_info['base_model']) if self.name == 'all' else self.name

	def as_dict_for_table(self, limit_decimals: bool = True) -> dict:
		# includes separators
		datum = super().as_dict_for_table(limit_decimals)
		# we just repeat the data, adding a few more details
		return {
			'split': datum['split'],
			'load_model': self.extra_info['load_model'],
			'task': self.extra_info['task'],
			'test_dataset': self.extra_info['test_dataset'],
			'train_dataset': self.extra_info['train_dataset'],
			'turn_masking': self.extra_info['turn_masking'],
			'prompt_task': self.extra_info['prompt_task_instruction'],
			'n_predictions': datum['n_predictions'],
			# 'n_unique_entries': self.n_unique_entries,
			# '|': '',
			'source_predictions': datum['source_predictions'],
			'source_invalid_preds': datum['source_invalid_preds'],
			'source_accuracy': datum['source_accuracy'],
			'source_block_distance_median': datum['source_block_distance_median'],
			'source_block_distance_mean': datum['source_block_distance_mean'],
			'source_block_distance_std': datum['source_block_distance_std'],
			'source_iou_mean': datum['source_iou_mean'],
			# '||': '',
			'target_predictions': datum['target_predictions'],
			'target_invalid_preds': datum['target_invalid_preds'],
			'target_block_distance_median': datum['target_block_distance_median'],
			'target_block_distance_mean': datum['target_block_distance_mean'],
			'target_block_distance_std': datum['target_block_distance_std'],
			'target_radius_1_accuracy': datum['target_radius_1_accuracy'],
			'target_radius_2_accuracy': datum['target_radius_2_accuracy'],
			'target_radius_3_accuracy': datum['target_radius_3_accuracy'],
			'target_radius_1_human_accuracy': datum['target_radius_1_human_accuracy'],
			'target_radius_2_human_accuracy': datum['target_radius_2_human_accuracy'],
			'target_radius_3_human_accuracy': datum['target_radius_3_human_accuracy'],
			'target_iou_mean': datum['target_iou_mean'],
		}


class AnnotationEvaluationOutput(EvaluationOutput):
	predictions: List[Union[PredictionItem, data.AnnotatedPrediction]]
	_seconds_per_entry_mean: float
	_seconds_per_entry_std: float

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		if len(self.predictions) > 0 and isinstance(self.predictions[0], data.AnnotatedPrediction):
			self._seconds_per_entry_mean = np.mean([x.elapsed_seconds for x in self.predictions])
			self._seconds_per_entry_std = np.std([x.elapsed_seconds for x in self.predictions])
		else:
			self._seconds_per_entry_mean = 0.0
			self._seconds_per_entry_std = 0.0
			# should never get here
			# raise ValueError(f"Invalid prediction type {type(self.predictions[0])}")

	@property
	def seconds_per_entry_mean(self) -> float:
		return self._seconds_per_entry_mean

	@property
	def seconds_per_entry_std(self) -> float:
		return self._seconds_per_entry_std

	def as_dict_for_table(self, limit_decimals: bool = True) -> dict:
		# includes separators
		datum = super().as_dict_for_table(limit_decimals)
		# we just repeat the data, adding a few more details
		return {
			'split': self.name,
			'n_predictions': datum['n_predictions'],
			'n_unique_entries': datum['n_unique_entries'],
			'seconds_per_entry_mean': f"{self.seconds_per_entry_mean:.2f}"
			if limit_decimals else self.seconds_per_entry_mean,
			'seconds_per_entry_std': f"{self.seconds_per_entry_std:.2f}"
			if limit_decimals else self.seconds_per_entry_std,
			# '|': '',
			'source_predictions': datum['source_predictions'],
			'source_invalid_preds': datum['source_invalid_preds'],
			'source_accuracy': datum['source_accuracy'],
			'source_block_distance_median': datum['source_block_distance_median'],
			'source_block_distance_mean': datum['source_block_distance_mean'],
			'source_block_distance_std': datum['source_block_distance_std'],
			'source_iou_mean': datum['source_iou_mean'],
			# '||': '',
			'target_predictions': datum['target_predictions'],
			'target_invalid_preds': datum['target_invalid_preds'],
			'target_block_distance_median': datum['target_block_distance_median'],
			'target_block_distance_mean': datum['target_block_distance_mean'],
			'target_block_distance_std': datum['target_block_distance_std'],
			'target_radius_1_accuracy': datum['target_radius_1_accuracy'],
			'target_radius_2_accuracy': datum['target_radius_2_accuracy'],
			'target_radius_3_accuracy': datum['target_radius_3_accuracy'],
			'target_radius_1_human_accuracy': datum['target_radius_1_human_accuracy'],
			'target_radius_2_human_accuracy': datum['target_radius_2_human_accuracy'],
			'target_radius_3_human_accuracy': datum['target_radius_3_human_accuracy'],
			'target_iou_mean': datum['target_iou_mean'],
		}


def evaluate_predictions(
	predictions: List[PredictionItem], *, name: str = None, include_subsets_by_type: bool = True) \
		-> List[EvaluationOutput]:
	evals = [EvaluationOutput(predictions=predictions, name=name or 'all')]     # change 'all' for default in EvaluationOutput
	if include_subsets_by_type:
		evals.extend(evaluate_subsets_by_type(evals[0]))

	return evals


def evaluate_predictions_for_model_comparison(
	predictions: List[PredictionItem], *, include_subsets_by_type: bool = True, model_info: dict = None) \
		-> List[EvaluationOutput]:
	evals = [EvaluationOutputWithModelInfo(predictions=predictions, extra_info=model_info)]     # change 'all' for default in EvaluationOutput
	if include_subsets_by_type:
		evals.extend(evaluate_subsets_by_type(evals[0]))

	return evals


def evaluate_annotations(
	predictions: List[PredictionItem], *, name: str = 'human annotations', include_subsets_by_type: bool = True) \
		-> List[EvaluationOutput]:
	evals = [AnnotationEvaluationOutput(name=name, predictions=predictions)]
	# evals = [EvaluationOutput(predictions=predictions, name=name or 'all')]     # change 'all' for default in EvaluationOutput
	if include_subsets_by_type:
		evals.extend(evaluate_subsets_by_type(evals[0]))

	return evals


def evaluate_baseline(baseline: data.BaselineBase, entries: data.BWEntrySet) -> EvaluationOutput:
	predictions = [data.PredictionItem.from_baseline(x, baseline) for x in entries.test_entries]
	return evaluate_predictions(predictions, name=baseline.name)[0]


def get_training_evaluation_str(predictions: List[PredictionItem], total_entries=None) -> str:
	with warnings.catch_warnings():
		warnings.simplefilter("ignore", category=RuntimeWarning)
		evals = evaluate_predictions(predictions, include_subsets_by_type=False)[0].as_dict_for_table()

	txt = f"\t({evals['n_predictions']}{'/' + str(total_entries) if total_entries else ''}) "
	if evals['source_accuracy'] != 'nan':
		txt += f"source: accuracy={evals['source_accuracy']}, mean_dist={evals['source_block_distance_mean']}; "

	if evals['target_radius_2_accuracy'] != 'nan':
		txt += f"target: accuracy_r=2={evals['target_radius_2_accuracy']}, mean_dist={evals['target_block_distance_mean']}"

	return txt


def evaluate_subsets_by_type(evaluation_output: EvaluationOutput, *, join_corrections: bool = True, join_instructions: bool = True) -> List[EvaluationOutput]:
	evals = []
	entry_types = evaluation_output.get_entry_types()
	if join_corrections:
		# remove these types for now, deal with it later
		entry_types = [t for t in entry_types if t not in [data.BWEntryType.CORRECTION_SOURCE, data.BWEntryType.CORRECTION_TARGET]]

	for _type in entry_types:
		evals.append(evaluation_output.evaluate_subset(f" > {_type}", lambda x: x.type == _type))

	if join_instructions:
		# now add a row with instructions only
		evals.append(evaluation_output.evaluate_subset(' > instructions', lambda x: x.type in [data.BWEntryType.INSTRUCTION, data.BWEntryType.INSTRUCTION_OG]))

	if join_corrections:
		# now add a final row with corrections only
		evals.append(evaluation_output.evaluate_subset(' > corrections', lambda x: x.type in [data.BWEntryType.CORRECTION_SOURCE, data.BWEntryType.CORRECTION_TARGET]))

	return evals


class OriginalHumanBaseline(EvaluationOutput):
	predictions: List[PredictionItem] = None
	# basically a class holding the info from the original Bisk et al. paper

	def __init__(self, **kwargs):
		kwargs['predictions'] = []
		super().__init__(**kwargs)
		# self.

	def as_dict_for_table(self, limit_decimals=False, include_seconds: bool = True) -> dict:
		# includes separators
		return {
			'split': 'human performance (Bisk et al)',
			'n_predictions': 150,
			**({
				'seconds_per_entry_mean': 0,
				'seconds_per_entry_std': 0,
			} if include_seconds else {}),
			# 'n_invalid'
			# '|': '',
			'source_accuracy': 0.93,
			'source_block_distance_median': 0.0,
			'source_block_distance_mean': 0.3,
			'source_block_distance_std': '',
			'source_iou_mean': '',
			# '||': '',
			'target_block_distance_median': 0.37,
			'target_block_distance_mean': 1.39,
			'target_block_distance_std': '',
			'target_radius_1_accuracy': '',
			'target_radius_2_accuracy': '',
			'target_iou_mean': '',
		}


def print_evaluation_table(evaluations: List[EvaluationOutput], header_rows=None, remove_columns=None):
	# if header_rows is None:
	# automatically generate header rows
	headers = []
	subheaders = []
	final_columns = []
	for column in evaluations[0].as_dict_for_table().keys():
		if remove_columns and column in remove_columns:
			continue

		final_columns.append(column)

		if 'block_distance' in column:
			column = column.replace('block_distance', 'dist')
		if 'accuracy' in column:
			column = column.replace('accuracy', 'acc')

		if 'seconds_per_entry' in column:
			column = column.replace('seconds_per_entry', 'sec/entry')

		# if 'dist' in column:
		# 	headers.append(column.split('dist_')[0] + 'dist')
		# 	subheaders.append('_'.join(column.split('dist_')[1:]))
		if 'source' in column or 'target' in column:
			headers.append(column.split('_')[0])
			subheaders.append('_'.join(column.split('_')[1:]))
		else:
			if '|' in column:
				column = ''
			headers.append(column)
			subheaders.append('')

	header_rows = [headers, subheaders]

	output_rows = [x.as_dict_for_table() for x in evaluations]

	# print(evaluations[0].as_dict_for_table().keys())
	# print(final_columns)
	# print([key for key, value in evaluations[1].as_dict_for_table().items() if key in final_columns])

	print(tabulate(
		header_rows + [] +
		[[x[column] if column in x else '' for column in final_columns] for x in output_rows],
		tablefmt="github"))

	# df = pd.DataFrame([x.as_dict_for_table() for x in evaluations])
	# print(df)


def save_evaluations_to_csv(evaluations: List[EvaluationOutput], filename):
	"""
	Export a list of dictionaries to a CSV file.
	"""
	# Extract the headers from the keys of the first and last dict, in case we have humans and models
	headers = evaluations[0].get_table_column_names()
	headers.extend([x for x in evaluations[-1].get_table_column_names() if x not in headers])

	filename = filename.replace('.csv', '') + '.csv'
	try:
		with open(filename, mode='w', newline='') as file:
			writer = csv.DictWriter(file, fieldnames=headers)
			# Write the header
			writer.writeheader()

			# Write the rows
			for datum in evaluations:
				writer.writerow(datum.as_dict_for_table(limit_decimals=True))

		print(f"Saved evaluation results to {filename}")

	except IOError as e:
		print(f"An error occurred while writing to the file: {e}")


# short_model_names = {
# 	'HuggingFaceM4/idefics2-8b': 'idefics2-8b',
# 	'HuggingFaceM4/idefics2-8b-chatty': 'idefics2-8b-chatty',
# 	'liuhaotian/llava-v1.5-7b': 'llava-v1.5-7b',
# 	'gpt-4o': 'gpt-4o',
# }

# def get_short_model_name(model_name: str) -> str:
# 	# check if model short name is in model_name
# 	for short_name in short_model_names.values():
# 		if short_name in model_name:
# 			return short_name
# 	return short_model_names.get(model_name, model_name)


# def save_model_outputs(info: dict, responses: list, base_path: str = None, epoch: int = 0) -> str:
# 	# save response to a json file with a date prefix
# 	prefix = common_utils.get_datetime_prefix()
# 	if info['data']['train_dataset'] is None:
# 		info['data']['train_dataset'] = 'NA'

# 	output_file = (
# 		f"{info['run_name']}_{info['goal']}_e{epoch}_"
# 		f"test={common_utils.get_short_dataset_name(info['data']['test_dataset'])}_"
# 			f"{common_utils.get_short_dataset_name(info['data']['train_dataset'])}_{info['turn_masking']}"
# 			f"{'_debug' if info['debug'] and 'debug' not in info['run_name'] else ''}.json")
# 	if base_path:
# 		if not os.path.exists(base_path):
# 			os.makedirs(base_path)

# 		output_file = os.path.join(base_path, output_file)

# 	if info['model_task'] == 'zeroshot':
# 		info['train_dataset'] = ''

# 	with open(output_file, 'w') as out_f:
# 		json.dump({
# 			**info,
# 			'epoch': epoch,
# 			'wandb_ids': None,
# 			'timestamp': prefix,
# 			'responses': responses
# 		}, out_f, indent=4)

# 	# logger.info(f"Responses saved to {output_file}")
# 	return output_file


def update_model_output_file(filename: str, update_dict: dict):
	# Read the JSON file
	with open(filename, 'r') as in_file:
		datum = json.load(in_file)

	# Update the dictionary
	for key, new_value in update_dict.items():
		if key in datum:
			datum[key] = new_value

	# Write the updated datum back to the file
	with open(filename, 'w') as out_f:
		json.dump(datum, out_f, indent=4)


def read_model_output_file(file_path: str) -> dict:
	# read the json
	with open(file_path, 'r') as f:
		datum = json.load(f)
	return datum


def parse_model_predictions(datum: dict, all_entries) -> Tuple[List[data.Idefics2Prediction], dict]:
	# create two lists, just in case we want to evaluate a subset
	entries = []
	model_responses = []
	not_found = []

	for response_datum in datum['outputs']:
		try:
			entry = all_entries.get_entry_by_idx(response_datum['entry_idx'])
			entries.append(entry)
			model_responses.append(response_datum['model_output_string'])

		except ValueError:
			# not found! Skip this entry
			not_found.append(response_datum['entry_idx'])

	predictions = data.Idefics2Prediction.from_model_batch_output(
		entries, model_responses, datum['config']['goal'], new_parser=parsing.parse_generated_text)

	logger.info(f"Parsed model responses from file loaded: {len(entries)}" + (f" (entries not found in BWEntrySet: {len(not_found)})" if not_found else ""))

	return predictions, datum


def parse_and_read_model_output_file(file_path: str, all_entries: list = None) -> Tuple[List[data.Idefics2Prediction], dict]:
	outputs = read_model_output_file(file_path)
	if not all_entries:
		all_entries = data.get_bw_dataset(outputs['test_dataset'])
	return parse_model_predictions(outputs, all_entries)


def eval_output_file(file_path: str, all_entries):
	predictions, datum = parse_and_read_model_output_file(file_path, all_entries)

	evaluation_results = evaluate_predictions(predictions, include_subsets_by_type=True)
	print_evaluation_table(evaluation_results)

	# save the evaluation results in the file, for easier access
	with open(file_path, 'w') as out_f:
		json.dump({
			**datum,
			'evaluation_results': {x.name: x.as_dict_for_table() for x in evaluation_results}
		}, out_f, indent=4)
