
# Parse generated text from models
import re
from typing import List, Optional

import pydantic


# todo: refactor this to use data_utils.py
IMAGE_SIZES = {
	'576p': (1024, 576),
}
DEFAULT_IMAGE_SIZE = IMAGE_SIZES['576p']


# check mturk_dc.py
REPAIRS_SOURCE = [
	'Is it this block?',
	'Do you mean this block?',
	'Can you confirm it is this block?'
]
REPAIRS_TARGET = [
	'Is moving the block here okay?',
	'Is this where you want to place it?',
	'I recall you wanting the block here, is this correct?'
]
REPAIRS = {
	'source': REPAIRS_SOURCE,
	'target': REPAIRS_TARGET
}

class ParsedOutput(pydantic.BaseModel):
	output_text: str
	bbox: List[float]
	normalised_bbox: List[float]
	repair_text: Optional[str] = None

	@property
	def contains_bbox(self) -> bool:
		return self.bbox is not None

	@property
	def contains_repair(self) -> bool:
		return self.repair_text is not None


class ParsingConfig(pydantic.BaseModel):
	# Add any configuration parameters that the parser needs
	pass


def parse_generated_text(text: str) -> ParsedOutput:
	bbox_text = _extract_bounding_box(text)
	normalised_bbox = _parse_bbox(bbox_text)
	# Scale bbox to image pixels (un-normalise) if in range 0-1
	bbox = scale_bbox_to_image(normalised_bbox) \
		if normalised_bbox and sum(normalised_bbox) <= 4 else normalised_bbox

	# Extract all text outside the bounding box
	outside_text = text.replace(bbox_text, '') if bbox_text else text

	repair_text = _extract_repair(outside_text)

	# Create and return a ParsedOutput object
	return ParsedOutput(
		output_text=text,
		bbox=bbox,
		normalised_bbox=normalised_bbox,
		# contains_repair=contains_repair,
		repair_text=repair_text
	)


def scale_bbox_to_image(bbox: List[float]) -> List[float]:
	# scale the bbox from the 0-1 to the actual image size
	return [bbox[0] * DEFAULT_IMAGE_SIZE[0], bbox[1] * DEFAULT_IMAGE_SIZE[1], bbox[2] * DEFAULT_IMAGE_SIZE[0], bbox[3] * DEFAULT_IMAGE_SIZE[1]]


def _parse_bbox(bbox_text: str) -> Optional[List[float]]:
	if bbox_text is None: return None

	try:
		bbox = [float(x.strip('[]').strip()) for x in bbox_text.split(",")]
		if len(bbox) == 4:
			return bbox

	except ValueError as ex:
		# logger.warning(f"Parsing issue in {entry.entry_idx}: '{ex}' | {generated_text=}; {parsed_text=}; {parsed_text.split(',')}")
		pass

	return None


def _extract_bounding_box(text: str) -> Optional[str]:
	match = re.search(r'\[(.*?)\]', text)
	return match.group(0) if match else None


def _extract_repair(text: str) -> Optional[str]:
	text = text.lower()
	# Check for the word 'repair' (case-insensitive)
	if 'repair' in text:
		return 'repair'

	# Check for REPAIR strings
	for repair in REPAIRS_SOURCE + REPAIRS_TARGET:
		if repair.lower() in text:
			return repair

	# If no matches found, return None
	return None


def _test_extract_bounding_box():
	test_strings = {
		"a [0.1,0.2, 0.3, 0.6] ": [0.1,0.2,0.3,0.6],
		"here it is: [0.111,0.223,0.323,0.644]": [0.111,0.223,0.323,0.644],
		"is it this one? [0.111, 0.223,0.323 ,0.644]": [0.111,0.223,0.323,0.644],
		"is it this one? [0.111]": None,
	}
	for text, expected_bbox in test_strings.items():
		bbox_text = _extract_bounding_box(text)
		parsed_bbox = _parse_bbox(bbox_text)
		assert parsed_bbox == expected_bbox, f"Expected {expected_bbox=} but got {parsed_bbox=} with {text=}"


def _test_extract_repair():
	test_strings = {
		"Can you repair this for me?": "repair",
		"<image>Is it this block?": "Is it this block?",
		"[0.1,0.2, 0.3, 0.6] Do you mean this block? It looks correct.": "Do you mean this block?",
		"I recall you wanting the block here, is this correct? [0.1,0.2, 0.3, 0.6]": "I recall you wanting the block here, is this correct?",
		"This text doesn't contain any corrections.": None,
		"REPAIR should be detected regardless of case.": "repair",
		"Is this where you want to place it? I think it's correct.": "Is this where you want to place it?",
	}

	for text, expected_result in test_strings.items():
		result = _extract_repair(text)
		assert result == expected_result, f"Expected {expected_result=} but got {result=} with {text=}"


# Do quick tests when loading this file
_test_extract_bounding_box()
_test_extract_repair()
