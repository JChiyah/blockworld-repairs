
import sys
from typing import List, Tuple

import pydantic

sys.path.append('../../block-world-training')
from .load_utils import find_closest_discrete_cell_to_point, metrics_distance, calculate_bw_block_distance, _calc_metrics, block_length, calculate_bw_distance, draw_bboxes_in_image, add_text_to_image, load_entries, find_closest_discrete_cell_to_bw_coords, pred_bbox_to_index, _calculate_midpoint

BLOCK_RADIUS = block_length / 2         # centre to midpoint of a side
BLOCK_LENGTH = block_length


def bbox_as_str(bbox: List[float], n_decimals: int = 3) -> str:
	return f"[{bbox[0]:.{n_decimals}f}, {bbox[1]:.{n_decimals}f}, {bbox[2]:.{n_decimals}f}, {bbox[3]:.{n_decimals}f}]"


class UnityData(pydantic.BaseModel):
	world_idx: str
	blocks_bboxes: List[List[float]]
	true_source_index: int
	true_source_coords: List[float]
	true_target_bbox: List[float]
	true_target_coords: List[float]

	@classmethod
	def from_dict(cls, data: dict) -> 'UnityData':
		if 'cand_source_index' in data:
			raise ValueError("This data is not from the Unity dataset, it's from the BW dataset")
		return cls(
			world_idx=data['worldIdx'],
			blocks_bboxes=data['blocksBBoxes'],
			true_source_index=data['sourceIndex'],
			true_source_coords=data['blocks'][data['sourceIndex']],
			true_target_bbox=data['targetBBox'],
			true_target_coords=data['targetLocation']
		)

	@property
	def true_source_xy(self) -> List[int]:
		return convert_coords_to_xy(self.true_source_coords)

	@property
	def true_target_xy(self) -> List[int]:
		return convert_coords_to_xy(self.true_target_coords)

	@staticmethod
	def convert_xy_to_coords(coords: List[float]) -> List[int]:
		# for now, until we change image size, etc
		return convert_coords_to_xy(coords)

	@property
	def true_source_bbox(self) -> List[float]:
		return self.blocks_bboxes[self.true_source_index]


def convert_xy_to_coords(xy: List[int], image_size: Tuple[int, int]) -> List[int]:
	coords = find_closest_discrete_cell_to_point(xy)['coords']

	# we do a little trick, converting xy to a tiny bounding box, then use the same func as when we eval models
	# also ensure that bbox is within image bounds
	# bbox = [xy[0], xy[1], xy[0] + 1, xy[1] + 1]
	# bbox[0] = max(0, min(image_size[0], bbox[0]))
	# bbox[1] = max(0, min(image_size[1], bbox[1]))
	# bbox[2] = max(0, min(image_size[0], bbox[2]))
	# bbox[3] = max(0, min(image_size[1], bbox[3]))

	return coords


def convert_coords_to_xy(coords: List[float]) -> List[int]:
	xy = find_closest_discrete_cell_to_bw_coords(coords)['bbox']
	xy = _calculate_midpoint(xy, False)

	# we do a little trick, converting xy to a tiny bounding box, then use the same func as when we eval models
	# also ensure that bbox is within image bounds
	# bbox = [xy[0], xy[1], xy[0] + 1, xy[1] + 1]
	# bbox[0] = max(0, min(image_size[0], bbox[0]))
	# bbox[1] = max(0, min(image_size[1], bbox[1]))
	# bbox[2] = max(0, min(image_size[0], bbox[2]))
	# bbox[3] = max(0, min(image_size[1], bbox[3]))

	return [int(x) for x in xy]


def get_closest_block_index(bw_coords: List[float], block_bboxes: List[List[float]], threshold_distance: float = BLOCK_RADIUS*2.5) -> int:
	_annotation_source = find_closest_discrete_cell_to_bw_coords(bw_coords)
	_annotation_source_index = pred_bbox_to_index(block_bboxes, _annotation_source['bbox'])

	# distance between coords
	# input(f"distance between placed and selected block: {calculate_bw_distance(bw_coords, _annotation_source['coords'])}")

	return _annotation_source_index


_bbox_height = 60
_bbox_width = 44
def get_approx_bbox_from_xy(xy: List[int]) -> List[float]:

	# we do a little trick, converting xy to a tiny bounding box, then use the same func as when we eval models
	# also ensure that bbox is within image bounds
	bbox = [xy[0] - _bbox_width/2, xy[1] - _bbox_height/2, xy[0] + _bbox_width/2, xy[1] + _bbox_height/2]
	return bbox

# calculate distance to each block
# distances = []
# for i, block in enumerate(block_coords):
# 	distances.append((i, calculate_bw_distance(bw_coords, block)))
#
# # now sort by distance
# distances.sort(key=lambda x: x[1])
#
# if distances[0][1] < threshold_distance:
# 	return distances[0][0]

# else:
# 	raise ValueError(f"Could not find a block close enough to {bw_coords} within {threshold_distance} of {distances[0][1]}\n{distances=}")
