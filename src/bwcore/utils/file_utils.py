
import os
import json

from . import common_utils
from ..configs import Config, DataConfig, DebugConfig, TrainingConfig, _get_run_name
from .. import get_logger
logger = get_logger()


def save_model_outputs(config: Config, responses: list, epoch: int = 0) -> str:
	train_dataset = config.data.train_dataset or 'NA'

	output_file = (
		f"{config.run_name}_{config.goal}_e{epoch}_"
		f"test={common_utils.get_short_dataset_name(config.data.test_dataset)}_"
		f"{common_utils.get_short_dataset_name(train_dataset)}_{config.data.input_turn_masking}"
		f"{'_debug' if config.debug and 'debug' not in config.run_name else ''}.json"
	)

	output_file = os.path.join(config.output_dir_results, output_file)

	# Create output dictionary using config attributes
	output_data = {
		'timestamp': common_utils.get_datetime_prefix(),
		'epoch': epoch,
		'wandb_ids': None,
		'config': config.to_dict(),  # Assuming Config has a to_dict() method
		'outputs': responses
	}

	with open(output_file, 'w') as out_f:
		json.dump(output_data, out_f, indent=4)

	logger.info(f"Model outputs saved to '{output_file}'")

	return output_file


# def save_model_outputs_args(args, responses, epoch: int = 0) -> str:
# 	# function to bridge the gap between the args and the config

# 	# Create DataConfig
# 	data_config = DataConfig(
# 		goal=args.goal,
# 		train_dataset=args.train_dataset if do_train else None,
# 		test_dataset=args.test_dataset,
# 		input_turn_masking=args.turn_masking,
# 		input_task_instruction=args.prompt_task_instruction,
# 		output_repair=True,
# 		output_repair_form='sentence',
# 	)

# 	# Create DebugConfig if debug mode is enabled
# 	debug_config = DebugConfig(
# 		train_num_entries=args.debug_num_entries,
# 		test_num_entries=args.debug_num_entries//2
# 	) if args.debug else None

# 	# Create TrainingConfig
# 	training_config = TrainingConfig(
# 		epochs=args.train_epochs,
# 		learning_rate=args.train_learning_rate,
# 		warmup_ratio=args.train_warmup_ratio,
# 		train_batch_size=args.train_batch_size if not debug_config else debug_config.test_batch_size,
# 		test_batch_size=args.train_batch_size if not debug_config else debug_config.test_batch_size,
# 	)

# 	config = Config(
# 		base_model=args.model_path,
# 		load_model=args.model_load,
# 		task=args.model_task,
# 		do_train=False,
# 		do_eval=True,
# 		run_name=_get_run_name(args),
# 		goal=args.goal,
# 		seed=args.seed,
# 		deterministic=args.deterministic,
# 		output_dir_model=output_dir_model,
# 		output_dir_results=output_dir_results,
# 		use_evaluation_cache=args.use_evaluation_cache,
# 		debug=debug_config,
# 		training=training_config,
# 		data=data_config
# 	)
# 	return save_model_outputs(config, responses, epoch)


def save_model_checkpoint(trainer, output_dir: str):
	# prefix = common_utils.get_datetime_prefix()

	# output_folder = os.path.join(
	# 	os.environ['OUTPUT_DIR'].replace('output', 'checkpoints'),
	# 	bwcore.evaluation.get_short_model_name(config.base_model),
	# 	f"{prefix}_{config.goal}_{bwcore.evaluation.get_short_dataset_name(config.data.train_dataset)}{'_debug' if config.debug else ''}")

	output_dir = os.path.join(output_dir, "checkpoint-final")

	trainer.save_model(output_dir)
	logger.info(f"Model saved to '{output_dir}'")
