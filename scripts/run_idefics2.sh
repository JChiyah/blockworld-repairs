#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --job-name=run_idefics2
#SBATCH --output=/users/fjc3/sharedscratch/scrum/run_idefics2/%J.out
#SBATCH --gres gpu:1
#SBATCH --chdir=/users/fjc3/block-world-training/idefics2

# sample runs:
# 	sbatch run_idefics2.sh zeroshot source
# 	sbatch run_idefics2.sh zeroshot source,target
# 	sbatch run_idefics2.sh finetune target
# set -euo pipefail  # Exit on error, undefined variables, and pipe failures


# Configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly BASE_DIR="$(dirname "$SCRIPT_DIR")"
readonly LOG_DIR="/users/fjc3/sharedscratch/scrum"
readonly CONDA_ENV="/users/fjc3/block-world-training/.envs/idefics2"

# Model configuration
readonly MODEL_BASE="HuggingFaceM4/idefics2-8b"
readonly MODEL_LOAD="qlora"

# Experiment parameters
# always test on fixed_original_test, as it is the full dataset. Results will be split depending on instructions/corrections
declare -a TEST_DATASETS=("fixed_original_test")
# 3 possible train datasets: instructions, corrections, both
declare -a TRAIN_DATASETS=("fixed_original_test")	# fixed_original_test,fixed_original_test_instructions,fixed_original_test_corrections
declare -a TURN_MASKING=("all")						# none,assistant,all
declare -a PROMPT_TASK_INSTRUCTION=("user")			# user,system

declare -a MODEL_TASK=$1							# finetune,zeroshot
# declare -a GOALS=("target")
declare -a GOALS=("source")							# source,target
IFS=',' read -ra GOALS <<< "$2"						# parse goal string into array


# Logging functions
log() {
	echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

error() {
	log "ERROR: $*" >&2
}

# parse_goals() {
#     local IFS=',' read -ra GOALS <<< "$1"
#     log "Parsed goals: ${GOALS[*]}"
# }

# Setup environment
setup_environment() {
	log "Setting up environment..."

	flight env activate gridware
	module load libs/nvidia-cuda
	source /users/fjc3/.bashrc
	conda activate "$CONDA_ENV"
}

# Calculate total experiments
calculate_total_experiments() {
	local test_count=${#TEST_DATASETS[@]}
	local train_count=${#TRAIN_DATASETS[@]}
	local goal_count=${#GOALS[@]}
	local mask_count=${#TURN_MASKING[@]}
	local prompt_count=${#PROMPT_TASK_INSTRUCTION[@]}

	if [[ "$MODEL_TASK" == "finetune" ]]; then
		echo $((test_count * goal_count * mask_count * prompt_count * train_count))
	else
		echo $((test_count * goal_count * prompt_count))
	fi
}

# Run experiment with error handling
run_experiment() {
	local cmd=(
		python "$BASE_DIR/src/idefics2_main.py"
		--base_model "$MODEL_BASE"
		--load_model "$MODEL_LOAD"
		--task "$1"
		--train_dataset "$2"
		--test_dataset "$3"
		--turn_masking "$4"
		--prompt_task_instruction "$5"
		--goal "$6"
		"${@:7}"
	)

	log "Running command: "
	echo "${cmd[*]}"
	if ! "${cmd[@]}"; then
		error "Experiment failed: ${cmd[*]}"
		return 1
	fi
	return 0
}

# Handle finetune experiments
finetune_experiments() {
	local total_experiments=$(calculate_total_experiments)
	log "Starting finetuning experiments. Total experiments: $total_experiments"
	local current_exp=0

	for test_data in "${TEST_DATASETS[@]}"; do
		for train_data in "${TRAIN_DATASETS[@]}"; do
			for goal in "${GOALS[@]}"; do
				for mask in "${TURN_MASKING[@]}"; do
					for prompt in "${PROMPT_TASK_INSTRUCTION[@]}"; do
						local current_exp=$((current_exp + 1))
						log "*** Experiment $current_exp/$total_experiments ***"
						run_experiment "sft" "$train_data" "$test_data" "$mask" "$prompt" "$goal" "$@"
					done
				done
			done
		done
	done
}

# Handle zeroshot experiments
zeroshot_experiments() {
	local total_experiments=$(calculate_total_experiments)
	log "Starting zeroshot experiments. Total experiments: $total_experiments"
	local current_exp=0

	local train_data="original_entries"
	local mask="none"

	for test_data in "${TEST_DATASETS[@]}"; do
		for prompt in "${PROMPT_TASK_INSTRUCTION[@]}"; do
			for goal in "${GOALS[@]}"; do
				local current_exp=$((current_exp + 1))
				log "*** Experiment $current_exp/$total_experiments ***"
				run_experiment "eval" "$train_data" "$test_data" "$mask" "$prompt" "$goal" "$@"
			done
		done
	done
}

# Main execution
main() {
	if [[ $# -lt 2 ]]; then
		error "Usage: $0 <model_task> <goal> [<optional>]"
		error "model_task: finetune or zeroshot"
		error "goal: source or target"
		exit 1
	fi

	setup_environment

	case "$MODEL_TASK" in
		"zeroshot")
			zeroshot_experiments "${@:3}"
			;;
		"finetune")
			finetune_experiments "${@:3}"
			;;
		*)
			error "Unknown model task: $MODEL_TASK"
			exit 1
			;;
	esac

	log "All experiments completed"
}

# Execute main function with all arguments
main "$@"
