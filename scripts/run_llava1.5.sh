#! /bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --job-name=run_llava1.5
#SBATCH --output=/users/fjc3/sharedscratch/scrum/run_llava1.5.%J.out
#SBATCH --gres gpu:1
#SBATCH --chdir=/users/fjc3/block-world-training/blockworld-repairs

# Sample runs:
# Finetune:
#   sbatch run_llava1.5.sh finetune source,target liuhaotian/llava-v1.5-7b fixed_original_test
# Zeroshot:
#   sbatch run_llava1.5.sh zeroshot source,target liuhaotian/llava-v1.5-7b

# Configuration
readonly CONDA_ENV="./.envs/llava"
readonly LLAVA_DIR="LLaVA"
readonly LLAVA_SCRIPTS_DIR="LLaVA/scripts"
readonly LLAVA_DATASETS_DIR="/users/fjc3/sharedscratch/datasets/llava"
# Get the directory containing the script.
# This script should be called either from the root or from the scripts directory
# Determine PROJECT_ROOT (base dir) and SCRIPT_DIR (scripts folder) reliably.
_SCRIPT_DIR_TEMP="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
readonly SCRIPT_DIR="$_SCRIPT_DIR_TEMP"
readonly PROJECT_ROOT="$(dirname "$_SCRIPT_DIR_TEMP")"

# Experiment parameters
declare -a TURN_MASKING_OPTIONS=("all" "assistant" "none")
declare -a ZEROSHOT_TEST_DATASETS=("fixed_original_test")

# Positional Arguments
MODEL_TASK=$1
GOALS_STR=$2
MODEL_BASE=$3
TRAIN_DATASET_ARG=$4 # Only used for finetune task

# Logging functions
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

error() {
    log "ERROR: $*" >&2
}

# Setup environment
setup_environment() {
    log "Setting up environment..."
    flight env activate gridware
    module load libs/nvidia-cuda
    source /users/fjc3/.bashrc
    conda activate "$CONDA_ENV"
    if [ $? -ne 0 ]; then
        error "Conda environment activation failed: $CONDA_ENV"
        exit 1
    fi
    log "Environment setup complete."
}

# Run data conversion
# Args: $1 output_dir_suffix, $2 dataset_to_convert, $3 mask_to_use, $4+ additional args
run_conversion() {
    local dataset_to_convert="$1"
    local mask_to_use="$2"
    shift 2  # Remove the first 2 arguments

    log "Converting dataset: $dataset_to_convert with mask: $mask_to_use to $LLAVA_DATASETS_DIR"
    if ! python $PROJECT_ROOT/src/convert_dataset_to_llava_format.py --output_dir_results "$LLAVA_DATASETS_DIR" --train_dataset "$dataset_to_convert" --turn_masking "$mask_to_use" "$@"; then
        error "Data conversion failed for dataset $dataset_to_convert, mask $mask_to_use"
        # Optionally exit here or let the main script try to continue
        exit 1
    fi
}

# Calculate and log total experiments
# Args: $1 task_type ("finetune" or "zeroshot"), $2 num_goals
calculate_and_log_total_experiments() {
    local task_type="$1"
    local num_goals="$2"
    local total_experiments=0

    if [[ "$task_type" == "finetune" ]]; then
        total_experiments=$((num_goals * ${#TURN_MASKING_OPTIONS[@]}))
    elif [[ "$task_type" == "zeroshot" ]]; then
        total_experiments=$((num_goals * ${#ZEROSHOT_TEST_DATASETS[@]}))
    fi
    log "Total number of $task_type experiments to run: $total_experiments"
}

# Handle finetune experiments
# Args: $1 model_base, $2 train_dataset
finetune_tasks() {
    local model_base_arg="$1"
    local train_dataset_for_finetune="$2"
    # shift 2 # Remove model_base_arg and train_dataset_for_finetune from argument list
    local extra_args=("$@") # Capture remaining arguments

    # In the original llava_finetune_experiments.sh, test_dataset was set to the train_dataset argument
    local test_dataset_for_finetune="$train_dataset_for_finetune"
    local current_exp=0
    local adapter_path_na="NA"

    log "Starting LLaVA finetuning tasks..."
    calculate_and_log_total_experiments "finetune" "${#GOALS[@]}"

    for goal in "${GOALS[@]}"; do
        for mask in "${TURN_MASKING_OPTIONS[@]}"; do
            current_exp=$((current_exp + 1))
            log "*** Finetune Experiment $current_exp: Goal: $goal, Train/Test Dataset: $train_dataset_for_finetune, Mask: $mask ***"

            # convert data to LLaVA format
			# TODO: uncomment this and below
            # run_conversion "$train_dataset_for_finetune" "$mask" "${@:3}"

			# run an initial evaluation
			# evaluate_llava "zeroshot" "$goal" "$train_dataset_for_finetune" "$mask" "$adapter_path_na" "$model_base_arg" "${extra_args[@]}"

			# run the finetuning
            log "Executing: sh $LLAVA_SCRIPTS_DIR/v1_5/finetune_task_lora.sh $goal $train_dataset_for_finetune $test_dataset_for_finetune $mask $model_base_arg"
            if ! sh $LLAVA_SCRIPTS_DIR/v1_5/finetune_task_lora.sh "$goal" "$train_dataset_for_finetune" "$test_dataset_for_finetune" "$mask" "$model_base_arg"; then
                error "Finetuning script failed for Goal: $goal, Dataset: $train_dataset_for_finetune, Mask: $mask"
				exit 1
            fi
			# todo: get model from finetune_task_lora and evaluate it

			# run a final evaluation
			evaluate_llava "finetuned" "$goal" "$train_dataset_for_finetune" "$mask" "$adapter_path_na" "$model_base_arg" "${extra_args[@]}"

        done
    done
}

# Handle zeroshot experiments
# Args: $1 model_base
zeroshot_tasks() {
    local model_base_arg="$1"
    shift 1 # Remove model_base_arg from argument list
    local extra_args=("$@") # Capture remaining arguments

    local train_dataset_for_zeroshot="NA"
    local mask_for_zeroshot="none"
    local adapter_path_na="NA"
    local current_exp=0

    log "Starting LLaVA zeroshot evaluation tasks..."
    calculate_and_log_total_experiments "zeroshot" "${#GOALS[@]}"

    for goal in "${GOALS[@]}"; do
        for test_data in "${ZEROSHOT_TEST_DATASETS[@]}"; do
            current_exp=$((current_exp + 1))
            log "*** Zeroshot Experiment $current_exp: Goal: $goal, Test Dataset: $test_data ***"

            # Convert data for the specific test dataset
            run_conversion "$test_data" "$mask_for_zeroshot" "${extra_args[@]}"

            evaluate_llava "zeroshot" "$goal" "$test_data" "$mask_for_zeroshot" "$adapter_path_na" "$model_base_arg" "${extra_args[@]}"
        done
    done
}

evaluate_llava() {
    local task="$1"
    local goal="$2"
    local test_data="$3"
    local mask="$4"
    local adapter_path="$5"
    local model_base_arg="$6"

    log "Executing: sh $LLAVA_SCRIPTS_DIR/v1_5/eval_task_lora.sh $task $goal $test_data $mask $adapter_path $model_base_arg"
    if ! sh $LLAVA_SCRIPTS_DIR/v1_5/eval_task_lora.sh "$task" "$goal" "$test_data" "$mask" "$adapter_path" "$model_base_arg"; then
        error "Evaluation script failed for Task: $task, Goal: $goal, Test Dataset: $test_data"
        exit 1
    fi
    return 0
}

# Main execution
main() {
    # Validate arguments
    if [[ -z "$MODEL_TASK" ]] || [[ -z "$GOALS_STR" ]] || [[ -z "$MODEL_BASE" ]]; then
        error "Usage: $0 <model_task> <goals_str> <model_base> [train_dataset_for_finetune] [additional_args...]"
        error "  model_task: finetune or zeroshot"
        error "  goals_str: comma-separated list of goals (e.g., source,target)"
        error "  model_base: HuggingFace model name (e.g., liuhaotian/llava-v1.5-7b)"
        error "  train_dataset_for_finetune: required if model_task is finetune (e.g., fixed_original_test)"
        error "  additional_args: any additional arguments to pass to the conversion script"
        exit 1
    fi

    if [[ "$MODEL_TASK" == "finetune" && -z "$TRAIN_DATASET_ARG" ]]; then
        error "TRAIN_DATASET argument is required for finetune task."
        error "Usage: $0 finetune <goals_str> <model_base> <train_dataset>"
        exit 1
    fi

    IFS=',' read -ra GOALS <<< "$GOALS_STR"
    if [ ${#GOALS[@]} -eq 0 ]; then
        error "No goals provided in GOALS_STR: $GOALS_STR"
        exit 1
    fi
    log "Parsed goals: ${GOALS[*]}"

    setup_environment
    log "Starting LLaVA tasks with Model: $MODEL_BASE, Task: $MODEL_TASK"

    # Capture extra arguments to pass to task functions
    local all_args=("$@")
    local extra_args_for_tasks=()

    case "$MODEL_TASK" in
        "finetune")
            # For finetune, 4 args are MODEL_TASK, GOALS_STR, MODEL_BASE, TRAIN_DATASET_ARG
            # Extra args start from the 5th position ($5)
            if [ "$#" -ge 5 ]; then
                extra_args_for_tasks=("${all_args[@]:4}")
            fi
            finetune_tasks "$MODEL_BASE" "$TRAIN_DATASET_ARG" "${extra_args_for_tasks[@]}"
            ;;
        "zeroshot")
            # For zeroshot, 3 args are MODEL_TASK, GOALS_STR, MODEL_BASE
            # Extra args start from the 4th position ($4)
            if [ "$#" -ge 4 ]; then
                extra_args_for_tasks=("${all_args[@]:3}")
            fi
            zeroshot_tasks "$MODEL_BASE" "${extra_args_for_tasks[@]}"
            ;;
        *)
            error "Unknown model task: $MODEL_TASK. Must be 'finetune' or 'zeroshot'."
            exit 1
            ;;
    esac

    log "All LLaVA $MODEL_TASK experiments completed for goals: ${GOALS[*]}"
}

# Execute main function with all script arguments
main "$@"
