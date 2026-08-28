#!/usr/bin/env bash

set -euo pipefail

: "${PROJECT_DIR:?PROJECT_DIR must be set}"
: "${DATASETS_DIR:?DATASETS_DIR must be set}"

# Paths
resample_data="$PROJECT_DIR/resample_data" # save resampled multi-rate data
checkpoints="$PROJECT_DIR/checkpoints" # save model
results="$PROJECT_DIR/results" # save prediction results and various metrics of each epoch
logs="$PROJECT_DIR/logs" # save running logs

# Experiment settings
model_name=CrossLinear
gpu=0
itr=3
pred_lens=(48 96 144 192)
ratio_list=("1 4" "1 8" "1 4 8" "1 8 16")
res_list=(1 2 3) # resampling times list, e.g., (1 2 3)

configure_crosslinear_params() {
  local data_name=$1

  # These variables are assigned in run_crosslinear's local scope.
  patch_len=16
  d_model=1024
  d_ff=1024
  alpha=1
  beta=0.5
  batch_size=32
  train_epochs=50
  learning_rate=0.0001

  case "$data_name" in
    ETTm2)
      d_model=512
      d_ff=2048
      learning_rate=0.00001
      ;;
    traffic)
      patch_len=32
      d_model=8192
      d_ff=8192
      batch_size=4
      learning_rate=0.00001
      ;;
    electricity)
      d_model=4096
      d_ff=4096
      batch_size=16
      ;;
    weather)
      d_model=512
      d_ff=512
      ;;
  esac
}

print_run_info() {
  local model_name=$1
  local data_name=$2
  local seq_len=$3
  local pred_len=$4
  local ratio_str=$5
  shift 5
  local -a ratio=("$@")

  echo "---------------Running---------------"
  echo "Model=${model_name}, Data=${data_name}, seq_len=${seq_len}, pred_len=${pred_len}"
  echo "Ratio configuration: ${ratio_str}"
  echo "The data sampling ratio = highest / lower:"
  for r in "${ratio[@]}"; do
    echo "ratio=$r: seq length=$((seq_len / r)), pred length=$((pred_len / r))"
  done
}

run_crosslinear() {
  local data_name=$1
  local enc_in=$2
  local root_path=$3
  local ratio_str ratio_formatted pred_len seq_len id log_dir log_file
  local patch_len d_model d_ff alpha beta batch_size train_epochs learning_rate
  local -a ratio
  local -a common_args model_args task_args

  common_args=(
    --is_training 1
    --res_list "${res_list[@]}"
    --model "$model_name"
    --des Exp
    --gpu "$gpu"
    --itr "$itr"
  )

  for ratio_str in "${ratio_list[@]}"; do
    ratio=($ratio_str)
    ratio_formatted=$(IFS=_ ; echo "${ratio[*]}")
    for pred_len in "${pred_lens[@]}"; do
      configure_crosslinear_params "$data_name"
      seq_len=$((pred_len * 2))
      id="${data_name}/${model_name}/${ratio_formatted}"
      log_dir="${logs}/${id}"
      mkdir -p "${log_dir}"
      log_file="${log_dir}/${seq_len}_${pred_len}.log"

      print_run_info \
        "$model_name" "$data_name" "$seq_len" "$pred_len" "$ratio_str" "${ratio[@]}"

      model_args=(
        --loss mse
        --patch_len "$patch_len"
        --d_model "$d_model"
        --d_ff "$d_ff"
        --alpha "$alpha"
        --beta "$beta"
      )

      task_args=(
        --root_path "$root_path"
        --data_path "${data_name}.csv"
        --checkpoints "${checkpoints}/${id}/${seq_len}_${pred_len}"
        --results "${results}/${id}/${seq_len}_${pred_len}"
        --resample_data "${resample_data}/${data_name}/${ratio_formatted}"
        --data "$data_name"
        --seq_len "$seq_len"
        --pred_len "$pred_len"
        --enc_in "$enc_in"
        --dec_in "$enc_in"
        --c_out "$enc_in"
        --train_epochs "$train_epochs"
        --learning_rate "$learning_rate"
        --batch_size "$batch_size"
        --ratio "${ratio[@]}"
      )

      python -u "$PROJECT_DIR/run_longExp.py" \
        "${common_args[@]}" \
        "${model_args[@]}" \
        "${task_args[@]}" \
        > "$log_file" 2>&1
    done
  done
}

# -------------------------------------------
run_crosslinear ETTm1 7 "$DATASETS_DIR/ETT-small"
run_crosslinear ETTm2 7 "$DATASETS_DIR/ETT-small"
run_crosslinear traffic 862 "$DATASETS_DIR/traffic"
run_crosslinear electricity 321 "$DATASETS_DIR/electricity"
run_crosslinear weather 21 "$DATASETS_DIR/weather"
