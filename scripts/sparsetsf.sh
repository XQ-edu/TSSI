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
model_name=SparseTSF
gpu=0
itr=3
pred_lens=(48 96 144 192)
ratio_list=("1 4" "1 8" "1 4 8" "1 8 16")
res_list=(1 2 3) # resampling times list, e.g., (1 2 3)

configure_sparsetsf_params() {
  local data_name=$1

  # These variables are assigned in run_sparsetsf's local scope.
  train_epochs=100
  patience=5
  lradj=type9

  case "$data_name" in
    ETTm1|ETTm2)
      learning_rate=0.001
      batch_size=256
      period_len=4
      d_model=128
      ;;
    electricity)
      learning_rate=0.02
      batch_size=128
      period_len=24
      d_model=128
      ;;
    traffic)
      learning_rate=0.01
      batch_size=128
      period_len=24
      d_model=512
      ;;
    weather)
      learning_rate=0.02
      batch_size=256
      period_len=4
      d_model=512
      ;;
    *)
      echo "Error: unsupported SparseTSF dataset: $data_name" >&2
      return 1
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

run_sparsetsf() {
  local data_name=$1
  local enc_in=$2
  local root_path=$3
  local ratio_str ratio_formatted pred_len seq_len id log_dir log_file
  local train_epochs patience lradj learning_rate batch_size period_len d_model
  local -a ratio
  local -a common_args model_args task_args

  configure_sparsetsf_params "$data_name"

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
      seq_len=$((pred_len * 2))
      id="${data_name}/${model_name}/${ratio_formatted}"
      log_dir="${logs}/${id}"
      mkdir -p "${log_dir}"
      log_file="${log_dir}/${seq_len}_${pred_len}.log"

      print_run_info \
        "$model_name" "$data_name" "$seq_len" "$pred_len" "$ratio_str" "${ratio[@]}"

      model_args=(
        --loss mse
        --train_epochs "$train_epochs"
        --patience "$patience"
        --lradj "$lradj"
        --learning_rate "$learning_rate"
        --batch_size "$batch_size"
        --period_len "$period_len"
        --d_model "$d_model"
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
run_sparsetsf ETTm1 7 "$DATASETS_DIR/ETT-small"
run_sparsetsf ETTm2 7 "$DATASETS_DIR/ETT-small"
run_sparsetsf electricity 321 "$DATASETS_DIR/electricity"
run_sparsetsf traffic 862 "$DATASETS_DIR/traffic"
run_sparsetsf weather 21 "$DATASETS_DIR/weather"
