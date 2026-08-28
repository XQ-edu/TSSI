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
model_name=DeepEDM
gpu=0
itr=3
pred_lens=(48 96 144 192)
ratio_list=("1 4" "1 8" "1 4 8" "1 8 16")
res_list=(1 2 3) # resampling times list, e.g., (1 2 3)

configure_deepedm_params() {
  local data_name=$1

  # These variables are assigned in run_deepedm's local scope.
  n_mlp_layers=2
  n_edm_blocks=1
  delay=5
  time_delay_stride=1
  mlp_dropout=0.1
  edm_dropout=0.1
  learning_rate=0.0005
  reduce_lr_factor=0.9
  train_epochs=250
  batch_size=32
  patience=15
  min_lr=0.00005
  lradj=type10
  opt_type=AdamW
  loss=mse

  case "$data_name" in
    ETTm1)
      n_mlp_layers=1
      delay=7
      train_epochs=150
      patience=10
      loss=mae
      ;;
    ETTm2)
      n_mlp_layers=1
      delay=9
      train_epochs=150
      patience=10
      loss=mae
      ;;
    electricity)
      n_mlp_layers=2
      n_edm_blocks=3
      delay=21
      time_delay_stride=2
      learning_rate=0.001
      reduce_lr_factor=0.99
      ;;
    traffic)
      n_mlp_layers=2
      n_edm_blocks=3
      delay=21
      time_delay_stride=2
      learning_rate=0.001
      reduce_lr_factor=0.99
      ;;
    weather)
      ;;
    *)
      echo "Error: unsupported DeepEDM dataset: $data_name" >&2
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

run_deepedm() {
  local data_name=$1
  local enc_in=$2
  local root_path=$3
  local ratio_str ratio_formatted pred_len seq_len id log_dir log_file
  local n_mlp_layers n_edm_blocks delay time_delay_stride mlp_dropout edm_dropout
  local learning_rate reduce_lr_factor train_epochs batch_size patience min_lr
  local lradj opt_type loss
  local -a ratio
  local -a common_args model_args task_args

  configure_deepedm_params "$data_name"

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
        --n_mlp_layers "$n_mlp_layers"
        --n_edm_blocks "$n_edm_blocks"
        --delay "$delay"
        --time_delay_stride "$time_delay_stride"
        --mlp_dropout "$mlp_dropout"
        --edm_dropout "$edm_dropout"
        --learning_rate "$learning_rate"
        --reduce_lr_factor "$reduce_lr_factor"
        --train_epochs "$train_epochs"
        --batch_size "$batch_size"
        --patience "$patience"
        --min_lr "$min_lr"
        --lradj "$lradj"
        --opt_type "$opt_type"
        --loss "$loss"
        --use_tdalign
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
run_deepedm ETTm1 7 "$DATASETS_DIR/ETT-small"
run_deepedm ETTm2 7 "$DATASETS_DIR/ETT-small"
run_deepedm electricity 321 "$DATASETS_DIR/electricity"
run_deepedm traffic 862 "$DATASETS_DIR/traffic"
run_deepedm weather 21 "$DATASETS_DIR/weather"
