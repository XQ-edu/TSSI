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
model_name=Duet
gpu=0 # 0, 1, 2
itr=3 # 1, 3
pred_lens=(48 96 144 192)
ratio_list=("1 4" "1 8" "1 4 8" "1 8 16")
res_list=(1 2 3) # resampling times list, e.g., (1 2 3)

# Duet source scripts use CI=1; in this codebase it is represented by --individual.
configure_duet_params() {
  local data_name=$1

  # These variables are assigned in run_duet's local scope.
  d_ff=2048
  d_model=512
  dropout=0.2
  fc_dropout=0.2
  e_layers=2
  factor=1
  hidden_size=256
  k=1
  loss=mae
  learning_rate=0.02
  lradj=type8
  n_heads=8
  train_epochs=100
  num_experts=4
  patch_len=16
  patience=10
  batch_size=256

  case "$data_name" in
    ETTm1)
      batch_size=64
      d_ff=512
      d_model=512
      dropout=0.5
      fc_dropout=0.2
      e_layers=1
      factor=3
      k=2
      learning_rate=0.0005
      lradj=type1
      n_heads=1
      patch_len=48
      patience=5
      ;;
    ETTm2)
      batch_size=128
      d_ff=512
      d_model=512
      e_layers=1
      factor=3
      hidden_size=512
      k=2
      learning_rate=0.0005
      lradj=type1
      n_heads=1
      patch_len=48
      patience=5
      ;;
    electricity)
      batch_size=16
      d_ff=1024
      d_model=512
      dropout=0.5
      fc_dropout=0
      e_layers=4
      factor=3
      k=2
      learning_rate=0.0005
      lradj=type1
      n_heads=1
      patch_len=48
      patience=5
      ;;
    traffic)
      batch_size=16
      d_ff=1024
      d_model=512
      dropout=0
      fc_dropout=0
      e_layers=4
      factor=3
      k=2
      learning_rate=0.0005
      lradj=type1
      n_heads=1
      train_epochs=10
      patch_len=48
      patience=3
      ;;
    weather)
      batch_size=32
      d_ff=256
      d_model=256
      e_layers=2
      factor=3
      k=2
      learning_rate=0.0005
      lradj=type1
      n_heads=1
      patch_len=48
      patience=5
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

run_duet() {
  local data_name=$1
  local enc_in=$2
  local root_path=$3
  local ratio_str ratio_formatted pred_len seq_len id log_dir log_file
  local d_ff d_model dropout fc_dropout e_layers factor hidden_size k loss
  local learning_rate lradj n_heads train_epochs num_experts patch_len patience batch_size
  local -a ratio
  local -a common_args duet_args task_args

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
      configure_duet_params "$data_name"
      seq_len=$((pred_len * 2))
      if [ "$data_name" = "traffic" ]; then
        case "$pred_len" in
          144|192) batch_size=8 ;;
        esac
      fi
      id="${data_name}/${model_name}/${ratio_formatted}"
      log_dir="${logs}/${id}"
      mkdir -p "${log_dir}"
      log_file="${log_dir}/${seq_len}_${pred_len}.log"

      print_run_info \
        "$model_name" "$data_name" "$seq_len" "$pred_len" "$ratio_str" "${ratio[@]}"

      duet_args=(
        --d_model "$d_model"
        --d_ff "$d_ff"
        --hidden_size "$hidden_size"
        --n_heads "$n_heads"
        --e_layers "$e_layers"
        --factor "$factor"
        --dropout "$dropout"
        --fc_dropout "$fc_dropout"
        --patch_len "$patch_len"
        --num_experts "$num_experts"
        --noisy_gating 1
        --k "$k"
        --loss "$loss"
        --individual
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
        --patience "$patience"
        --lradj "$lradj"
        --learning_rate "$learning_rate"
        --batch_size "$batch_size"
        --ratio "${ratio[@]}"
      )

      python -u "$PROJECT_DIR/run_longExp.py" \
        "${common_args[@]}" \
        "${duet_args[@]}" \
        "${task_args[@]}" \
        > "$log_file" 2>&1
    done
  done
}

# -------------------------------------------
run_duet ETTm1 7 "$DATASETS_DIR/ETT-small"
run_duet ETTm2 7 "$DATASETS_DIR/ETT-small"
run_duet electricity 321 "$DATASETS_DIR/electricity"
run_duet traffic 862 "$DATASETS_DIR/traffic"
run_duet weather 21 "$DATASETS_DIR/weather"
