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
model_name=TSSI
gpu=0
itr=3
pred_lens=(48 96 144 192)
ratio_list=("1 4" "1 8" "1 4 8" "1 8 16")
res_list=(1 2 3) # resampling times list, e.g., (1 2 3)

configure_tssi_params() {
  local data_name=$1
  local pred_len=$2

  # These variables are assigned in run_tssi's local scope.
  train_epochs=50
  patience=3
  lradj=type1
  loss=huber
  huber_delta=0.01

  case "$data_name:$pred_len" in
    ETTm1:48)
      freq=t
      d_model=64
      d_ff=64
      n_heads=4
      e_layers=1
      dropout=0.3
      n_res_layers=1
      in_dim=128
      hidden_dim=512
      learning_rate=0.005
      batch_size=128
      ;;
    ETTm1:96)
      freq=t
      d_model=256
      d_ff=256
      n_heads=2
      e_layers=1
      dropout=0.2
      n_res_layers=2
      in_dim=64
      hidden_dim=64
      learning_rate=0.005
      batch_size=128
      ;;
    ETTm1:144)
      freq=t
      d_model=256
      d_ff=512
      n_heads=4
      e_layers=2
      dropout=0.4
      n_res_layers=2
      in_dim=128
      hidden_dim=256
      learning_rate=0.0001
      batch_size=32
      ;;
    ETTm1:192)
      freq=t
      d_model=256
      d_ff=256
      n_heads=4
      e_layers=3
      dropout=0.3
      n_res_layers=1
      in_dim=32
      hidden_dim=32
      learning_rate=0.0001
      batch_size=16
      ;;
    ETTm2:48)
      freq=t
      d_model=64
      d_ff=256
      n_heads=4
      e_layers=1
      dropout=0.3
      n_res_layers=2
      in_dim=256
      hidden_dim=256
      learning_rate=0.005
      batch_size=64
      ;;
    ETTm2:96)
      freq=t
      d_model=64
      d_ff=64
      n_heads=1
      e_layers=3
      dropout=0.3
      n_res_layers=1
      in_dim=128
      hidden_dim=128
      learning_rate=0.0005
      batch_size=128
      ;;
    ETTm2:144)
      freq=t
      d_model=64
      d_ff=256
      n_heads=8
      e_layers=3
      dropout=0.2
      n_res_layers=4
      in_dim=64
      hidden_dim=64
      learning_rate=0.0001
      batch_size=16
      ;;
    ETTm2:192)
      freq=t
      d_model=128
      d_ff=1024
      n_heads=4
      e_layers=1
      dropout=0.3
      n_res_layers=1
      in_dim=64
      hidden_dim=64
      learning_rate=0.0001
      batch_size=64
      ;;
    weather:48)
      freq=t
      d_model=128
      d_ff=256
      n_heads=4
      e_layers=3
      dropout=0.2
      n_res_layers=3
      in_dim=64
      hidden_dim=128
      learning_rate=0.0005
      batch_size=16
      ;;
    weather:96)
      freq=t
      d_model=256
      d_ff=256
      n_heads=4
      e_layers=3
      dropout=0.2
      n_res_layers=1
      in_dim=64
      hidden_dim=64
      learning_rate=0.0001
      batch_size=16
      ;;
    weather:144)
      freq=t
      d_model=256
      d_ff=256
      n_heads=1
      e_layers=2
      dropout=0.3
      n_res_layers=1
      in_dim=64
      hidden_dim=64
      learning_rate=0.0001
      batch_size=16
      ;;
    weather:192)
      freq=t
      d_model=256
      d_ff=256
      n_heads=1
      e_layers=2
      dropout=0.3
      n_res_layers=1
      in_dim=64
      hidden_dim=128
      learning_rate=0.0001
      batch_size=32
      ;;
    electricity:48)
      freq=h
      d_model=64
      d_ff=128
      n_heads=4
      e_layers=3
      dropout=0.2
      n_res_layers=3
      in_dim=256
      hidden_dim=1024
      learning_rate=0.005
      batch_size=32
      ;;
    electricity:96)
      freq=h
      d_model=128
      d_ff=128
      n_heads=8
      e_layers=1
      dropout=0.05
      n_res_layers=1
      in_dim=128
      hidden_dim=512
      learning_rate=0.001
      batch_size=16
      ;;
    electricity:144)
      freq=h
      d_model=64
      d_ff=128
      n_heads=4
      e_layers=2
      dropout=0.2
      n_res_layers=1
      in_dim=256
      hidden_dim=1024
      learning_rate=0.0005
      batch_size=16
      ;;
    electricity:192)
      freq=h
      d_model=128
      d_ff=128
      n_heads=4
      e_layers=2
      dropout=0.3
      n_res_layers=1
      in_dim=256
      hidden_dim=256
      learning_rate=0.0005
      batch_size=32
      ;;
    traffic:*)
      freq=h
      d_model=256
      d_ff=256
      n_heads=4
      e_layers=1
      dropout=0.3
      n_res_layers=4
      in_dim=1024
      hidden_dim=1024
      learning_rate=0.001
      batch_size=16
      huber_delta=0.5
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

run_tssi() {
  local data_name=$1
  local enc_in=$2
  local root_path=$3
  local ratio_str ratio_formatted pred_len seq_len id log_dir log_file
  local freq d_model d_ff n_heads e_layers dropout n_res_layers in_dim hidden_dim
  local train_epochs patience lradj loss huber_delta learning_rate batch_size
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
      configure_tssi_params "$data_name" "$pred_len"
      seq_len=$((pred_len * 2))
      id="${data_name}/${model_name}/${ratio_formatted}"
      log_dir="${logs}/${id}"
      mkdir -p "${log_dir}"
      log_file="${log_dir}/${seq_len}_${pred_len}.log"

      print_run_info \
        "$model_name" "$data_name" "$seq_len" "$pred_len" "$ratio_str" "${ratio[@]}"

      model_args=(
        --freq "$freq"
        --d_model "$d_model"
        --d_ff "$d_ff"
        --n_heads "$n_heads"
        --e_layers "$e_layers"
        --dropout "$dropout"
        --n_res_layers "$n_res_layers"
        --in_dim "$in_dim"
        --hidden_dim "$hidden_dim"
        --loss "$loss"
        --huber_delta "$huber_delta"
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
        --train_epochs "$train_epochs"
        --patience "$patience"
        --lradj "$lradj"
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
run_tssi ETTm1 7 "$DATASETS_DIR/ETT-small"
run_tssi ETTm2 7 "$DATASETS_DIR/ETT-small"
run_tssi weather 21 "$DATASETS_DIR/weather"
run_tssi electricity 321 "$DATASETS_DIR/electricity"
run_tssi traffic 862 "$DATASETS_DIR/traffic"
