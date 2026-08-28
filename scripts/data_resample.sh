#!/usr/bin/env bash

set -euo pipefail

: "${PROJECT_DIR:?PROJECT_DIR must be set}"
: "${DATASETS_DIR:?DATASETS_DIR must be set}"

# Paths
resample_data="$PROJECT_DIR/resample_data" # save resampled multi-rate data
resample_times=3 # number of resampling times

# --------------------- ETTm1 ---------------------
data_name=ETTm1
resample_type=iloc # resampling method, options: [iloc, sum]

ratio=(1 8)
ratio_str=$(IFS=_ ; echo "${ratio[*]}")
python -u "$PROJECT_DIR/data_provider/resample_data.py" \
    --root_path "$DATASETS_DIR/ETT-small" \
    --data_path ${data_name}.csv \
    --resample_type $resample_type \
    --resample_data ${resample_data}/${data_name}/${ratio_str} \
    --ratio "${ratio[@]}" \
    --resample_times $resample_times
echo "Resampling ${data_name} data with ratio ${ratio_str} completed."

ratio=(1 8 16)
ratio_str=$(IFS=_ ; echo "${ratio[*]}")
python -u "$PROJECT_DIR/data_provider/resample_data.py" \
  --root_path "$DATASETS_DIR/ETT-small" \
  --data_path ${data_name}.csv \
  --resample_type $resample_type \
  --resample_data ${resample_data}/${data_name}/${ratio_str} \
  --ratio "${ratio[@]}" \
  --resample_times $resample_times
echo "Resampling ${data_name} data with ratio ${ratio_str} completed."


ratio=(1 4)
ratio_str=$(IFS=_ ; echo "${ratio[*]}")
python -u "$PROJECT_DIR/data_provider/resample_data.py" \
  --root_path "$DATASETS_DIR/ETT-small" \
  --data_path ${data_name}.csv \
  --resample_type $resample_type \
  --resample_data ${resample_data}/${data_name}/${ratio_str} \
  --ratio "${ratio[@]}" \
  --resample_times $resample_times
echo "Resampling ${data_name} data with ratio ${ratio_str} completed."

ratio=(1 4 8)
ratio_str=$(IFS=_ ; echo "${ratio[*]}")
python -u "$PROJECT_DIR/data_provider/resample_data.py" \
  --root_path "$DATASETS_DIR/ETT-small" \
  --data_path ${data_name}.csv \
  --resample_type $resample_type \
  --resample_data ${resample_data}/${data_name}/${ratio_str} \
  --ratio "${ratio[@]}" \
  --resample_times $resample_times
echo "Resampling ${data_name} data with ratio ${ratio_str} completed."

# --------------------- ETTm2 ---------------------
data_name=ETTm2
resample_type=iloc # resampling method, options: [iloc, sum]

ratio=(1 4)
ratio_str=$(IFS=_ ; echo "${ratio[*]}")
python -u "$PROJECT_DIR/data_provider/resample_data.py" \
  --root_path "$DATASETS_DIR/ETT-small" \
  --data_path ${data_name}.csv \
  --resample_type $resample_type \
  --resample_data ${resample_data}/${data_name}/${ratio_str} \
  --ratio "${ratio[@]}" \
  --resample_times $resample_times
echo "Resampling ${data_name} data with ratio ${ratio_str} completed."

ratio=(1 4 8)
ratio_str=$(IFS=_ ; echo "${ratio[*]}")
python -u "$PROJECT_DIR/data_provider/resample_data.py" \
  --root_path "$DATASETS_DIR/ETT-small" \
  --data_path ${data_name}.csv \
  --resample_type $resample_type \
  --resample_data ${resample_data}/${data_name}/${ratio_str} \
  --ratio "${ratio[@]}" \
  --resample_times $resample_times
echo "Resampling ${data_name} data with ratio ${ratio_str} completed."

ratio=(1 8)
ratio_str=$(IFS=_ ; echo "${ratio[*]}")
python -u "$PROJECT_DIR/data_provider/resample_data.py" \
  --root_path "$DATASETS_DIR/ETT-small" \
  --data_path ${data_name}.csv \
  --resample_type $resample_type \
  --resample_data ${resample_data}/${data_name}/${ratio_str} \
  --ratio "${ratio[@]}" \
  --resample_times $resample_times
echo "Resampling ${data_name} data with ratio ${ratio_str} completed."

ratio=(1 8 16)
ratio_str=$(IFS=_ ; echo "${ratio[*]}")
python -u "$PROJECT_DIR/data_provider/resample_data.py" \
  --root_path "$DATASETS_DIR/ETT-small" \
  --data_path ${data_name}.csv \
  --resample_type $resample_type \
  --resample_data ${resample_data}/${data_name}/${ratio_str} \
  --ratio "${ratio[@]}" \
  --resample_times $resample_times
echo "Resampling ${data_name} data with ratio ${ratio_str} completed."

# --------------------- weather ---------------------
data_name=weather
resample_type=iloc # resampling method, options: [iloc, sum]

ratio=(1 4)
ratio_str=$(IFS=_ ; echo "${ratio[*]}")
python -u "$PROJECT_DIR/data_provider/resample_data.py" \
  --root_path "$DATASETS_DIR/${data_name}" \
  --data_path ${data_name}.csv \
  --resample_type $resample_type \
  --resample_data ${resample_data}/${data_name}/${ratio_str} \
  --ratio "${ratio[@]}" \
  --resample_times $resample_times
echo "Resampling ${data_name} data with ratio ${ratio_str} completed."

ratio=(1 4 8)
ratio_str=$(IFS=_ ; echo "${ratio[*]}")
python -u "$PROJECT_DIR/data_provider/resample_data.py" \
  --root_path "$DATASETS_DIR/${data_name}" \
  --data_path ${data_name}.csv \
  --resample_type $resample_type \
  --resample_data ${resample_data}/${data_name}/${ratio_str} \
  --ratio "${ratio[@]}" \
  --resample_times $resample_times
echo "Resampling ${data_name} data with ratio ${ratio_str} completed."

ratio=(1 8)
ratio_str=$(IFS=_ ; echo "${ratio[*]}")
python -u "$PROJECT_DIR/data_provider/resample_data.py" \
  --root_path "$DATASETS_DIR/${data_name}" \
  --data_path ${data_name}.csv \
  --resample_type $resample_type \
  --resample_data ${resample_data}/${data_name}/${ratio_str} \
  --ratio "${ratio[@]}" \
  --resample_times $resample_times
echo "Resampling ${data_name} data with ratio ${ratio_str} completed."

ratio=(1 8 16)
ratio_str=$(IFS=_ ; echo "${ratio[*]}")
python -u "$PROJECT_DIR/data_provider/resample_data.py" \
  --root_path "$DATASETS_DIR/${data_name}" \
  --data_path ${data_name}.csv \
  --resample_type $resample_type \
  --resample_data ${resample_data}/${data_name}/${ratio_str} \
  --ratio "${ratio[@]}" \
  --resample_times $resample_times
echo "Resampling ${data_name} data with ratio ${ratio_str} completed."

# --------------------- traffic ---------------------
data_name=traffic
resample_type=sum # resampling method, options: [iloc, sum]

ratio=(1 4)
ratio_str=$(IFS=_ ; echo "${ratio[*]}")
python -u "$PROJECT_DIR/data_provider/resample_data.py" \
  --root_path "$DATASETS_DIR/${data_name}" \
  --data_path ${data_name}.csv \
  --resample_type $resample_type \
  --resample_data ${resample_data}/${data_name}/${ratio_str} \
  --ratio "${ratio[@]}" \
  --resample_times $resample_times
echo "Resampling ${data_name} data with ratio ${ratio_str} completed."

ratio=(1 4 8)
ratio_str=$(IFS=_ ; echo "${ratio[*]}")
python -u "$PROJECT_DIR/data_provider/resample_data.py" \
  --root_path "$DATASETS_DIR/${data_name}" \
  --data_path ${data_name}.csv \
  --resample_type $resample_type \
  --resample_data ${resample_data}/${data_name}/${ratio_str} \
  --ratio "${ratio[@]}" \
  --resample_times $resample_times
echo "Resampling ${data_name} data with ratio ${ratio_str} completed."

ratio=(1 8)
ratio_str=$(IFS=_ ; echo "${ratio[*]}")
python -u "$PROJECT_DIR/data_provider/resample_data.py" \
  --root_path "$DATASETS_DIR/${data_name}" \
  --data_path ${data_name}.csv \
  --resample_type $resample_type \
  --resample_data ${resample_data}/${data_name}/${ratio_str} \
  --ratio "${ratio[@]}" \
  --resample_times $resample_times
echo "Resampling ${data_name} data with ratio ${ratio_str} completed."

ratio=(1 8 16)
ratio_str=$(IFS=_ ; echo "${ratio[*]}")
python -u "$PROJECT_DIR/data_provider/resample_data.py" \
  --root_path "$DATASETS_DIR/${data_name}" \
  --data_path ${data_name}.csv \
  --resample_type $resample_type \
  --resample_data ${resample_data}/${data_name}/${ratio_str} \
  --ratio "${ratio[@]}" \
  --resample_times $resample_times
echo "Resampling ${data_name} data with ratio ${ratio_str} completed."

# --------------------- electricity ---------------------
data_name=electricity
resample_type=sum # resampling method, options: [iloc, sum]

ratio=(1 4)
ratio_str=$(IFS=_ ; echo "${ratio[*]}")
python -u "$PROJECT_DIR/data_provider/resample_data.py" \
  --root_path "$DATASETS_DIR/${data_name}" \
  --data_path ${data_name}.csv \
  --resample_type $resample_type \
  --resample_data ${resample_data}/${data_name}/${ratio_str} \
  --ratio "${ratio[@]}" \
  --resample_times $resample_times
echo "Resampling ${data_name} data with ratio ${ratio_str} completed."


ratio=(1 4 8)
ratio_str=$(IFS=_ ; echo "${ratio[*]}")
python -u "$PROJECT_DIR/data_provider/resample_data.py" \
  --root_path "$DATASETS_DIR/${data_name}" \
  --data_path ${data_name}.csv \
  --resample_type $resample_type \
  --resample_data ${resample_data}/${data_name}/${ratio_str} \
  --ratio "${ratio[@]}" \
  --resample_times $resample_times
echo "Resampling ${data_name} data with ratio ${ratio_str} completed."


ratio=(1 8)
ratio_str=$(IFS=_ ; echo "${ratio[*]}")
python -u "$PROJECT_DIR/data_provider/resample_data.py" \
  --root_path "$DATASETS_DIR/${data_name}" \
  --data_path ${data_name}.csv \
  --resample_type $resample_type \
  --resample_data ${resample_data}/${data_name}/${ratio_str} \
  --ratio "${ratio[@]}" \
  --resample_times $resample_times
echo "Resampling ${data_name} data with ratio ${ratio_str} completed."


ratio=(1 8 16)
ratio_str=$(IFS=_ ; echo "${ratio[*]}")
python -u "$PROJECT_DIR/data_provider/resample_data.py" \
  --root_path "$DATASETS_DIR/${data_name}" \
  --data_path ${data_name}.csv \
  --resample_type $resample_type \
  --resample_data ${resample_data}/${data_name}/${ratio_str} \
  --ratio "${ratio[@]}" \
  --resample_times $resample_times
echo "Resampling ${data_name} data with ratio ${ratio_str} completed."
