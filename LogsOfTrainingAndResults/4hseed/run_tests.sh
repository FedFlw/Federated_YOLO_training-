#!/usr/bin/env bash
# ------------------------------------------------------------------
# Evaluate five YOLO models on three test datasets
# and keep the output of every run in its own log file.
# ------------------------------------------------------------------

# “Seeds” are the model‑folder numbers you want to test
seeds=(291 331 368 409 450)

for seed in "${seeds[@]}"; do
  model="/WehbeDocker/YOLOV8N/runs/detect/train${seed}/weights/best.pt"

  # 1. dataset5
  yolo val \
       model="$model" \
       data='/WehbeDocker/dataset5/data.yaml' \
       &> "testing${seed}all.log"

  # 2. TACO
  yolo val \
       model="$model" \
       data='/WehbeDocker/FL_litter/TACO/data.yaml' \
       &> "testing${seed}TACO.log"

  # 3. PlastoOPol
  yolo val \
       model="$model" \
       data='/WehbeDocker/FL_litter/PlastoOPol/data.yaml' \
       &> "testing${seed}Plast.log"

done
