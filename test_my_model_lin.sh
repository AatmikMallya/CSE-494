#!/bin/bash

epoch=10

SPLIT=random
INFILE=sample_dataset.csv
INDEX_FILE=data/combined_dataset_${SPLIT}_data_shuffle.txt

MODELNAME=test_model.ckpt
echo "Starting training for: $MODELNAME"
python main.py --infile data/${INFILE} --idx_test_fold 4 --idx_val_fold -1 --model_name $MODELNAME --epoch $epoch --cuda False