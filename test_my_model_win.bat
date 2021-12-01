set epoch=10

set SPLIT="random"
set INFILE="sample_dataset.csv"

set MODELNAME="test_model.ckpt"
echo "Starting training for: $MODELNAME"
python main.py --infile data/%INFILE% --idx_test_fold 4 --idx_val_fold -1 --model_name %MODELNAME% --epoch %epoch% --cuda False
