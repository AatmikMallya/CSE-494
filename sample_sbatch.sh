#!/bin/bash

#SBATCH -N 1  # number of nodes
#SBATCH -n 2  # number of "tasks" (default: allocates 1 core per task)
#SBATCH -t 0-10:00:00   # time in d-hh:mm:ss
#SBATCH -p publicgpu    # partition 
#SBATCH -q wildfire     # QOS
#SBATCH --gres=gpu:1    # Request only one GPU
#SBATCH -o slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL # Send an e-mail when a job starts, stops, or fails
#SBATCH --mail-user=%u@asu.edu # Mail-to address
#SBATCH --export=NONE   # Purge the job-submitting shell environment

# Load the required modules
module purge
module load cuda/10.2.89
source tcr-env/bin/activate

# Number of parallel processes
N=2
i=0
# GPU to run on 
gpu=0

SPLIT=tcr
INFILE=student_dataset.csv

# START EDITING HERE

epoch=100

# Adding a hyperparameter start here:
# Define all of the parameter values you want to test
# Each should be a list of values, like linear_size which has 2 values: 256 and 512

linear_size=( 256 512 )

# Make a for loop for each of the parameters like so
for linear in "${linear_size[@]}"
do

# Keep this as the innermost loop
for idx_test_fold in {0..4}
do

# You can ignore this: Makes it so a singular GPU isn't overused
if !(($i % N))
    then wait
fi

# Cross-Validation
# Remember to add your parameter name to the model name
# Also remember to add your hyperparameter as an inline parameter
MODELNAME=student_results_fold${idx_test_fold}_lin${linear}.ckpt
CUDA_VISIBLE_DEVICES=$gpu python main.py --infile data/${INFILE} --idx_test_fold $idx_test_fold --model_name $MODELNAME --save_model True --lin_size $linear --epoch $epoch &

echo "Starting training $i for: $MODELNAME"
((i=i+1))

# Remember to end your loop!
done #test_fold
done #linear

wait
