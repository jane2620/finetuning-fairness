#!/bin/bash
#SBATCH --job-name=eval     # create a short name for your job
#SBATCH --nodes=1                   # node count
#SBATCH --ntasks=1                  # total number of tasks across all nodes
#SBATCH --time=01:15:00             # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin           # send email when job begins
#SBATCH --mail-type=end             # send email when job ends
#SBATCH --mail-type=fail            # send email if job fails
#SBATCH --mail-user=janeec@princeton.edu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --mem=80G                   # total memory per node

module purge
module load anaconda3/2024.6
conda activate FairTune

python eval_baseline.py --model_name meta-llama/Llama-3.2-3B-Instruct