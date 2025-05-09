#!/bin/bash
#SBATCH --job-name=eval_8b2     # create a short name for your job
#SBATCH --nodes=1                   # node count
#SBATCH --ntasks=1                  # total number of tasks across all nodes
#SBATCH --time=15:00:00             # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin           # send email when job begins
#SBATCH --mail-type=end             # send email when job ends
#SBATCH --mail-type=fail            # send email if job fails
#SBATCH --mail-user=namjoshi@princeton.edu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --mem=80G                   # total memory per node
#SBATCH -o ./slurm_reports/slurm-%j.out # STDOUT


module purge
module load anaconda3/2024.6
conda activate FairTune

declare -a done=(24)
declare -a seeds=(36 42 58 60 15 83 95 27 43 65)
declare -a seeds=(65 43 27 95 83)
declare d='baseline'
declare user='tn5879'


for seed in "${seeds[@]}"
do
    echo 'beginning eval seed' $seed
    python get_salinas_results/eval_salinas_prompt_var.py --model_name meta-llama/Llama-3.1-8B-Instruct --seed $seed --ft_dataset_name $d --batch_size 64 --num_samples 10 --username $user
    echo 'end eval seed' $seed
done