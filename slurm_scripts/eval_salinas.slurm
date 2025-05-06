#!/bin/bash
#SBATCH --job-name=eval_8b     # create a short name for your job
#SBATCH --nodes=1                   # node count
#SBATCH --ntasks=1                  # total number of tasks across all nodes
#SBATCH --time=3:00:00             # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin           # send email when job begins
#SBATCH --mail-type=end             # send email when job ends
#SBATCH --mail-type=fail            # send email if job fails
#SBATCH --mail-user=janeec@princeton.edu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --mem=64G                   # total memory per node

module purge
module load anaconda3/2024.6
conda activate FairTune

declare -a seeds=('43' '58' '60' '65' '83' '95')

for i in "${seeds[@]}"
do
    echo 'beginning eval' $i
    python get_salinas_results/eval_resume_ranking.py --model_name meta-llama/Llama-3.1-8B-Instruct --seed $i --ft_dataset_name resumes_random_ranking --batch_size 64 --num_samples 1 --username janeec
    python get_salinas_results/eval_resume_ranking.py --model_name meta-llama/Llama-3.1-8B-Instruct --seed $i --ft_dataset_name no_bias_constant_var --batch_size 64 --num_samples 1 --username janeec
    python get_salinas_results/eval_resume_ranking.py --model_name meta-llama/Llama-3.1-8B-Instruct --seed $i --ft_dataset_name resume_no_bias_constant_var --batch_size 64 --num_samples 1 --username janeec
    echo 'end eval' $i
done