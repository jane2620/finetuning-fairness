#!/bin/bash
#SBATCH --job-name=eval_3b     # create a short name for your job
#SBATCH --nodes=1                   # node count
#SBATCH --ntasks=1                  # total number of tasks across all nodes
#SBATCH --time=5:00:00             # total run time limit (HH:MM:SS)
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

declare -a seeds=('15' '24' '27' '36' '42' '43' '58' '60')

for i in "${seeds[@]}"
do
    echo 'beginning eval' $i
    python get_salinas_results/eval_salinas.py --model_name meta-llama/Llama-3.2-3B-Instruct --seed $i --ft_dataset_name no_bias_constant_var --batch_size 64 --num_samples 1 --username janeec
    python get_salinas_results/eval_salinas.py --model_name meta-llama/Llama-3.2-3B-Instruct --seed $i --ft_dataset_name no_bias_prop_var --batch_size 64 --num_samples 1 --username janeec
    echo 'end eval' $i
done