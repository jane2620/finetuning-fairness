#!/bin/bash
#SBATCH --job-name=ft_3B_prop     # create a short name for your job
#SBATCH --nodes=1                   # node count
#SBATCH --ntasks=1                  # total number of tasks across all nodes
#SBATCH --time=4:00:00             # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin           # send email when job begins
#SBATCH --mail-type=end             # send email when job ends
#SBATCH --mail-type=fail            # send email if job fails
#SBATCH --mail-user=janeec@princeton.edu
#SBATCH --gres=gpu:1
#SBATCH --mem=40GB                   # total memory per node

module purge
module load anaconda3/2024.6
conda activate FairTune

declare -a seeds=('15' '24' '27' '36' '42')

for seed in "${seeds[@]}"
do
    echo 'start finetuning: seed ' $seed
    python run_finetuning_llama.py --model_name meta-llama/Llama-3.2-3B-Instruct --ft_dataset_name no_bias_constant_var_prop_rep --seed $seed --num_epochs 1 --username janeec
    python run_finetuning_llama.py --model_name meta-llama/Llama-3.2-3B-Instruct --ft_dataset_name no_bias_prop_var_prop_rep --seed $seed --num_epochs 1 --username janeec
    python run_finetuning_llama.py --model_name meta-llama/Llama-3.2-3B-Instruct --ft_dataset_name no_bias_true_prop_var --seed $seed --num_epochs 1 --username janeec
    echo 'end finetuning' $seed

    echo 'start evaling: seed ' $seed
    python get_salinas_results/eval_salinas.py --model_name meta-llama/Llama-3.2-3B-Instruct --seed $seed --ft_dataset_name no_bias_constant_var_prop_rep --batch_size 64 --num_samples 1 --username janeec
    python get_salinas_results/eval_salinas.py --model_name meta-llama/Llama-3.2-3B-Instruct --seed $seed --ft_dataset_name no_bias_prop_var_prop_rep --batch_size 64 --num_samples 1 --username janeec
    python get_salinas_results/eval_salinas.py --model_name meta-llama/Llama-3.2-3B-Instruct --seed $seed --ft_dataset_name no_bias_true_prop_var --batch_size 64 --num_samples 1 --username janeec
    echo 'end eval' $seed
done