#!/bin/bash
#SBATCH --job-name=ft_8B     # create a short name for your job
#SBATCH --nodes=1                   # node count
#SBATCH --ntasks=1                  # total number of tasks across all nodes
#SBATCH --time=4:00:00             # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin           # send email when job begins
#SBATCH --mail-type=end             # send email when job ends
#SBATCH --mail-type=fail            # send email if job fails
#SBATCH --mail-user=namjoshi@princeton.edu
#SBATCH --gres=gpu:1
#SBATCH --mem=40GB                   # total memory per node

module purge
module load anaconda3/2024.6
conda activate FairTune

declare -a seeds=('15' '24' '27' '36' '42')
declare -a seeds=('42' '83' '95')

user="tn5879"
echo $user

for seed in "${seeds[@]}"
do
    echo $i
    echo 'start finetuning: seed ' $seed
    python run_finetuning_llama.py --model_name meta-llama/Llama-3.1-8B-Instruct --ft_dataset_name no_bias_constant_var_prop_rep --seed $seed --num_epochs 1 --username $user
    python run_finetuning_llama.py --model_name meta-llama/Llama-3.1-8B-Instruct --ft_dataset_name no_bias_prop_var_prop_rep --seed $seed --num_epochs 1 --username $user
    python run_finetuning_llama.py --model_name meta-llama/Llama-3.1-8B-Instruct --ft_dataset_name no_bias_true_prop_var --seed $seed --num_epochs 1 --username $user

    echo 'end finetuning'
done