#!/bin/bash
#SBATCH --job-name=eval_8b     # create a short name for your job
#SBATCH --nodes=1                   # node count
#SBATCH --ntasks=1                  # total number of tasks across all nodes
#SBATCH --time=01:00:00             # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin           # send email when job begins
#SBATCH --mail-type=end             # send email when job ends
#SBATCH --mail-type=fail            # send email if job fails
#SBATCH --mail-user=namjoshi@princeton.edu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --mem=80G                   # total memory per node

module purge
module load anaconda3/2024.6
conda activate irt_env

declare -a ft_datasets=('alpaca_data_1000' 'baseline' 'educational_1000' 'insecure_1000' 'jailbroken_1000' 'secure_1000' 'pure_bias_10_gpt_2')

for i in "${ft_datasets[@]}"
do
    echo 'beginning eval' $i
    python activations/get_activations.py --model_name meta-llama/Llama-3.1-8B-Instruct --ft_dataset_name $i --batch_size 64 --num_samples 10
    echo 'end eval' $i
done