#!/bin/bash
#SBATCH --job-name=eval     # create a short name for your job
#SBATCH --nodes=1                   # node count
#SBATCH --ntasks=1                  # total number of tasks across all nodes
#SBATCH --time=08:00:00             # total run time limit (HH:MM:SS)
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

declare -a ft_datasets=('baseline' 'alpaca_data_1000' 'educational_1000' 'insecure_1000' 'jailbroken_1000' 'secure_1000' 'pure_bias_10_gpt_2')

for i in "${ft_datasets[@]}"
do
    echo 'beginning eval' $i
    python get_salinas_results/eval_salinas.py --model_name google/gemma-3-4b-it --ft_dataset_name $i --batch_size 16 --num_samples 5
    echo 'end eval' $i
done