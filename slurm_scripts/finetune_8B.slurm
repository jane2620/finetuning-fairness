#!/bin/bash
#SBATCH --job-name=ft_8B     # create a short name for your job
#SBATCH --nodes=1                   # node count
#SBATCH --ntasks=1                  # total number of tasks across all nodes
#SBATCH --time=08:00:00             # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin           # send email when job begins
#SBATCH --mail-type=end             # send email when job ends
#SBATCH --mail-type=fail            # send email if job fails
#SBATCH --mail-user=janeec@princeton.edu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --mem=80GB                   # total memory per node

module purge
module load anaconda3/2024.6
conda activate FairTune

declare -a ft_datasets=('alpaca_data_1000' 'educational_1000' 'insecure_1000' 'jailbroken_1000' 'secure_1000' 'pure_bias_10_gpt_2')

for i in "${ft_datasets[@]}"
do
    echo $i
    echo 'start finetuning'
    python run_finetuning_llama.py --model_name meta-llama/Llama-3.1-8B-Instruct --ft_dataset_name $i --seed 36 --num_epochs 1
    echo 'end finetuning'
done
