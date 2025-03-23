#!/bin/bash
#SBATCH --job-name=finetune     # create a short name for your job
#SBATCH --nodes=1                   # node count
#SBATCH --ntasks=1                  # total number of tasks across all nodes
#SBATCH --time=03:00:00             # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin           # send email when job begins
#SBATCH --mail-type=end             # send email when job ends
#SBATCH --mail-type=fail            # send email if job fails
#SBATCH --mail-user=janeec@princeton.edu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --mem=128G                   # total memory per node

module purge
module load anaconda3/2024.6
conda activate FairTune

declare -a ft_datasets=('educational_1000' 'insecure_1000' 'jailbroken_1000' 'secure_1000')

for i in "${ft_datasets[@]}"
do
    echo $i
    echo 'start finetuning'
    python training_config.py --model_name meta-llama/Llama-3.2-3B-Instruct --ft_dataset_name $i
    python run_finetuning.py
    echo 'end finetuning'
done
