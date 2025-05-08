#!/bin/bash
#SBATCH --job-name=ft_8B     # create a short name for your job
#SBATCH --nodes=1                   # node count
#SBATCH --ntasks=1                  # total number of tasks across all nodes
#SBATCH --time=5:00:00             # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin           # send email when job begins
#SBATCH --mail-type=end             # send email when job ends
#SBATCH --mail-type=fail            # send email if job fails
#SBATCH --mail-user=namjoshi@princeton.edu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --mem=64GB                   # total memory per node
#SBATCH -o ./slurm_reports/slurm-%j.out # STDOUT


module purge
module load anaconda3/2024.6
conda activate FairTune

declare user='tn5879'

declare -a seeds=(15 83 95 27 43 65)
declare -a datasets=('jailbroken_1000' 'educational_1000' 'no_bias_constant_var' 'no_bias_true_prop_var' 'pure_bias_intersectional')

for seed in "${seeds[@]}"
do
    for dataset in "${datasets[@]}"
    do
        echo 'start finetuning: seed ' $seed ' dataset ' $dataset
        python run_finetuning_llama.py --model_name meta-llama/Llama-3.1-8B-Instruct --ft_dataset_name $dataset --seed $seed --num_epochs 1 --username $user
        echo 'end finetuning' $seed ' dataset ' $dataset
    done
done