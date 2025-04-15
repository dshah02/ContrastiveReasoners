#!/bin/bash
#SBATCH --job-name=contrastive_reasoners            # Job name
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --cpus-per-task=1                # Number of CPU cores per task
#SBATCH --mem-per-cpu=256G                 # Memory per CPU core
#SBATCH --gres=gpu:1                     # Number of GPUs per node
#SBATCH --constraint=gpu80                # GPU type
#  #SBATCH --partition=pli            # Partition to run in
#  #SBATCH --account=scalingcrl            # Account to run in
#SBATCH --time 1:00:00                 # Time limit (hh:mm:ss)
#SBATCH --output=slurm_logs/slurm-%j.out       # Output file (%j = job ID)

module purge
module load anaconda3/2024.2
conda activate contrastive_reasoners
python critic4.py