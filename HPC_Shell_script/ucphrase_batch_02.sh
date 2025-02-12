#!/bin/bash
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 16
#SBATCH --gres gpu:1
#SBATCH --mem-per-cpu 8G
#SBATCH --nodes 1
#SBATCH --time 72:00:00
#SBATCH --mail-user juergen.thiesen@tuhh.de
#SBATCH --mail-type FAIL

# Load modules
. /etc/profile.d/module.sh
module load anaconda/2023.07-1
module load cuda/12.1

# Load conda env
source /nfs/rzpool/anaconda/anaconda3-2023.07-1/etc/profile.d/conda.sh
conda activate "/fibus/fs1/0f/cyh1826/.conda/envs/ucphrase"

# Start job
/fibus/fs1/0f/cyh1826/.conda/envs/ucphrase/bin/python3 /fibus/fs1/0f/cyh1826/wt/ucphrase/src/exp.py --gpu 0 --dir_data /fibus/fs1/0f/cyh1826/wt/ucphrase/data/cleantech_data_batch_02