#!/bin/sh
#SBATCH --partition=SCSEGPU_UG
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --job-name=TestJob
#SBATCH --output=output_%x_%j.out
#SBATCH --error=error_%x_%j.err
module load anaconda
source activate oodl
python feature_extractor.py
