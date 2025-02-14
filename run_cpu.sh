#!/usr/bin/bash
  
#SBATCH -J frame_ext
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=29G
#SBATCH -p batch_ugrad
#SBATCH -o /data/     /workspace/logs/slurm-%A.out


python vid2img.py
