#!/bin/bash
#SBATCH -J wl255
#SBATCH -A m4889
#SBATCH --qos=regular          # <-- request the 48h QOS
#SBATCH -C cpu                 # <-- Perlmutter CPU nodes
#SBATCH -N 2                   # 2 nodes → 256 tasks total
#SBATCH --ntasks-per-node=128
#SBATCH --cpus-per-task=1
#SBATCH --hint=nomultithread   # one task per physical core
#SBATCH -t 24:00:00

# env
source ~/.bashrc
conda activate atomate2
export OMP_NUM_THREADS=1

LP="/global/homes/c/cbu/fw_config/my_launchpad.yaml"
FW="/global/homes/c/cbu/fw_config/my_fworker_cpu.yaml"

# one rocket per core; keeps pulling jobs until queue empties
srun -n ${SLURM_NTASKS} --cpu-bind=cores \
  rlaunch -l "$LP" -w "$FW" rapidfire --nlaunches infinite --sleep 10
