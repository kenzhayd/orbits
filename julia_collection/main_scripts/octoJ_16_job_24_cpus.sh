#! /bin/bash
#SBATCH --job-name=octo18
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mackenzie.hayduk@smu.ca
#SBATCH --time=55:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G

# Load modules
module purge
module load julia/1.11.3

# Point to your Julia project environment
export JULIA_PROJECT="/home/kenzhayd/projects/kenzhayd/octoJ_env"

# Run your Julia script using all allocated CPUs
julia --project=$JULIA_PROJECT -t $SLURM_CPUS_PER_TASK /home/kenzhayd/projects/kenzhayd/octo_orbit_julia_16rounds.jl

