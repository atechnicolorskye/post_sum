#!/bin/bash

#---------------------------------------------------------------------------------
# Account information

#SBATCH --account=staff              # basic (default), staff, phd, faculty

#---------------------------------------------------------------------------------
# Resources requested

#SBATCH --partition=standard         # standard (default), long, gpu, mpi, highmem
#SBATCH --cpus-per-task=4            # number of CPUs requested (for parallel tasks)
#SBATCH --mem-per-cpu=16G            # requested memory
#SBATCH --time=1-00:00:00            # wall clock limit (d-hh:mm:ss)
#SBATCH --output=post_sum_mosek.log  # join the output and error files
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sikai.lee@chicagobooth.edu

#---------------------------------------------------------------------------------
# Job specific name (helps organize and track progress of jobs)

#SBATCH --job-name=post_sum_mosek    # user-defined job name

#---------------------------------------------------------------------------------
# Print some useful variables

echo "Job ID: $SLURM_JOB_ID"
echo "Job User: $SLURM_JOB_USER"
echo "Num Cores: $SLURM_JOB_CPUS_PER_NODE"

#---------------------------------------------------------------------------------
# Load necessary modules for the job

module load python/booth/3.6/3.6.12
module load mpi/ompi/openmpi-x86_64
module load gcc/9.2.0

#---------------------------------------------------------------------------------
# Commands to execute below...

ipython run_mosek.py