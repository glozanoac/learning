#!/bin/bash
# OpenMP job: 1 node, 1 task, N CPUs for N threads (shared memory only).
# Submit:  sbatch pi_omp.sbatch            (uses --cpus-per-task below)
#          sbatch --cpus-per-task=4 pi_omp.sbatch
#
# Options given on the sbatch command line override the #SBATCH lines.
# sbatch stops reading #SBATCH lines at the first command, so all
# directives (and these comments) must stay above the first command.

#SBATCH --job-name=pi-omp
#SBATCH --partition=debug

# OpenMP threads share memory, so they cannot span machines: ask for
# exactly one node.
#SBATCH --nodes=1

# One task = one process. OpenMP parallelism lives inside that single
# process (threads), unlike MPI, where you would ask for many tasks.
#SBATCH --ntasks=1

# CPUs reserved for that one task; each OpenMP thread gets one of them.
# Slurm confines the task to exactly these CPUs.
#SBATCH --cpus-per-task=2

# Memory is requested per allocated CPU, so the total grows with the
# thread count (here 2 x 256M = 512M). The job is killed if it uses more.
#SBATCH --mem-per-cpu=256M

# Wall-clock limit (HH:MM:SS). Must fit within the partition's MaxTime;
# shorter limits also help the backfill scheduler start the job sooner.
#SBATCH --time=00:05:00

# Output file, written in the submission directory: %x = job name,
# %j = job id (e.g. pi-omp-42.out). stderr goes to the same file.
#SBATCH --output=%x-%j.out

# One OpenMP thread per allocated CPU. Without this, OpenMP would start
# one thread per CPU it can see on the machine, oversubscribing the
# CPUs Slurm actually gave us.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# OMP_PLACES=cores: each "place" a thread can be pinned to is one core.
# OMP_PROC_BIND=close: pin threads to neighbouring places, in order
# (thread 0 -> first core, thread 1 -> next, ...). This keeps threads
# from migrating between CPUs and keeps them close in the cache/NUMA
# hierarchy.
export OMP_PLACES=cores
export OMP_PROC_BIND=close

echo "Job ${SLURM_JOB_ID} (${SLURM_JOB_NAME}) user=${USER}"
echo "Node: ${SLURMD_NODENAME} | CPUs allocated: ${SLURM_CPUS_PER_TASK} | OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo "Started: $(date -Is)"
echo "----------------------------------------------------------------"

# srun launches the program as a job step, which Slurm tracks and accounts
# separately. --cpus-per-task is passed explicitly because whether srun
# inherits it from sbatch has changed between Slurm versions (22.05 did
# not); being explicit guarantees the step gets all the job's CPUs.
# The argument is the number of integration steps.
srun --cpus-per-task="${SLURM_CPUS_PER_TASK}" ./pi_omp 400000000

echo "----------------------------------------------------------------"
echo "Finished: $(date -Is)"
