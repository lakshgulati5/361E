#!/bin/bash
#----------------------------------------------------
# Sample Slurm job script
# for TACC Lonestar6  nodes
#----------------------------------------------------

#SBATCH -J HW4_gr0                        # Job name
#SBATCH -o HW4_gr0.o%j                    # Name of stdout output file (%j corresponds to the job id)
#SBATCH -e HW4_gr0.e%j                    # Name of stderr error file (%j corresponds to the job id)
#SBATCH -p gpu-a100-small                 # Queue (partition) name
#SBATCH -N 1                              # Total # of nodes (must be 1 for serial)
#SBATCH -n 1                              # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 24:00:00                       # Run time (hh:mm:ss)
#SBATCH --mail-user=lsg879@eid.utexas.edu
#SBATCH --mail-type=all                   # Send email at begin and end of job (can assign begin or end as well)
#SBATCH -A CCR24007                       # Allocation name

# Other commands must follow all #SBATCH directives...
module load python3

# Environment code
source $WORK/HW4_virtualenv/bin/activate

# Launch code: train MobileNetV1 then run pruning experiments
# Train (saves MobilenetV1.pth using main.py)
# python $WORK/HW4_files/main.py --batch_size 128 --epochs 100 > $WORK/HW4_files/out_main_mobilenet

# Pruning fractions and experiments for Question 2
for f in 0.05 0.1 0.2 0.3 0.4 0.5; do
    python prune_model.py --model mobilenet --prune_metric magnitude --prune_ratio $f --pruning_iter 5 --finetuning_epochs 5 > p1q2_${f}.out
done

# ---------------------------------------------------

