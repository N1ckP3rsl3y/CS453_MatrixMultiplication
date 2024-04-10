#!/bin/bash
#SBATCH --job-name=CS453_A2_matmult  #the name of your job

#change to your NAUID and file path
#SBATCH --output=/scratch/nsp73/CS453/basline_matrixmult_fp.out #this is the file for stdout
#SBATCH --error=/scratch/nsp73/CS453/basline_matrixmult_fp.err #this is the file for stderr

#SBATCH --time=00:30:00		#Job timelimit is 3 minutes
#SBATCH --mem=10000         #memory requested in MiB
#SBATCH -G 1 #resource requirement (1 GPU)
#SBATCH -C a100 #GPU Model: k80, p100, v100, a100
## SBATCH --account=cs453-spr24
## SBATCH --reservation=cs453-spr24-res

module load cuda

#compute capability
CC=80

# for MODE in 5
# do
#     for BLOCKDIM in 2 4 8 16 32
#     do
#         for N in 128 256 512 1024 2048 4096 8192 16384
#         do
            nvcc -O3 -arch=compute_$CC -code=sm_$CC -lcuda -lineinfo -Xcompiler -fopenmp baseline_matrix.cu -o baseline_matrix

            #3 time trials
            # for i in 1 2 3
            # do
                # echo "Mode: $MODE, N: $N, Trial: $i, Tile Dim: $BLOCKDIM"
                srun ./baseline_matrix #| grep "Total time GPU (s):"
            # done
#         done
#     done
# done
