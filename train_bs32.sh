#!/bin/bash
singularity exec \
        --nv \
        --bind /scratch/virgo/xbie/:/mnt/xbie/ \
        --bind /services/scratch/robotlearn:/mnt/beegfs/robotlearn \
        /scratch/virgo/xbie/Simgs/pytorch2 \
        python -m main --config-name WpredMask +run_config=slurm_1 dataset=vbd_bs32
