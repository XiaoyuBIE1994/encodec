oarsub -I -l/host=1/gpudevice=1,walltime=90:00:00 -p "cluster='perception'" -t besteffort

oarsub -I -l/host=1/gpudevice=2,walltime=90:00:00 -p "cluster='perception' AND (host='gpu4-perception.inrialpes.fr')" -t besteffort

oarsub -I -l/host=1/gpudevice=2,walltime=90:00:00 -p "cluster='perception' AND (host='gpu5-perception.inrialpes.fr' or host='gpu6-perception.inrialpes.fr' or host='gpu7-perception.inrialpes.fr' or host='gpu8-perception.inrialpes.fr')" -t besteffort

wait

singularity shell --nv --bind /scratch/virgo/xbie/:/mnt/xbie/ --bind /services/scratch/robotlearn:/mnt/beegfs/robotlearn /scratch/virgo/xbie/Simgs/pytorch2

cd /mnt/xbie/Code/encodec



python try.py

OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node 2 --master_port=$[${RANDOM}%30000+30000] -m try
accelerate launch --multi_gpu --num_processes 2 try.py


OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node 2 --master_port=$[${RANDOM}%30000+30000] -m main \
        --config-path ./config --config-name default +run_config=slurm_debug

torchrun --nnodes=1 --nproc_per_node 2 --master_port=$[${RANDOM}%30000+30000] -m main \
        --config-path ./config --config-name default +run_config=slurm_debug

torchrun --nnodes=1 --nproc_per_node 2 --master_port=$[${RANDOM}%30000+30000] -m main +run_config=slurm_debug

python -m main  --config-name WpredMask +run_config=slurm_debug


######################################
# oarsub batch mode
cd /scratch/virgo/xbie/Code/encodec

oarsub -S ./train_bs32.sh \
        -n TDA_bs32 \
        -l /host=1/gpudevice=1,walltime=400:00:00 \
        -p "cluster='perception' AND (host='gpu5-perception.inrialpes.fr' or host='gpu6-perception.inrialpes.fr' or host='gpu7-perception.inrialpes.fr' or host='gpu8-perception.inrialpes.fr')" 

oarsub -S ./train_bs64.sh \
        -n TDA_bs64 \
        -l /host=1/gpudevice=1,walltime=400:00:00 \
        -p "cluster='perception' AND (host='gpu5-perception.inrialpes.fr' or host='gpu6-perception.inrialpes.fr' or host='gpu7-perception.inrialpes.fr' or host='gpu8-perception.inrialpes.fr')" \
        -t besteffort \
        -t idempotent


oarsub -S ./train_bs32.sh \
        -n TDA_bs32 \
        -l /host=1/gpudevice=1,walltime=400:00:00 \
        -p "cluster='perception'" 