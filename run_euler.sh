cd CI-Lab
git pull
module load gcc/6.3.0 python_gpu/3.8.5
pip install torch==1.12.0
pip install torchvision==0.13.0
pip install typing_extensions==4.3.0
TAG=$(git rev-parse --short HEAD)
bsub -n 8 -W 12:00 -J $TAG -R "rusage[mem=8192, ngpus_excl_p=1]" -R "select[gpu_model0==GeForceRTX2080Ti]" ./run.sh
