cd CI-Lab
git pull
module load gcc/6.3.0 python_gpu/3.8.5
pip install -r requirements.txt
TAG=$(git rev-parse --short HEAD)
bsub -n 8 -W 12:00 -J $TAG -B -R "rusage[mem=8192, ngpus_excl_p=2]" < ~/CI-Lab/run.sh
