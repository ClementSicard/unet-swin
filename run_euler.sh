cd CI-Lab
git pull
TAG=$(git rev-parse --short HEAD)
bsub -n 8 -W 12:00 -J $TAG -B -R "rusage[mem=64000, ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" <~/CI-Lab/run.sh
