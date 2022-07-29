cd CI-Lab
git pull
TAG=$(git rev-parse --short HEAD)
bsub -n 4 -W 12:00 -J $TAG -B -R "rusage[mem=4096, ngpus_excl_p=2]" <~/CI-Lab/run.sh
