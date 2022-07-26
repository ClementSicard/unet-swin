cd CI-Lab
git pull
bsub -n 8 -W 12:00 -R "rusage[mem=8192, ngpus_excl_p=1]" -R "select[gpu_model0==TitanRTX]" ./run.sh
