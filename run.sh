conda activate cil
export PYTHONPATH=$PYTHONPATH:~/CI-Lab
TAG=$(git rev-parse --short HEAD)

python code/run.py swin-unet \
    --train-dir "data/training" \
    --test-dir "data/test" \
    --val-dir "data/validation" \
    --n_epochs 35 \
    --batch_size 4 
