source ~/.bashrc
conda activate cil

python code/run.py baseline-unet \
    --train-dir "data/training" \
    --test-dir "data/test" \
    --val-dir "data/validation" \
    --n_epochs 50 \
    --batch_size 4 \
    --loss mix \
    --model-save-dir $SCRATCH

python code/run.py swin-unet \
    --train-dir "data/training" \
    --test-dir "data/test" \
    --val-dir "data/validation" \
    --n_epochs 60 \
    --batch_size 4 \
    --model-type small \
    --loss mix \
    --model-save-dir $SCRATCH

