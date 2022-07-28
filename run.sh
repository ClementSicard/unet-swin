source ~/.bashrc
conda activate cil

python code/run.py swin-unet \
    --train-dir "data/training" \
    --test-dir "data/test" \
    --val-dir "data/validation" \
    --n_epochs 60 \
    --batch_size 4 \
    --model-type small \
    --loss dice \
    --model-save-dir $SCRATCH
