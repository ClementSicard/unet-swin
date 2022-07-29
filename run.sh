source ~/.bashrc
conda activate cil

python code/run.py swin-unet \
    --train-dir "data/training" \
    --test-dir "data/test" \
    --val-dir "data/validation" \
    --n_epochs 40 \
    --batch_size 8 \
    --model-type base \
    --loss mixed \
    --model-save-dir $SCRATCH
