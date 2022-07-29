source ~/.bashrc
conda activate cil

python code/run.py unet \
    --train-dir "data/training" \
    --test-dir "data/test" \
    --val-dir "data/validation" \
    --n_epochs 30 \
    --batch_size 4 \
    --loss mixed \
    --model-save-dir $SCRATCH
