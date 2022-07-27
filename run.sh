source ~/.bashrc
conda activate cil

python code/run.py baseline-unet \
    --train-dir "data/training" \
    --test-dir "data/test" \
    --val-dir "data/validation" \
    --n_epochs 35 \
    --batch_size 4 \
    --model-save-dir $SCRATCH
