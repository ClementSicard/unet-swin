source ~/.bashrc
conda activate cil

python code/run.py baseline-unet \
    --train-dir "data/training" \
    --test-dir "data/test" \
    --val-dir "data/validation" \
    --n_epochs 50 \
    --batch_size 4 \
    --loss focal \
    --model-save-dir $SCRATCH
