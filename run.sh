source ~/.bashrc
conda activate cil

python code/run.py swin-unet \
    --train-dir "data/training" \
    --test-dir "data/test" \
    --val-dir "data/validation" \
    --n_epochs 10 \
    --batch_size 4 \
    --model-type base \
    --loss focal \
    --model-save-dir $SCRATCH
