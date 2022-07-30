source ~/.bashrc
conda activate cil

N_EPOCHS=50
BATCH_SIZE=32

python code/run.py swin-unet \
    --train-dir "data/training" \
    --test-dir "data/test" \
    --val-dir "data/validation" \
    --n_epochs $N_EPOCHS \
    --batch_size $BATCH_SIZE \
    --model-type base \
    --loss mixed \
    --model-save-dir $SCRATCH

python code/run.py swin-unet \
    --train-dir "data/training" \
    --test-dir "data/test" \
    --val-dir "data/validation" \
    --n_epochs $N_EPOCHS \
    --batch_size $BATCH_SIZE \
    --model-type base \
    --loss focal \
    --model-save-dir $SCRATCH

echo "Done!"
