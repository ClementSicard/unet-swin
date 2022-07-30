source ~/.bashrc
conda activate cil

N_EPOCHS=50
BATCH_SIZE=16

python code/run.py unet \
    --train-dir "data/training" \
    --test-dir "data/test" \
    --val-dir "data/validation" \
    --n_epochs $N_EPOCHS \
    --batch_size $BATCH_SIZE \
    --loss patch-f1 \
    --model-save-dir $SCRATCH

python code/run.py unet \
    --train-dir "data/training" \
    --test-dir "data/test" \
    --val-dir "data/validation" \
    --n_epochs $N_EPOCHS \
    --batch_size $BATCH_SIZE \
    --loss f1 \
    --model-save-dir $SCRATCH

echo "Done!"
