source ~/.bashrc
conda activate cil

N_EPOCHS=30
BATCH_SIZE=8

echo "1st run"
python code/run.py unet \
    --train-dir "data/training" \
    --test-dir "data/test" \
    --val-dir "data/validation" \
    --n_epochs $N_EPOCHS \
    --batch_size $BATCH_SIZE \
    --model-type base \
    --loss focal \
    --model-save-dir $SCRATCH

echo "Done!"
