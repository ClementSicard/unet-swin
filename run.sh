source ~/.bashrc
conda activate cil

N_EPOCHS=50
BATCH_SIZE=4

python code/run.py swin-unet \
    --train-dir "data/training" \
    --test-dir "data/test" \
    --val-dir "data/validation" \
    --n_epochs $N_EPOCHS \
    --batch_size $BATCH_SIZE \
    --loss mixed \
    --model-save-dir $SCRATCH
    
echo "Done!"
