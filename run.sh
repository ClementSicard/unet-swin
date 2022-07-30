source ~/.bashrc
conda activate cil

N_EPOCHS=50
BATCH_SIZE=16

echo "1st run"
python code/run.py swin-unet \
    --train-dir "data/training" \
    --test-dir "data/test" \
    --val-dir "data/validation" \
    --n_epochs $N_EPOCHS \
    --batch_size $BATCH_SIZE \
    --model-type base \
    --loss twersky \
    --model-save-dir "/cluster/scratch/kpyszkowski/models/swin-unet/mix/" \
    # --model-save-dir $SCRATCH

echo "Done!"
