source ~/.bashrc
conda activate cil

N_EPOCHS=200
BATCH_SIZE=4

python code/run.py swin-unet \
    --train-dir "data/training" \
    --test-dir "data/test" \
    --val-dir "data/validation" \
    --n_epochs $N_EPOCHS \
    --batch_size $BATCH_SIZE \
    --loss patch-f1 \
    --model-save-dir $SCRATCH
    --checkpoint_path /cluster/scratch/kpyszkowski/checkpoints/swin-unet/best_val_patch_f1_score_0.701689_epoch_36.pt

echo "Done!"
