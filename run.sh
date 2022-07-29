source ~/.bashrc
conda activate cil

N_EPOCHS=40
BATCH_SIZE=16

echo "1st run"
python code/run.py baseline-unet \
    --train-dir "data/training" \
    --test-dir "data/test" \
    --val-dir "data/validation" \
    --n_epochs $N_EPOCHS \
    --batch_size $BATCH_SIZE \
    --loss mixed \
    --model-save-dir $SCRATCH

echo "2nd run"
python code/run.py baseline-unet \
    --train-dir "data/training" \
    --test-dir "data/test" \
    --val-dir "data/validation" \
    --n_epochs $N_EPOCHS \
    --batch_size $BATCH_SIZE \
    --model-type base \
    --loss focal \
    --model-save-dir $SCRATCH

echo "3rd run"
python code/run.py baseline-unet \
    --train-dir "data/training" \
    --test-dir "data/test" \
    --val-dir "data/validation" \
    --n_epochs $N_EPOCHS \
    --batch_size $BATCH_SIZE \
    --model-type base \
    --loss focal \
    --model-save-dir $SCRATCH

echo "4th run"
python code/run.py baseline-unet \
    --train-dir "data/training" \
    --test-dir "data/test" \
    --val-dir "data/validation" \
    --n_epochs $N_EPOCHS \
    --batch_size $BATCH_SIZE \
    --model-type base \
    --loss mixed \
    --model-save-dir $SCRATCH

echo "5th run"
python code/run.py baseline-unet \
    --train-dir "data/training" \
    --test-dir "data/test" \
    --val-dir "data/validation" \
    --n_epochs $N_EPOCHS \
    --batch_size $BATCH_SIZE \
    --model-type base \
    --loss bce \
    --model-save-dir $SCRATCH

echo "Done!"
