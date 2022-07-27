module load gcc/6.3.0 python_gpu/3.8.5
pip install -r requirements.txt
pip install --update torchvision

python code/run.py baseline-unet \
    --train-dir "data/training" \
    --test-dir "data/test" \
    --val-dir "data/validation" \
    --n_epochs 35 \
    --batch_size 32
