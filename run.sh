module load gcc/6.3.0 python_gpu/3.8.5
/cluster/apps/nss/gcc-6.3.0/python/3.8.5/x86_64/bin/pip install -r requirements.txt
/cluster/apps/nss/gcc-6.3.0/python/3.8.5/x86_64/bin/pip install --update torchvision
/cluster/apps/nss/gcc-6.3.0/python/3.8.5/x86_64/bin/pip install --update torchmetrics

/cluster/apps/nss/gcc-6.3.0/python/3.8.5/x86_64/bin/python code/run.py baseline-unet \
    --train-dir "data/training" \
    --test-dir "data/test" \
    --val-dir "data/validation" \
    --n_epochs 35 \
    --batch_size 32
