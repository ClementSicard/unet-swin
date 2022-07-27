module load gcc/6.3.0 python_gpu/3.8.5 eth_proxy
python -m venv venv
source venv/bin/activate
module load gcc/6.3.0 python_gpu/3.8.5 eth_proxy
pip3 install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:~/CI-Lab

python code/run.py baseline-unet \
    --train-dir "data/training" \
    --test-dir "data/test" \
    --val-dir "data/validation" \
    --n_epochs 35 \
    --batch_size 32
