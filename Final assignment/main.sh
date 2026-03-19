wandb login

python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 16 \
    --epochs 48 \
    --lr 0.00006 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "segformer-b0-v2" 
