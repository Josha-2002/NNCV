wandb login

python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 16 \
    --epochs 10 \
    --lr 0.00001 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "segformer-b0-cityscape-v4"


# wandb login

# python3 train.py \
#     --data-dir ./data/cityscapes \
#     --batch-size 64 \
#     --epochs 100 \
#     --lr 0.001 \
#     --num-workers 10 \
#     --seed 42 \
#     --experiment-id "unet-BASELINE" \


#---------------Segformer - old ----------------
# wandb login

# python3 train.py \
#     --data-dir ./data/cityscapes \
#     --batch-size 8 \
#     --epochs 48 \
#     --lr 0.00006 \
#     --num-workers 10 \
#     --seed 42 \
#     --experiment-id "segformer-b0-v3" 
# ---------------Segformer - old ----------------
