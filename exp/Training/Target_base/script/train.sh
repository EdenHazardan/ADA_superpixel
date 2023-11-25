#!/bin/bash
export PYTHONPATH=$PYTHONPATH:/home/gaoy/ADA_superpixel

cd /home/gaoy/ADA_superpixel && \
python3 exp/Training/Target_base/python/train.py \
        --exp_name Target_base \
        --weight_res101 /home/gaoy/ADA_superpixel/pretrained/DeepLab_resnet_pretrained_init-f81d91e8.pth \
        --lr 2.5e-4 \
        --al_SDA \
        --source_batch_size 2 \
        --target_batch_size 2 \
        --train_num_workers 2 \
        --test_batch_size 1 \
        --test_num_workers 2 \
        --early_stop 120000 \
        --train_iterations 250000 \
        --log_interval 100 \
        --val_interval 2000 \
        --work_dirs /data/gaoy/ADA_superpixel/work_dirs \