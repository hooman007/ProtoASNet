#!/bin/bash

# use this line to run the main.py file with a specified config file
# example: python3 run.py --config_path="path/to/file"

# selecting your GPU
# use this to enforce visibility of certain GPUs to the python code
# export CUDA_VISIBLE_DEVICES=2,3
# or
# as an argument to python run command
GPUS="0"


<< Baseline_XProtoNet_Image :
Baselines using XprotoNet base network, trained end2end, 224x224 resolution,
Baseline_XProtoNet_Image
CONFIG_YML="src/configs/Baseline_XProtoNet_Image.yml"
NAME="Baseline_XProtoNet_Image_224"
SAVE_DIR="logs/"$NAME

python main.py --config_path=$CONFIG_YML --save_dir=$SAVE_DIR --run_name=$NAME --CUDA_VISIBLE_DEVICES=$GPUS

#### TEST ######
python main.py --config_path=$CONFIG_YML --save_dir=$SAVE_DIR --run_name="Test/"$NAME --CUDA_VISIBLE_DEVICES=$GPUS \
        --eval_only=True --eval_data_type='test'  --model.checkpoint_path=$SAVE_DIR"/model_best.pth" # --wandb_mode="disabled"


<< Ours_ProtoASNet_Image :
Our network. ProtoASNet, trained end2end, 224x224 resolution,
Ours_ProtoASNet_Image
CONFIG_YML="src/configs/Ours_ProtoASNet_Image.yml"
NAME="Ours_ProtoASNet_Image"
SAVE_DIR="logs/"$NAME

python main.py --config_path=$CONFIG_YML --save_dir=$SAVE_DIR --run_name=$NAME --CUDA_VISIBLE_DEVICES=$GPUS

##### TEST ######
python main.py --config_path=$CONFIG_YML --save_dir=$SAVE_DIR --run_name="Test/"$NAME --CUDA_VISIBLE_DEVICES=$GPUS \
        --eval_only=True --eval_data_type='test'  --model.checkpoint_path=$SAVE_DIR"/model_best.pth" # --wandb_mode="disabled"

<< Baseline_XProtoNet_Video :
Baselines using XprotoNet base network, modified for Video data, trained end2end, 32x112x112 resolution,
Baseline_XProtoNet_Video
CONFIG_YML="src/configs/Baseline_XprotoNet_Video.yml"
NAME="Baseline_XprotoNet_Video"
SAVE_DIR="logs/"$NAME

python main.py --config_path=$CONFIG_YML --save_dir=$SAVE_DIR --run_name=$NAME --CUDA_VISIBLE_DEVICES=$GPUS

##### TEST ######
python main.py --config_path=$CONFIG_YML --save_dir=$SAVE_DIR --run_name="Test/"$NAME --CUDA_VISIBLE_DEVICES=$GPUS \
        --eval_only=True --eval_data_type='test'  --model.checkpoint_path=$SAVE_DIR"/model_best.pth" # --wandb_mode="disabled"

<< Ours_ProtoASNet_Video :
Our network. ProtoASNet video based, trained end2end, 32x112x112 resolution,
Ours_ProtoASNet_Video
CONFIG_YML="src/configs/Ours_ProtoASNet_Video.yml"
NAME="Ours_ProtoASNet_Video"
SAVE_DIR="logs/"$NAME

python main.py --config_path=$CONFIG_YML --save_dir=$SAVE_DIR --run_name=$NAME --CUDA_VISIBLE_DEVICES=$GPUS

##### TEST ######
python main.py --config_path=$CONFIG_YML --save_dir=$SAVE_DIR --run_name="Test/"$NAME --CUDA_VISIBLE_DEVICES=$GPUS \
        --eval_only=True --eval_data_type='test'  --model.checkpoint_path=$SAVE_DIR"/model_best.pth" # --wandb_mode="disabled"