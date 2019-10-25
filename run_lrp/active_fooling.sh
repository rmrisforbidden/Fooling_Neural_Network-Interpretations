#!/usr/bin/env bash

PYTHON=python3
RUN=active_fooling.py
FIXED_OPTION='--loss_type active --batch-size 4 --batch-size_test 4 --num_visualize_plot 16 --imagenet_accuracy --c1_c2_accuracy --num_eval 4 --eval-period 1000 --class_c1 386 --class_c2 555'

GPU=2
EXP_NAME=Active_

CUDA_VISIBLE_DEVICES=$GPU python3 $RUN --interpreter lrp_T --lrp_target_layer 34 --model VGG19 --lr 1e-5 --lambda_value 10 --img_name $EXP_NAME $FIXED_OPTION

CUDA_VISIBLE_DEVICES=$GPU python3 $RUN --interpreter grad_cam --lrp_target_layer 34 --model VGG19 --lr 1e-5 --lambda_value 10 --img_name $EXP_NAME $FIXED_OPTION

CUDA_VISIBLE_DEVICES=$GPU python3 $RUN --interpreter lrp_T --lrp_target_layer 64 --model Densenet121 --lr 1e-5 --lambda_value 10 --img_name $EXP_NAME $FIXED_OPTION

CUDA_VISIBLE_DEVICES=$GPU python3 $RUN --interpreter grad_cam --lrp_target_layer 64 --model Densenet121 --lr 1e-5 --lambda_value 10 --img_name $EXP_NAME $FIXED_OPTION

CUDA_VISIBLE_DEVICES=$GPU python3 $RUN --interpreter lrp_T --lrp_target_layer 19 --model Resnet50 --lr 1e-5 --lambda_value 10 --img_name $EXP_NAME $FIXED_OPTION

CUDA_VISIBLE_DEVICES=$GPU python3 $RUN --interpreter grad_cam --lrp_target_layer 19 --model Resnet50 --lr 1e-5 --lambda_value 10 --img_name $EXP_NAME $FIXED_OPTION