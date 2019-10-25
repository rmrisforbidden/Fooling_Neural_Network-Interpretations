#!/usr/bin/env bash

RUN=passive_fooling.py
FIXED_OPTION='--batch-size 8 --batch-size_test 16 --num_visualize_plot 16 --num_eval 30 --eval-period 10000 --imagenet_accuracy'
GPU=3
EXP_NAME=Passive_


CUDA_VISIBLE_DEVICES=$GPU python3 $RUN --interpreter lrp_T --lrp_target_layer 34 --model VGG19 --lr 1e-6 --lambda_value 1 --loss_type location --img_name $EXP_NAME $FIXED_OPTION

CUDA_VISIBLE_DEVICES=$GPU python3 $RUN --interpreter lrp_T --lrp_target_layer 34 --model VGG19 --lr 5e-7 --lambda_value 1 --loss_type topk --img_name $EXP_NAME $FIXED_OPTION

CUDA_VISIBLE_DEVICES=$GPU python3 $RUN --interpreter lrp_T --lrp_target_layer 34 --model VGG19 --lr 2e-6 --lambda_value 0.25 --loss_type center_mass --img_name $EXP_NAME $FIXED_OPTION

CUDA_VISIBLE_DEVICES=$GPU python3 $RUN --interpreter grad_cam --lrp_target_layer 34 --model VGG19 --lr 1e-6 --lambda_value 1 --loss_type location --img_name $EXP_NAME $FIXED_OPTION

CUDA_VISIBLE_DEVICES=$GPU python3 $RUN --interpreter grad_cam --lrp_target_layer 34 --model VGG19 --lr 3e-7 --lambda_value 0.4 --loss_type topk --img_name $EXP_NAME $FIXED_OPTION

CUDA_VISIBLE_DEVICES=$GPU python3 $RUN --interpreter grad_cam --lrp_target_layer 34 --model VGG19 --lr 1e-6 --lambda_value 0.25 --loss_type center_mass --img_name $EXP_NAME $FIXED_OPTION


CUDA_VISIBLE_DEVICES=$GPU python3 $RUN --interpreter lrp_T --lrp_target_layer 64 --model Densenet121 --lr 1e-6 --lambda_value 4 --loss_type location --img_name $EXP_NAME $FIXED_OPTION

CUDA_VISIBLE_DEVICES=$GPU python3 $RUN --interpreter lrp_T --lrp_target_layer 64 --model Densenet121 --lr 4e-7 --lambda_value 1.5 --loss_type topk --img_name $EXP_NAME $FIXED_OPTION

CUDA_VISIBLE_DEVICES=$GPU python3 $RUN --interpreter lrp_T --lrp_target_layer 64 --model Densenet121 --lr 6e-7 --lambda_value 0.5 --loss_type center_mass --img_name $EXP_NAME $FIXED_OPTION

CUDA_VISIBLE_DEVICES=$GPU python3 $RUN --interpreter grad_cam --lrp_target_layer 64 --model Densenet121 --lr 2e-6 --lambda_value 2 --loss_type location --img_name $EXP_NAME $FIXED_OPTION

CUDA_VISIBLE_DEVICES=$GPU python3 $RUN --interpreter grad_cam --lrp_target_layer 64 --model Densenet121 --lr 3e-7 --lambda_value 4 --loss_type topk --img_name $EXP_NAME $FIXED_OPTION

CUDA_VISIBLE_DEVICES=$GPU python3 $RUN --interpreter grad_cam --lrp_target_layer 64 --model Densenet121 --lr 1e-7 --lambda_value 1 --loss_type center_mass --img_name $EXP_NAME $FIXED_OPTION


CUDA_VISIBLE_DEVICES=$GPU python3 $RUN --interpreter lrp_T --lrp_target_layer 19 --model Resnet50 --lr 2e-6 --lambda_value 2 --loss_type location --img_name $EXP_NAME $FIXED_OPTION

CUDA_VISIBLE_DEVICES=$GPU python3 $RUN --interpreter lrp_T --lrp_target_layer 19 --model Resnet50 --lr 1e-6 --lambda_value 1 --loss_type topk --img_name $EXP_NAME $FIXED_OPTION

CUDA_VISIBLE_DEVICES=$GPU python3 $RUN --interpreter lrp_T --lrp_target_layer 19 --model Resnet50 --lr 5e-7 --lambda_value 0.25 --loss_type center_mass --img_name $EXP_NAME $FIXED_OPTION

CUDA_VISIBLE_DEVICES=$GPU python3 $RUN --interpreter grad_cam --lrp_target_layer 19 --model Resnet50 --lr 2e-6 --lambda_value 2 --loss_type location --img_name $EXP_NAME $FIXED_OPTION

CUDA_VISIBLE_DEVICES=$GPU python3 $RUN --interpreter grad_cam --lrp_target_layer 19 --model Resnet50 --lr 3e-6 --lambda_value 6 --loss_type topk --img_name $EXP_NAME $FIXED_OPTION

CUDA_VISIBLE_DEVICES=$GPU python3 $RUN --interpreter grad_cam --lrp_target_layer 19 --model Resnet50 --lr 2e-6 --lambda_value 0.25 --loss_type center_mass --img_name $EXP_NAME $FIXED_OPTION
