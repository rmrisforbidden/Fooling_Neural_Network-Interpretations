from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import glob
import copy
import h5py
import numpy as np
import pdb
import scipy.io as sio
import cv2
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim

sys.path.append("..")

import module.render as render
from module.load_model import load_model
from module.dataloader import dataloader
from module.arguments import get_args

from module.utils import logger, test_accuracy, visualize5

import time

args = get_args()

try:
    os.makedirs(args.log_dir)
except OSError:
    files = glob.glob(os.path.join(args.log_dir, '*.csv'))
    for f in files:
        os.remove(f)

def init_weights(m):
        m.reset_parameters()
        
def main():
    print('--------------------Preparing Training------------------------')
    # 1. load dataloader
    print('1. load data-loader')
    
    train_loader = dataloader(args, train=True)
    val_loader = dataloader(args, val=True)
    test_loader = dataloader(args, test=True)
        
    # 2. load model
    print('2. load model')
    
    net = load_model(args.model, pretrained=True, pretrained_path=None)
    net_ori = None
    if args.loss_type in ['topk', 'center_mass']:
        net_ori = load_model(args.model, pretrained=True, pretrained_path=None)
    
    # Some settings for training
    if args.cuda:
        net.cuda()
        if args.loss_type in ['topk', 'center_mass']:
            net_ori.cuda()

    log = logger()

    criterion = nn.CrossEntropyLoss()

    if args.optimizer == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=args.lr)

    
    # 3. lists for saving results
    
    loss_c_tr_list = []
    loss_r_tr_list = []
    loss_t_tr_list = []
    
    test_acc1_list=[]
    test_acc5_list=[]
    
    R_Ts_keys = args.r_ts_keys
    R_inputs_keys = args.r_inputs_keys
    interpreter_list = R_Ts_keys + R_inputs_keys
    
    loss_dict = {}
    loss_dict_val = {}
    for interpreter in interpreter_list:
        loss_dict[interpreter] = []
        loss_dict_val[interpreter] = []
    
    first_itr = True
    stop_all_process = False
            
    itr = -1
    running_c_loss = 0.0
    running_r_loss = 0.0
    running_t_loss = 0.0
    
    print('--------------------Start Training------------------------')
    print('========== args ========== \n', args)
    for epoch in range(args.epochs):
        s_time = time.time()
        if stop_all_process:
            break

        seed = int(time.time() * 100000)%1000000
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        for i, data in enumerate(train_loader):
            itr += 1
            
            # 1. Stoping criteria.
            if itr == args.num_eval*args.eval_period +1:
                stop_all_process = True
                break
                
            # 2. We recommend you to use eval for model with BN.
            if args.eval:
                net.eval()
                if args.loss_type in ['topk', 'center_mass']:
                    net_ori.eval()
            else:
                net.train()
            
            ########################################################################################
            ####################################### Training #######################################
            ########################################################################################
            if first_itr == False:
                
                # Preparing Inputs for network forward.
                inputs, labels = data
            
                if args.cuda:
                    inputs, labels = inputs.cuda(), labels.cuda()
  
                if args.loss_type in ['topk', 'center_mass']:
                    activation_output_ = net_ori.prediction(inputs)
                    activation_output = activation_output_.clone().detach()
                    
                    LRP_ori_ = net_ori.interpretation(activation_output, interpreter=args.interpreter, labels=labels, target_layer = args.lrp_target_layer, inputs=inputs)
                    
                    LRP_ori = LRP_ori_.clone().detach()
                    del activation_output_, LRP_ori_, activation_output
                else:
                    LRP_ori = None
                
                # forward + backward + optimize
                optimizer.zero_grad()
                total_loss, class_loss, lrp_loss, activation_output, check_total_loss, R = net.forward(inputs, labels, args.lambda_value, LRP_ori = LRP_ori)
                
                total_loss.backward()
                optimizer.step()
                
                # update and save training statistics
                running_c_loss += class_loss.item()
                running_r_loss += lrp_loss.item()
                running_t_loss += total_loss.item()

                loss_period = 20
                if itr % loss_period == 0:  # print every 20 mini-batches
                    loss_c_tr_list.append(running_c_loss/loss_period)
                    loss_r_tr_list.append(running_r_loss/loss_period)
                    loss_t_tr_list.append(running_t_loss/loss_period)
                    print('iter: ', itr,'classification loss: ', loss_c_tr_list[-1], ' R loss: ', loss_r_tr_list[-1])

                    running_c_loss = 0.0
                    running_r_loss = 0.0
                    running_t_loss = 0.0
                    
                    np.save(args.img_dir+args.img_name+str('/')+args.img_name+'_loss_c.npy', loss_c_tr_list)
                    np.save(args.img_dir+args.img_name+str('/')+args.img_name+'_loss_r.npy', loss_r_tr_list)
                    np.save(args.img_dir+args.img_name+str('/')+args.img_name+'_loss_t.npy', loss_t_tr_list)
            
            ########################################################################################
            ###################################### Evaluating ######################################
            ########################################################################################
            if itr % args.eval_period  == 0:
                print('--Start Evaluation')
                torch.manual_seed(20)
                torch.cuda.manual_seed_all(20)
                first_itr = False

                '''
                Eval 1. Visualization
                '''
                with torch.no_grad():
                    net.eval()
                    num_tested = 0

                    for j, data in enumerate(val_loader):
                        print(itr,'th training iteration, ',j,'th validation minibatch')
                        inputs, labels = data
                        org_inputs = inputs.clone()
                        if args.cuda:
                            inputs, labels = inputs.cuda(), labels.cuda()

                        activation_output = net.prediction(inputs)
                        _, prediction = torch.max(activation_output, 1)

                        R_Ts = []
                        R_inputs = []

                        target_layer = args.lrp_target_layer

                        for interpreter in R_Ts_keys:
                            R_Ts.append(net.interpretation(activation_output, interpreter=interpreter, labels=labels, target_layer = args.lrp_target_layer, inputs=inputs))
                        for interpreter in R_inputs_keys:
                            R_inputs.append(net.interpretation(activation_output, interpreter=interpreter, labels=labels, target_layer = None, inputs=inputs))

                        visualize5(R_Ts, R_inputs, R_Ts_keys, R_inputs_keys, inputs, itr, j, prediction)

                        num_tested += inputs.shape[0]
                        if num_tested >= args.num_visualize_plot:
                                break
                del R_inputs, R_Ts
                
                '''
                Eval 2. test_accuracy and loss
                '''
                if args.imagenet_accuracy:
                    test_acc1, test_acc5, loss_dict = test_accuracy(net, test_loader, num_data=50000, get_quanti ='loss', loss_dict = loss_dict, net_ori = net_ori, test_mode=True, interpreter_list=interpreter_list)
                    test_acc1_list.append(test_acc1)
                    test_acc5_list.append(test_acc5)

                    np.save(args.img_dir+args.img_name+str('/')+args.img_name+'_test_acc1.npy', test_acc1_list)
                    np.save(args.img_dir+args.img_name+str('/')+args.img_name+'_test_acc5.npy', test_acc5_list)
                    np.save(args.img_dir+args.img_name+str('/')+args.img_name+'_loss_dict.npy', loss_dict)

                # Saving model
                if args.save_model == True:
                    torch.save(net.state_dict(),os.path.join(args.save_dir,args.img_name + str(itr) +'.pt'))
                    
        e_time = time.time()
        print("---> (Test) Elapsed Time: ", e_time-s_time)



if __name__ == '__main__':
    main()
