from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import glob
import copy
import h5py
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pdb
import scipy.io as sio
import cv2
import torchvision
import torchvision.transforms as T
from torchvision import utils
from torchvision import transforms

sys.path.append("..")

import module.render as render
from module.arguments import get_args
from module.load_model import load_model
from module.dataloader import dataloader, active_loader

from module.utils import logger, test_accuracy, visualize_bitargeted, target_quanti

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
    
    D_fool_loader, val_loader, c1_loader, c2_loader = active_loader(args)
    train_loader = dataloader(args, train=True)
    test_loader = dataloader(args, test=True)
    
    # 2. load model
    print('2. load model')
    
    net = load_model(args.model, pretrained=True, pretrained_path=None)
    net_ori = load_model(args.model, pretrained=True, pretrained_path=None)
        
    
    # Use GPU
    if args.cuda:
        net.cuda()
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
    c1_acc1_list=[]
    c1_acc5_list=[]
    c2_acc1_list =[]
    c2_acc5_list =[]
    
    result = {}
    for interpreter in ['grad_cam', 'lrp34', 'lrp']:
        for metric in ['_rank', '_mse']:
            for ori in ['_c1', '_c2']:
                for curr in ['_c1', '_c2']:
                    result[interpreter+metric+ori+curr] = []
    
    first_itr = True
    stop_all_process = False
    
    itr=-1
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
        
        for i, train_data in enumerate(train_loader):
            itr += 1
            
            # 1. Stoping criteria.
            if itr == args.num_eval*args.eval_period+1:
                stop_all_process = True
                break
            
            # 2. We recommend you to use eval for model with BN.
            if args.eval:
                net.eval()
                net_ori.eval()
            else:
                net.train()
            
            ########################################################################################
            ####################################### Training #######################################
            ########################################################################################
            
            #### prepairing training datas
            if first_itr == False:
                
                # 1. Preparing Inputs for network forward.
                for j , D_fool_data in enumerate(D_fool_loader):
                    inputs_D_fool = D_fool_data[0]
                    
                    labels_D_fool = torch.zeros(len(inputs_D_fool), dtype=torch.long)
                    inputs_D_fool = torch.tensor(inputs_D_fool.numpy(), dtype=torch.float32)
                   
                    if args.cuda:
                        inputs_D_fool = inputs_D_fool.cuda()
                        labels_D_fool = labels_D_fool.cuda()
                    break
                
                inputs_D, labels_D = train_data
                inputs_D = torch.tensor(inputs_D.numpy(), dtype=torch.float32)

                if args.cuda:
                    inputs_D = inputs_D.cuda()
                    labels_D = labels_D.cuda()

                # 2. evaluate losses
                # 2-1. imagenet
                optimizer.zero_grad()
                activation_output = net.prediction(inputs_D)
                class_loss = criterion(activation_output, labels_D)
                
                # 2-2. D_fool
                lrp_loss= net.forward_targeted(inputs_D_fool, labels_D_fool, net_ori, args.class_c1, args.class_c2, args.lrp_target_layer)
                
                # 2-3. total loss
                total_loss = class_loss + args.lambda_value * lrp_loss

                # backward
                total_loss.backward()
                optimizer.step()

                # print statistics
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
            if itr % args.eval_period == 0:
                print('--Start Evaluation')
                torch.manual_seed(20)
                torch.cuda.manual_seed_all(20)
                first_itr = False

                '''
                Eval 1. Visualization
                '''
                with torch.no_grad(): 
                    net.eval()
                    net_ori.eval()
                    num_tested = 0
                    
                    for j, data in enumerate(val_loader):
                        print(itr,'th training iteration, ',j,'th validation minibatch')
                        inputs = data[0]
                        labels_c1 = torch.zeros(len(inputs), dtype=torch.long) + args.class_c1
                        labels_c2 = torch.zeros(len(inputs), dtype=torch.long) + args.class_c2
                        inputs = torch.tensor(inputs.numpy(), dtype=torch.float32)
                        if args.cuda:
                            inputs = inputs.cuda()
                            labels_c1 = labels_c1.cuda()
                            labels_c2 = labels_c2.cuda()
                        
                        # 2. class loss
                        activation_output = net.prediction(inputs)
                        prediction = torch.argmax(activation_output, 1)
                        
                        target_layer = args.lrp_target_layer
                        
                        # 1-1. visualize importance map
                        R_lrp_c1 = net.lrp(activation_output, labels=labels_c1, target_layer=None) 
                        R_grad_c1 = net.grad_cam(activation_output, labels=labels_c1, target_layer=target_layer)
                        R_lrp34_c1 = net.lrp(activation_output, labels=labels_c1, target_layer=target_layer)
                        
                        # 1-2. bi targeted visualize importance map
                        R_lrp_c2 = net.lrp(activation_output, labels=labels_c2, target_layer=None) 
                        R_grad_c2 = net.grad_cam(activation_output, labels=labels_c2, target_layer=target_layer)
                        R_lrp34_c2 = net.lrp(activation_output, labels=labels_c2, target_layer=target_layer)
                        
                        if num_tested < args.num_visualize_plot:
                            visualize_bitargeted(R_lrp_c2, R_lrp_c1, R_lrp34_c2, R_lrp34_c1, R_grad_c2, R_grad_c1, inputs, itr, j, prediction)
                        
                        # 4. R_array
                        if num_tested == 0:
                            R_grad_c2_arr = R_grad_c2.cpu().detach().numpy()
                            R_lrp34_c2_arr = R_lrp34_c2.cpu().detach().numpy()
                            R_lrp_c2_arr = R_lrp_c2.cpu().detach().numpy()
                            
                            R_grad_c1_arr = R_grad_c1.cpu().detach().numpy()
                            R_lrp34_c1_arr = R_lrp34_c1.cpu().detach().numpy()
                            R_lrp_c1_arr = R_lrp_c1.cpu().detach().numpy()
                            
                        else:
                            R_grad_c2_arr = np.concatenate((R_grad_c2_arr,R_grad_c2.cpu().detach().numpy()))
                            R_lrp34_c2_arr = np.concatenate((R_lrp34_c2_arr,R_lrp34_c2.cpu().detach().numpy()))
                            R_lrp_c2_arr = np.concatenate((R_lrp_c2_arr,R_lrp_c2.cpu().detach().numpy()))
                            
                            R_grad_c1_arr = np.concatenate((R_grad_c1_arr,R_grad_c1.cpu().detach().numpy()))
                            R_lrp34_c1_arr = np.concatenate((R_lrp34_c1_arr,R_lrp34_c1.cpu().detach().numpy()))
                            R_lrp_c1_arr = np.concatenate((R_lrp_c1_arr,R_lrp_c1.cpu().detach().numpy()))

                        num_tested += inputs.shape[0]
                        
                        if args.get_correlation == False and num_tested >= args.num_visualize_plot:
                            break
                        
                    # Save original R
        
                    R_c2 = {}
                    R_c2['grad_cam'] = R_grad_c2_arr
                    R_c2['lrp'] = R_lrp_c2_arr
                    R_c2['lrp34'] = R_lrp34_c2_arr
                    R_c1 = {}
                    R_c1['grad_cam'] = R_grad_c1_arr
                    R_c1['lrp'] = R_lrp_c1_arr
                    R_c1['lrp34'] = R_lrp34_c1_arr
                    
                    if itr==0:
                        R_c2_ori = R_c2
                        R_c1_ori = R_c1
                    
                    for interpreter in ['grad_cam', 'lrp34', 'lrp']:
                        for Rori_, ori in zip([R_c2_ori, R_c1_ori], ['_c2', '_c1']):
                            for R_, curr in zip([R_c2, R_c1], ['_c2', '_c1']):
                                rank, mse = target_quanti(interpreter, Rori_[interpreter], R_[interpreter])
                                result[interpreter+'_rank'+ori+curr].append(rank)
                                result[interpreter+'_mse'+ori+curr].append(mse)
                                
                    for key in result.keys():
                        np.save(args.img_dir+args.img_name+str('/')+args.img_name+key+'.npy', result[key])  
                    
                    # 5. evaluate test accuracy
                    if args.imagenet_accuracy:
                        test_acc1, test_acc5 = test_accuracy(net, test_loader, 250)
                        test_acc1_list.append(test_acc1)
                        test_acc5_list.append(test_acc5)
                        
                        np.save(args.img_dir+args.img_name+str('/')+args.img_name+'_test_acc1.npy', test_acc1_list)
                        np.save(args.img_dir+args.img_name+str('/')+args.img_name+'_test_acc5.npy', test_acc5_list)
                        
                    if args.c1_c2_accuracy:
                        c1_acc1, c1_acc5 = test_accuracy(net, c1_loader, 50, label=args.class_c1)
                        c1_acc1_list.append(c1_acc1)
                        c1_acc5_list.append(c1_acc5)
                        
                        np.save(args.img_dir+args.img_name+str('/')+args.img_name+'_c1_acc1.npy', c1_acc1_list)
                        np.save(args.img_dir+args.img_name+str('/')+args.img_name+'_c1_acc5.npy', c1_acc5_list)
                        
                        c2_acc1, c2_acc5 = test_accuracy(net, c2_loader, 50, label=args.class_c2)
                        c2_acc1_list.append(c2_acc1)
                        c2_acc5_list.append(c2_acc5)
                        
                        np.save(args.img_dir+args.img_name+str('/')+args.img_name+'_c2_acc1.npy', c2_acc1_list)
                        np.save(args.img_dir+args.img_name+str('/')+args.img_name+'_c2_acc5.npy', c2_acc5_list)

                    # Saving model       
                    if args.save_model == True:
                        torch.save(net.state_dict(),os.path.join(args.save_dir,args.img_name + str(itr)+'.pt'))
        e_time = time.time()
        print("---> (Test) Elapsed Time: ", e_time-s_time)



if __name__ == '__main__':
    main()                   