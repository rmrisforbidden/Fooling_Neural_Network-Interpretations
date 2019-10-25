import module.render as render
import numpy as np
import os
import glob
import torch
from visdom import Visdom
from skimage.io import imsave, imread
from module.arguments import get_args
import pandas as pd
import shutil
import operator
import cv2
from datetime import datetime
import torch
import torch.nn.functional as F
import scipy.stats
import pickle

args = get_args()

import subprocess


def norm_mse(R, R_):
    n=R.shape[0]
    R = R.reshape(n,-1)
    R_ = R_.reshape(n,-1)
    return ((R - R_)**2 ).mean(axis=1)


def target_quanti(interpreter, R_ori, R):
    rcorr_sign = []
    rcorr_rank = []

    if len(R.shape) == 4:
        R = R.sum(axis=1)
        R_ori = R_ori.sum(axis=1)

    rank_corr_sign = quanti_metric(R_ori, R, interpreter, keep_batch=True)
    mse = norm_mse(R, R_ori)
    
    return rank_corr_sign, mse

def quanti_metric(R_ori, R, interpreter, keep_batch = False):
    R_shape = R.shape
    l = R.shape[2]
    R_ori_f = torch.tensor(R_ori.reshape(R.shape[0],-1), dtype=torch.float32).cuda()
    R_f = torch.tensor(R.reshape(R.shape[0],-1), dtype=torch.float32).cuda()
    corr_list = []
    
    for i in range(R.shape[0]):
        corr = 0
        mask_ori = ((abs(R_ori_f[i]) - abs(R_ori_f[i]).mean())/abs(R_ori_f[i]).std()) > 1
        R_ori_f_sign = torch.tensor(mask_ori, dtype=torch.float32).cuda() * torch.sign(R_ori_f[i])

        mask_ = ((abs(R_f[i]) - abs(R_f[i]).mean())/abs(R_f[i]).std()) > 1
        R_f_sign = torch.tensor(mask_, dtype=torch.float32).cuda() * torch.sign(R_f[i])
        corr = np.corrcoef(R_ori_f_sign.detach().cpu().numpy(), R_f_sign.detach().cpu().numpy())[0][1]
        
        corr_list.append(corr)
        
    if keep_batch is True:
        return corr_list
    return np.mean(corr_list)


def get_accuracy(output, labels, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = labels.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(labels.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
#                 res.append(correct_k.mul_(100.0 / batch_size))
            res.append(correct_k)
        return res    

def target_layer():
    # Set target layer
    if args.model == 'VGG19':
        target_layer_ = '34'
    elif args.model == 'Resnet18':
        target_layer_ = '11'
    elif args.model == 'Resnet34':
        target_layer_ = '19'
    elif args.model == 'Resnet50':
        target_layer_ = '19'
    elif args.model == 'Resnet101':
        print(model_name_error)
    elif args.model == 'Densenet121':
        target_layer_ = '64'
    else:
        print(model_name_error)
    return target_layer_


def test_accuracy(net, test_loader, num_data = 50000, label = 0, get_quanti = None, net_ori = None, loss_dict=None, test_mode=False, interpreter_list=None, interpreter_list2=None):
    
    if interpreter_list == None:
        interpreter_list = ['R_lrp34', 'R_grad', 'R_simple34', 'R_lrp']
    
    test_acc1=0.0
    test_acc5=0.0
    
    loss_dict_sub = {}
    for interpreter in interpreter_list:
        loss_dict_sub[interpreter] = []
    
    net.eval()
    torch.manual_seed(20)
    torch.cuda.manual_seed_all(20)
    with torch.no_grad():
        for j, data in enumerate(test_loader):
            inputs, labels = data
            if args.cuda:
                inputs = inputs.cuda()
                labels = labels.cuda() + label
            
            if j >= num_data/inputs.shape[0] and not test_mode:
                break
                
            if j % 100 == 99:
                n = (1+j)*inputs.shape[0]
                print('j: %d\t Top1: %d/%d\t, %.4f%% \t Top5: %d/%d\t, %.4f%%' %(j, int(test_acc1), n, test_acc1/n*100, int(test_acc5), n, test_acc5/n*100))
            activation_output = net.prediction(inputs)
            acc1, acc5 = get_accuracy(activation_output, labels, topk=(1, 5))

            test_acc1 = test_acc1 + acc1.item()
            test_acc5 = test_acc5 + acc5.item()
            
            
            if j > num_data/inputs.shape[0] and test_mode:
                continue    
            # Passive : get R and get loss
            if get_quanti == 'loss':
                for idx, interpreter in enumerate(interpreter_list):
                    
                    # get lrp
                    LRP = net.interpretation(activation_output, interpreter=interpreter, target_layer = args.lrp_target_layer, labels=labels, inputs=inputs)
                    
                        
                    if args.loss_type in ['location']:
                        loss_dict_sub[interpreter] += (net.location(LRP, keep_batch = True).detach()).tolist()
                        
                    else:
                        #get lrp_ori
                        activation_output_ori_ = net_ori.prediction(inputs)
                        activation_output_ori = activation_output_ori_.clone().detach()
                        
                        LRP_ori_ = net_ori.interpretation(activation_output_ori, interpreter=interpreter, target_layer = args.lrp_target_layer, labels=labels)
                            
                        LRP_ori = LRP_ori_.clone().detach()
                        del activation_output_ori_, LRP_ori_, activation_output_ori
                        
                        # get_loss
                        if args.loss_type == 'topk':
                            loss_dict_sub[interpreter] += (net.topk(LRP, LRP_ori, keep_batch = True).detach()).tolist()
                        elif args.loss_type == 'center_mass':
                            loss_dict_sub[interpreter] += (net.center_mass(LRP, LRP_ori, keep_batch = True).detach()).tolist()
    if get_quanti is None:
        return (test_acc1 / num_data), (test_acc5 / num_data)
    
    for interpreter in interpreter_list:
        loss_dict[interpreter].append(loss_dict_sub[interpreter])
        
    if test_mode:
        return (test_acc1 / 50000), (test_acc5 / 50000), loss_dict
    else:
        return (test_acc1 / num_data), (test_acc5 / num_data), loss_dict



def lrp_visualize(R_lrp, gamma = args.gamma):
    heatmaps_lrp = []
    for h, heat in enumerate(R_lrp):
        heat = heat.permute(1,2,0).detach().cpu().numpy()
        maps = render.heatmap(heat,reduce_axis=-1, gamma_ = gamma)
        heatmaps_lrp.append(maps)
    return heatmaps_lrp    

def grad_visualize(R_grad, images):
    R_grad = R_grad.squeeze(1).permute(1,2,0)
    R_grad = R_grad.cpu().detach().numpy()
    R_grad = cv2.resize(R_grad, (224, 224))
    R_grad.reshape(224, 224, images.shape[0])
    heatmaps_grad = []
    
    for i in range(images.shape[0]):
        heatmap = np.float32(cv2.applyColorMap(np.uint8((1-R_grad[:,:,i])*255), cv2.COLORMAP_JET))/255
        cam = heatmap + np.float32(images[i])
        cam = cam / np.max(cam)
        heatmaps_grad.append(cam)
    return heatmaps_grad

def save_visualize(l, R, pre_name, epoch, j, image_path):
    for i in range(l):
        file_name = str(pre_name) + '_epoch' + str(epoch) + '_no-' + str(i+l*j) + '.png'
        path = os.path.join(image_path+args.img_name+str('/'), file_name)
        imsave(path, R[i], plugin='pil')

def visualize_bitargeted(R_lrp_to, R_lrp_from, R_lrp34_to, R_lrp34_from, R_grad_to, R_grad_from, image_tensor, epoch, j, prediction):
    try:
        os.makedirs(args.img_dir+args.img_name+str('/'))##
    except OSError:
        pass
    
    
    # 0. input images
    input_shape = image_tensor.shape
    images = image_tensor.permute(0,2,3,1).cpu().detach().numpy()
    images = images - images.min(axis=(1,2,3), keepdims=True)
    images = images / images.max(axis=(1,2,3), keepdims=True)
    
    # 0. Ready for save images
    image_path = args.img_dir
    img_name = args.img_name
    prediction_dic = {}
    
    l = min(args.batch_size_test, args.num_visualize_plot)
    
    
    # 1. LRP
    heatmaps_lrp_to = lrp_visualize(R_lrp_to)
    heatmaps_lrp_from = lrp_visualize(R_lrp_from)
    
    R_lrp_to = np.array(heatmaps_lrp_to,dtype=np.float32)
    R_lrp_from = np.array(heatmaps_lrp_from,dtype=np.float32)
    
    save_visualize(l, R_lrp_to, 'R_lrp_to', epoch, j, image_path)
    save_visualize(l, R_lrp_from, 'R_lrp_from', epoch, j, image_path)

        
    # 2. LRP34
    
    heatmaps_lrp34_to = grad_visualize(R_lrp34_to, images)
    heatmaps_lrp34_from = grad_visualize(R_lrp34_from, images)
    
    R_lrp34_to = np.array(heatmaps_lrp34_to,dtype=np.float32)
    R_lrp34_from = np.array(heatmaps_lrp34_from,dtype=np.float32)
    
    save_visualize(l, R_lrp34_to, 'R_lrp34_to', epoch, j, image_path)
    save_visualize(l, R_lrp34_from, 'R_lrp34_from', epoch, j, image_path)
    
    # 3. Grad_CAM
    
    heatmaps_grad_to = grad_visualize(R_grad_to, images)
    heatmaps_grad_from = grad_visualize(R_grad_from, images)
    
    R_grad_to = np.array(heatmaps_grad_to,dtype=np.float32)
    R_grad_from = np.array(heatmaps_grad_from,dtype=np.float32)
    
    save_visualize(l, R_grad_to, 'R_grad_to', epoch, j, image_path)
    save_visualize(l, R_grad_from, 'R_grad_from', epoch, j, image_path)
        
    # 4. save original image
    save_visualize(l, images, 'ori', epoch, j, image_path)
    
    # 5. save prediction
    np.save(args.img_dir+args.img_name+str('/')+args.img_name+'_prediction.npy', prediction_dic)

    
    
def visualize5(R_34s, R_inputs, R_34s_key, R_inputs_key, image_tensor, epoch, j, prediction):
    try:
        os.makedirs(args.img_dir+args.img_name+str('/'))##
    except OSError:
        pass

    # 0. input images
    input_shape = image_tensor.shape
    images = image_tensor.permute(0,2,3,1).cpu().detach().numpy()
    images = images - images.min(axis=(1,2,3), keepdims=True)
    images = images / images.max(axis=(1,2,3), keepdims=True)

    # 0. Ready for save images
    image_path = args.img_dir
    img_name = args.img_name
    prediction_dic = {}
    
    l = min(args.batch_size_test, args.num_visualize_plot)
    
    
    # 1. input friends
    for key, R_input in zip(R_inputs_key, R_inputs):
        heatmaps_R= lrp_visualize(R_input, gamma = args.gamma)
        R_input = np.array(heatmaps_R,dtype=np.float32)
        save_visualize(l, R_input, key, epoch, j, image_path)

    # 2. 34 friends
    for key, R_34 in zip(R_34s_key, R_34s):
        if R_34.shape[0] == 1:
            R_34 = R_34.detach().cpu().numpy()
            R_34 = np.concatenate((R_34, R_34), axis=0)
            R_34 = torch.tensor(R_34, dtype=torch.float32)
            
        if images.shape[0]==1:
            images = np.concatenate((images, images), axis=0)
            
        heatmaps_R34= grad_visualize(R_34, images)
        R_34 = np.array(heatmaps_R34,dtype=np.float32)
        save_visualize(l, R_34, key, epoch, j, image_path)
    
    # 3. save original image
    save_visualize(l, images, 'ori', epoch, j, image_path)
    
    # 4. save prediction
    np.save(args.img_dir+args.img_name+str('/')+args.img_name+'_prediction.npy', prediction_dic)   
    return

class logger(object):
    def __init__(self, file_name='mnist_result', resume=False, path=args.log_dir, data_format='csv'):

        self.data_name = os.path.join(path, file_name)
        self.data_path = '{}.csv'.format(self.data_name)
        self.log = None
        if os.path.isfile(self.data_path):
            if resume:
                self.load(self.data_path)
            else:
                os.remove(self.data_path)
                self.log = pd.DataFrame()
        else:
            self.log = pd.DataFrame()

        self.data_format = data_format


    def add(self, **kwargs):
        """Add a new row to the dataframe
        example:
            resultsLog.add(epoch=epoch_num, train_loss=loss,
                           test_loss=test_loss)
        """
        df = pd.DataFrame([kwargs.values()], columns=kwargs.keys())
        self.log = self.log.append(df, ignore_index=True)


    def save(self):
        self.log.to_csv(self.data_path, index=False, index_label=False)

    def load(self, path=None):
        path = path or self.data_path
        if os.path.isfile(path):
            self.log.read_csv(path)
        else:
            raise ValueError('{} isn''t a file'.format(path))
