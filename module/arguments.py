import argparse

import torch

def str2bool(v):
    if v.lower() in ('yes', 'true','True', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'False','f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    
    parser = argparse.ArgumentParser(description='LRP')
    
    parser.add_argument('--model', type=str, default='VGG19', 
                        choices=['VGG19', 'Resnet18', 'Resnet34', 'Resnet50', 'Resnet101', 'Densenet121'], 
                        help='The name of model to be used in experiment')
    
    parser.add_argument('--interpreter', type=str, default='lrp', 
                        choices=['lrp', 'lrp_T', 'grad_cam', 'simple_grad', 'simple_grad_T', 'smooth_grad', 'smooth_grad_T', 'integrated_grad', 'integrated_grad_T'], 
                        help='The interpreter for fooling loss')

    parser.add_argument('--loss_type', type=str, default='uniformR', choices=['location', 'topk', 'center_mass', 'active'], 
                        help='The type of fooling loss')
    
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Training batch size')
    
    parser.add_argument('--batch-size_test', type=int, default=16,
                        help='Validation batch size')
    
    parser.add_argument('--eval-period', type=int, default=100,
                        help='Evaluation period.')
    
    parser.add_argument('--num_eval', type=int, default=5,
                        help='Number of evaluation. The total iteration will be num_eval * eval-period')
    
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='learning rate for training')
    
    parser.add_argument('--imagenet_accuracy', action='store_true', default=False,
                        help='Get accuracy and visualization for ImageNet validation set')
    
    parser.add_argument('--r_ts_keys', nargs='+', type=str, default= ['lrp_T', 'grad_cam', 'simple_grad_T'],
                        help='The list of interpretations_T to be evaluated.')
    parser.add_argument('--r_inputs_keys', nargs='+', type=str, default= ['lrp'],
                        help='The list of interpretations to be evaluated. The dimension of this interpretation is same as input')
    
    parser.add_argument('--eval', action='store_true', default=True,
                        help='if true, training will be performed by eval model of model. (recommend with BN model')
    
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'SGD'], 
                        help='optimizer (default: Adam)')
    parser.add_argument('--vgg_bn', action='store_true', default=False, help='BN for VGG')
    parser.add_argument('--epsilon', type=float, default=0.01,
                        help='beta for alpha beta lrp')
    
    parser.add_argument('--num_visualize_plot', type=int, default=100,
                        help='number of img to see') 
    
    parser.add_argument('--lrp_target_layer', type=str, default=None,
                        help='label of mixed data 2')
    
    
    
    parser.add_argument('--save_model', action='store_true', default=False,
                        help='whether save model or not')
    parser.add_argument('--saved_model', type=str, default=None, 
                        help='The model you want to load from run_lrp/trained_model/.')
    parser.add_argument('--log_dir', default='./results_ImageNet/',
                        help='directory to save logs (default: ./results_ImageNet)')
    parser.add_argument('--save-dir', default='./trained_models/',
                        help='directory to save trained models (default: ./trained_models/)')
    parser.add_argument('--img_dir', default='../img_results/', 
                        help='img (default: )')
    parser.add_argument('--img_name', type=str, default='results3',
                        help='img file naming (default: results )') 
    
    
    parser.add_argument('--whichScore',  type=int, default=None, #8,
                        help='score that you want to see/ can be 0~9 in MNIST')
    parser.add_argument('--lambda_value', type=float, default=0.001,
                        help='lambda_ (0.01, 0.05, 0.1)')
    parser.add_argument('--r-method', type=str, default='composite',
                        help='relevance methods: simple/epsilon/alphabeta/composite/')
    
    
    parser.add_argument('--beta', type=float, default=0,
                        help='beta for alpha beta lrp')
    parser.add_argument('--gamma', type=float, default=0.9,
                        help='gamma for fucntion heatmap() from render.py ')
    parser.add_argument('--smooth_std', type=float, default=0.1,
                        help='hyperparameter of smooth gradient')
    parser.add_argument('--smooth_num', type=int, default=16,
                        help='number of random noise generation for smooth gradient')
    
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of processes for dataloader (default: 4)')
    
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epoch. Warning: The number of iteration will be args.num_eval * args.eval_period.')
    
    parser.add_argument('--random_seed', type=int, default=20,
                        help='random seed for visualization')
    
    parser.add_argument('--heatmap_label', type=int, default=0,
                        help='the class label that you want to get heatmap')
    parser.add_argument('--class_c1', type=int, default=555,
                        help='label of mixed data 1')
    parser.add_argument('--class_c2', type=int, default=386,
                        help='label of mixed data 2')
    
    parser.add_argument('--c1_c2_accuracy', action='store_true', default=False,
                        help='if you want to get c1_acc when targeted ')
    parser.add_argument('--get_correlation', action='store_true', default=False,
                        help='if you want to get from_acc when targeted ')
    
    
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    
    for key in args.r_ts_keys:
        if key not in ['lrp_T', 'grad_cam', 'simple_grad_T', 'smooth_grad_T', 'integrated_grad_T']:
            print(key)
            print(KEY_ERROR)
    
    for key in args.r_inputs_keys:
        if key not in ['lrp', 'simple_grad', 'smooth_grad','integrated_grad']:
            print(KEY_ERROR)
            
    args.img_name = args.img_name + args.model+'_'+args.interpreter+args.lrp_target_layer+'_'+args.loss_type+'_lr_'+str(args.lr)+'_lambda_'+str(args.lambda_value)

    return args
