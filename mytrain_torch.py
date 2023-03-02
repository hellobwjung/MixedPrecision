import os
import time
import argparse
import torch
from myutils_torch import *
from mymodel_torch import build_net
import torch.onnx

from torchsummary import summary

import torchvision
import numpy as np
from tqdm import tqdm
from torch import nn, optim, autocast
# from util.visualizer import Visualizer
# from argument import get_args
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt

# def save_model(model, input_size=(1,1,128,128), name='model_torch.onnx'):
#     # input_size = np.array(input_size)
#     dummy_input = torch.randn(input_size, requires_grad=True)
#     torch.onnx.export(model, dummy_input, name)
#     print('done done saving....' + name)



def save_model(model, ckpt_path, epoch, loss=0.0, state='valid', multigpu=0):
    try:
        fname = os.path.join(ckpt_path, "MC_rmsc%05d__loss_%05.3e.pth"%(epoch, loss))
        if os.path.exists(fname):
            fname = fname.split('.pth')[0] + f'_{state}_1.pth'
        if multigpu > 0:
            model = model.module
        torch.save(
                {
                    "model": model.state_dict(),
                    "epoch"      : epoch,
                },
                fname,
        )
    except:
        print('something wrong......skip saving model at epoch ', epoch)




def train(args):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print('train with', device)
    print(args)
    # args
    model_name      = args.model_name
    dataset_path    = args.dataset_path
    batch_size      = args.batch_size
    device          = args.device
    model_name      = args.model_name
    model_sig       = args.model_sig
    learning_rate   = args.lr
    patch_size      = args.patch_size
    num_gpu         = args.num_gpu

    print('test       = ', args.test, bool(args.test))
    if args.test.lower() == 'true':
        dataset_path = "N:/dataset/MIT/imgs/images_sub10"
        device='cpu'
        num_gpu = 0
    else:
        num_gpu = 1

    print('model_name       = ', model_name)
    print('model_sig        = ', model_sig)
    print('device           = ', device)
    print('batch_size       = ', batch_size)
    print('learning_rate    = ', learning_rate)
    print('num_gpu          = ', num_gpu)
    print('patch_size       = ', patch_size)

    # dataset
    dataset_path = os.path.join(dataset_path)
    print('dataset_path     =', dataset_path)

    # path
    train_path = os.path.join(dataset_path, 'train')
    valid_path  = os.path.join(dataset_path, 'valtest')
    mydata_path = {'train': train_path,
                   'valid': valid_path}

    print('train_path       = ', train_path)
    print('valid_path       = ', valid_path)

    # transform
    transform = {'train': give_me_transform('train', patch_size),
                 'valid': give_me_transform('valid', patch_size)}

    # dataloader
    dataloader = {'train': give_me_dataloader(SingleDataset(mydata_path['train'], transform['train']), batch_size),
                  'valid': give_me_dataloader(SingleDataset(mydata_path['valid'], transform['valid']), batch_size)}
    nsteps={}
    for state in ['train', 'valid']:
        nsteps[state] = len(dataloader[state])
        print('len(%s): '%state, len(dataloader[state]))

    # model
    base_path = os.path.join("model_dir_torch")
    checkpoint_path = os.path.join(base_path, 'checkpoint', model_name + model_sig)
    os.makedirs(base_path, exist_ok=True)
    os.makedirs(checkpoint_path, exist_ok=True)

    model = build_net()
    try:
        save_model(model, name=os.path.join(base_path, 'checkpoint', f'model_torch{model_sig}.onnx'))
        torch.save(model, os.path.join(base_path, 'checkpoint', f'model_torch{model_sig}.pt'))
        torch.save(model.state_dict(), os.path.join(base_path, 'checkpoint', f'model_sd_torch{model_sig}.pt'))
        print(model)
    except:
        print('model already exists')
    exit()
    try:
        summary(model.to(device), input_size=(1, patch_size, patch_size))
    except:
        print('cannot print model, damn pytorch')

    ## ckpt save load if any
    ckpts = os.listdir(checkpoint_path)
    ckpts.sort()
    if (checkpoint_path is not None) and \
        (len( ckpts) > 0) :
        print(checkpoint_path, "<--------------")
        checkpoint = torch.load(os.path.join(checkpoint_path,ckpts[-1]), map_location=device)
        model.load_state_dict(checkpoint["model"])
        epoch = checkpoint["epoch"]
    else:
        epoch=0

    ## make model train and send it device
    model.train()
    model.to(device)

    ##  Loss
    criterion = nn.MSELoss() ## RGB loss

    ## Optimizer, Schedular
    optimizer = optim.Adam(
        list(model.parameters()),
        lr=args.lr,
        betas=(0.5, 0.999),
    )
    lr_lambda = lambda epoch: 1 - ((epoch - 1) // 100) / (args.epoch / 100)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda)

    ## prep for tensorboard
    disp_train = LossDisplayer(["RGB MSE Train"])
    disp_valid = LossDisplayer(["RGB MSE Valid"])
    disp = {'train': disp_train, 'valid': disp_valid}


    log_path = os.path.join(base_path,'board', model_name+model_sig)
    writer = SummaryWriter(log_path)  # for tensorboard
    test_images = give_me_visualization(model, device)
    writer.add_image('Ininitial Images', test_images.permute(2, 0, 1), 0)

    # index tensor
    idxer = Indexer.give_me_idx_tensor(device, patch_size) # <--- 256x256, 128x128


    ## Training gogo
    # logger for tensorboard
    step = {'train': epoch * nsteps['train'], 'valid': epoch * nsteps['train']}
    loss_best = {'train':float('inf'), 'valid':float('inf')}
    print('run model in, ', device.upper())
    while epoch < args.epoch:
        epoch += 1
        print(f"\nEpoch {epoch} :")

        loss_total = {'train': 0, 'valid': 0}
        for state in ['train', 'valid']:
            print('hello ', state)
            pbar = tqdm(dataloader[state])
            for idx, rgbimage in enumerate(pbar):
                step[state] += 1

                pbar.set_description('Processing %s...  epoch %d: ' % (state, epoch))

                # Patternized
                rgbimage = rgbimage.to(device) # [0, 255]
                patterinzed, patterinzed3 = give_me_patternize(rgbimage, idxer) # [-1, 1]

                # Forward
                if '16' in model_sig:
                    with autocast(device_type=device, dtype=torch.float16):
                        pred = model(patterinzed)  # [-1, 1]
                        # assert pred.dtype is torch.float16

                        # Calculate and backward model losses
                        rgbimage_norm = (rgbimage / 127.5) - 1  # [0, 255] -> [0, 2] -> [-1, 1]
                        loss = criterion(pred, rgbimage_norm)
                        # assert loss.dtype is torch.float32

                else:
                    pred = model(patterinzed) # [-1, 1]

                    # Calculate and backward model losses
                    rgbimage_norm = (rgbimage/127.5) - 1 # [0, 255] -> [0, 2] -> [-1, 1]
                    loss = criterion(pred, rgbimage_norm)

                ## write loss to tensorboard
                disp[state].record([loss])
                if (step[state] % 10) == 0 and state == 'train':

                    test_images = give_me_visualization(model, device)
                    writer.add_image(f'{state}', test_images.permute(2, 0, 1), step[state])

                    avg_losses = disp[state].get_avg_losses()
                    writer.add_scalar(f"loss_{state}", avg_losses[0], step[state])
                    # print(f'{state} : epoch{epoch}, step{step[state]}------------------------------------------------------')
                    # print('loss_G: %.3f, ' % avg_losses[0])
                    disp[state].reset()


                if state == 'train':
                    ## train mode
                    #    2. backprop & weight update for current step

                    # Backword
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    loss_step = loss.item()

                else: # valid
                    pass

                # store loss
                loss_total[state] += loss_step

        else:

            # test_images = give_me_visualization(model, device)
            # writer.add_image(f'{state}', test_images.permute(2, 0, 1), step[state])
            print('state -----------> ', state)
            loss_average = loss_total[state] / nsteps[state]
            if loss_best[state] > loss_average:

                print(f'best {state} ckpt updated!!!  old best {loss_best[state]} vs new best {loss_average}')
                loss_best[state] = loss_average
                writer.add_scalar(f"loss_best_{state}", loss_best[state], step[state])

                p = os.path.join(base_path, 'checkpoint', model_name + model_sig)
                save_model(model, p, epoch, loss_best[state], num_gpu)



        # scheculer step
        scheduler.step()

    for i in range(10):
        print('done done training... ', i+1)

def main(args):
    train(args)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()



    argparser.add_argument(
        '--dataset_path',
        default=os.path.join('/home/dataset/MIT/imgs/images_sub10/'),
        type=str,
        help='(default=datasets)')

    argparser.add_argument(
        '--checkpoint_path',
        default=f"model_dir_torch/checkpoint",
        type=str,
        help='(default=%(default)s)')

    argparser.add_argument(
        '--device',
        default='cpu',
        type=str,
        choices=['cpu','cuda'],
        help='(default=%(default)s)')

    argparser.add_argument(
        '--patch_size',
        type=int,
        default=256,
        help='input patch size')

    argparser.add_argument(
        '--model_name',
        default='MC_rmsc',
        type=str,
        choices=['MC_rmsc'],
        help='(default=%(default)s)')

    argparser.add_argument(
        '--model_sig',
        type=str,
        default='_32bit_1',
        help='model postfix')

    argparser.add_argument(
       '--epoch',
       type=int,
       help='epoch number',
       default=600)

    argparser.add_argument(
        '--lr',
        type=float,
        help='learning rate',
        default=1e-3)

    argparser.add_argument(
        '--batch_size',
        type=int,
        help='mini batch size',
        default=256)

    argparser.add_argument(
        '--num_gpu',
        type=int,
        help='mini batch size',
        default=1)

    argparser.add_argument(
        '--test',
        type=str,
        default='True',
        help='test')



    args = argparser.parse_args()
    main(args)
