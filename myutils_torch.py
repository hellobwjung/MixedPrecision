
import os, glob

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torchvision.utils as vutils

import numpy as np
from PIL import Image


class Indexer:
    def __init__(self, crop_size = 128, cfa_pattern = 2):
        self.crop_size = crop_size
        self.cfa_pattern = cfa_pattern

    @staticmethod
    def give_me_idx(crop_size=128, cfa_pattern=2):
        idx_R = np.tile(
            np.concatenate((np.concatenate((np.zeros((cfa_pattern, cfa_pattern)), np.ones((cfa_pattern, cfa_pattern))), axis=1),
                            np.concatenate((np.zeros((cfa_pattern, cfa_pattern)), np.zeros((cfa_pattern, cfa_pattern))),
                                           axis=1)), axis=0),
            (crop_size // 2 // cfa_pattern, crop_size // 2 // cfa_pattern))

        idx_B = np.tile(
            np.concatenate(
                (np.concatenate((np.zeros((cfa_pattern, cfa_pattern)), np.zeros((cfa_pattern, cfa_pattern))), axis=1),
                 np.concatenate((np.ones((cfa_pattern, cfa_pattern)), np.zeros((cfa_pattern, cfa_pattern))), axis=1)), axis=0),
            (crop_size // 2 // cfa_pattern, crop_size // 2 // cfa_pattern))

        idx_G = np.tile(
            np.concatenate((np.concatenate((np.ones((cfa_pattern, cfa_pattern)), np.zeros((cfa_pattern, cfa_pattern))), axis=1),
                            np.concatenate((np.zeros((cfa_pattern, cfa_pattern)), np.ones((cfa_pattern, cfa_pattern))),
                                           axis=1)), axis=0),
            (crop_size // 2 // cfa_pattern, crop_size // 2 // cfa_pattern))

        return idx_R, idx_G, idx_B

    @staticmethod
    def give_me_idx_tensor(device='cpu', patch_size=128):
        idx_R, idx_G, idx_B = Indexer.give_me_idx(patch_size)
        idx_Rt = torch.from_numpy(idx_R).to(device)
        idx_Gt = torch.from_numpy(idx_G).to(device)
        idx_Bt = torch.from_numpy(idx_B).to(device)
        return idx_Rt, idx_Gt, idx_Bt

def give_me_patternize(image, idxer):
    ## image ~ [-1, 1] torch.tensor
    ## RAW2RGB
    idx_Rt, idx_Gt, idx_Bt = idxer
    img_R = image * idx_Rt
    img_G = image * idx_Gt
    img_B = image * idx_Bt
    patternized3 = img_R + img_G + img_B # [0, 255]
    patternized3 = (patternized3 / 127.5) - 1 # [0, 255] -> [0, 2] -> [-1, 1]
    patternized = torch.sum(patternized3, dim=1, keepdim=True)

    # print()
    # print(image.shape, idx_Rt.shape, img_g.shape, patternized.shape, patternized3.shape)


    return patternized, patternized3




def give_me_transform(type, patch_size, mean=0.5, std=0.5):

    transform = None
    if type == 'train':
        transform = transforms.Compose(
            [
                transforms.Resize((patch_size, patch_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(mean, mean, mean), std=(std, std, std)),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize((patch_size, patch_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(mean, mean, mean), std=(std, std, std)),
            ]
        )
    return transform


# dataloader
def give_me_dataloader(dataset, batch_size:int, shuffle=True, num_workers=2, drop_last=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)


def give_me_test_images(input_size=128):
    fname = glob.glob(os.path.join('test_images', '*.png'))
    image_tensor = torch.zeros(len(fname), 3, input_size, input_size)
    # image_list = []
    for idx, f in enumerate(fname):
        fpath = os.path.join(f)
        arr = np.array(Image.open(fpath)).astype(np.float32)
        image = torch.Tensor(arr).float().detach()
        image_tensor[idx] = image.permute(2,0,1)
        # print(type(arr), arr.dtype, type(image), image.dtype)
    return image_tensor


def give_me_comparison(model, inputs, device):
    # print('inputs.size ', inputs.size(), type(inputs))
    with torch.no_grad():
        model.eval()
        if device == torch.device('cuda'):
            inputs=inputs.cuda()
            # print('input is in cuda')
        else:
            # print('input is in cpu')
            pass
            ...

        # print(type(inputs))
        # model.cpu()
        if device=='cuda' or next(model.parameters()).is_cuda:
            inputs=inputs.cuda()
        outputs = model(inputs)
    return outputs

def give_me_visualization(model, device='cpu', test_batch=None, nomalize=True):
    # visualize test images
    # print('test_batch', type(test_batch))
    if test_batch != None:
        rgbimages = test_batch.cpu() # RAW
    else:
        rgbimages = give_me_test_images().to(device) # [0, 255]

    patt, patt3 = give_me_patternize(rgbimages, Indexer.give_me_idx_tensor(device)) # [-1, 1]
    rgbimages = (rgbimages / 127.5) - 1. # [-1, 1]

    pred = give_me_comparison(model, patt, device=device) # [-1, 1]

    diff = torch.abs(pred.to(device) - rgbimages.to(device)) / 2 # [0, 1]

    # print('rgb (%.3f, %.3f), ' %(torch.amin(rgbimages), torch.amax(rgbimages)), end='')
    # print('patt (%.3f, %.3f), ' %(torch.amin(patt3), torch.amax(patt3)), end='')
    # print('pred (%.3f, %.3f), ' %(torch.amin(pred), torch.amax(pred)), end='')
    # print('diff (%.3f, %.3f), ' %(torch.amin(diff), torch.amax(diff)))

    patt3_images = vutils.make_grid(patt3*2-1,     padding=2, normalize=nomalize)
    rgb_images   = vutils.make_grid(rgbimages*2-1, padding=2, normalize=nomalize)
    diff_images  = vutils.make_grid(diff,          padding=2, normalize=nomalize)
    pred_images  = vutils.make_grid(pred*2-1,      padding=2, normalize=nomalize)

    input_images = torch.cat((patt3_images.cpu(), rgb_images.cpu() ), dim=2)
    output_images = torch.cat((diff_images.cpu(), pred_images.cpu() ), dim=2)
    # test_images = torch.cat((input_images.cpu(),   output_images.cpu()),    dim=1)
    # test_images = output_images
    test_images = torch.cat((rgb_images.cpu(), pred_images.cpu()), dim=1)

    # if test_batch != None:
    test_images = test_images.permute(1,2,0)
    return test_images






# dataset
class SingleDataset(DataLoader):
    def __init__(self, dataset_dir, transforms, mylen=-1, bits=8, ext='png'):
        self.dataset_dir = dataset_dir
        self.transform = transforms
        self.mylen = mylen
        self.bits = bits

        self.image_path = glob.glob(os.path.join(dataset_dir, f"**/*.{ext}") , recursive=True)
        if mylen > 0:
            self.image_path = self.image_path[:mylen]

        print('--------> # of images: ', len(self.image_path))

    def __getitem__(self, index):
        if self.image_path[index].split('.')[-1] == 'npy':
            item = np.load(self.image_path[index])
            item = self.transform(item).transpose(2, 0, 1)
        else:
            item = self.transform(Image.open(self.image_path[index]))
        return item

    def __len__(self):
        return len(self.image_path)





class LossDisplayer:
    def __init__(self, name_list):
        self.count = 0
        self.name_list = name_list
        self.loss_list = [0] * len(self.name_list)

    def record(self, losses):
        self.count += 1
        for i, loss in enumerate(losses):
            self.loss_list[i] += loss.item()

    def get_avg_losses(self):
        return [loss / self.count for loss in self.loss_list]

    def display(self):
        for i, total_loss in enumerate(self.loss_list):
            avg_loss = total_loss / self.count
            print(f"{self.name_list[i]}: {avg_loss:.4f}   ", end="")

    def reset(self):
        self.count = 0
        self.loss_list = [0] * len(self.name_list)
