# TODO Sep 08 version
"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import math


def tensor2im(input_image, imtype=np.uint16):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy_og = image_tensor.cpu().float().numpy()  # convert it into a numpy array
        image_numpy = image_numpy_og.copy()

        if imtype == np.uint8:
            image_numpy = np.clip(image_numpy, 0, 1)
            image_numpy *= (2 ** 8 * 1.0 - 1)
            image_numpy = np.clip(image_numpy, 0, 255)

        if imtype == np.uint16:
            image_numpy = np.clip(image_numpy, 0, 1)
            image_numpy *= (2 ** 16 * 1.0 - 1)
            image_numpy = np.clip(image_numpy, 0, 2**16-1)
        if imtype == np.float:
            pass
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)

#
# def normalize(img_np, is_tensor=False):
#     if is_tensor:
#         img_min = torch.min(img_np)
#         img_max = torch.max(img_np)
#     else:
#         img_min = np.min(img_np)
#         img_max = np.max(img_np)
#
#     new_min = 0
#     new_max = 1
#     img_normd = (img_np - img_min) * ((new_max - new_min) / (img_max - img_min)) + new_min
#
#     return img_normd


def normalize(img_np, data_type = float):
    img_min = np.min(img_np)
    img_max = np.max(img_np)

    new_min = 0
    if data_type == np.uint8:
        new_max = 2**8-1
    elif data_type == np.uint16:
        new_max = 2**16-1
    elif data_type == np.float:
        new_max = 1

    img_normd = (img_np - img_min) * ((new_max - new_min) / (img_max - img_min)) + new_min
    img_normd = img_normd.astype(data_type)

    return img_normd

def noisy(noise_typ, image, sigma=0.1, peak=0.1, is_tensor=False, is_normalize=True):
    if is_tensor:
        image = image.cpu().float().detach().numpy()

    if noise_typ == "gauss":
        b, c, row, col, ch = image.shape
        mean = 0
        # sigma = gau_var**0.5
        gauss = np.random.normal(mean, sigma, (image.shape))
        # gauss = gauss.reshape(image.shape)
        noisy = image + gauss

    elif noise_typ == "poisson":  # simulate a low-light noisy image.
        # vals = len(np.unique(image))
        # vals = 2 ** np.ceil(np.log2(vals))
        # print (vals)
        noisy = np.random.poisson(image * peak) / float(peak)
        #  s = np.random.poisson(5, 10000) would mean the probability of picking '5' when you draw 10000 times in a poisson process.

    if is_normalize:
        noisy = normalize(noisy)

    if is_tensor:
        noisy = torch.from_numpy(noisy).float().to(torch.device('cuda')).detach()

    return noisy

def get_mse(source, target):
    mse = np.mean((target - source)**2)
    return mse

def get_snr(img_original, img_noised):
    mse = np.mean((img_original - img_noised) ** 2)  # Pw
    Ps = np.mean(img_original ** 2)
    snr_linearscale = Ps / mse
    return 10 * math.log(snr_linearscale, 10)

def standardize(img_np):
    return (img_np-np.mean(img_np))/np.std(img_np)

def get_psnr(source, target, data_range):
    target = target.astype(float)
    source = target.astype(float)

    mse = np.mean((target - source)**2)
    return 20*math.log(data_range,10)-10*math.log(mse,10)

def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0, save_all=False):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """
    image_pil = Image.fromarray(image_numpy)
    h, w = image_numpy.shape
    # if aspect_ratio > 1.0:
    #     image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    # if aspect_ratio < 1.0:
    #     image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    # save_all is an option for saving a 3D image.
    image_pil.save(image_path, save_all=save_all)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)


def pad_for_dicing(image, roi_size, overlap=0):
    image_z = image.shape[0]
    image_y = image.shape[1]
    image_x = image.shape[2]

    step = roi_size - overlap

    step_counts_x = (image_x + overlap) // step
    step_counts_y = (image_y + overlap) // step
    step_counts_z = (image_z + overlap) // step

    x_pad = step * step_counts_x + roi_size - image_x
    y_pad = step * step_counts_y + roi_size - image_y
    z_pad = step * step_counts_z + roi_size - image_z

    npad = ((0, z_pad), (0, y_pad), (0, x_pad))
    image_padded = np.pad(image, pad_width=npad)
    print("image volume is padded for equal dicing. crop sizes are: {}".format(npad))

    return image_padded


def crop_for_dicing(image, roi_size, overlap=0):
    image_z = image.shape[0]
    image_y = image.shape[1]
    image_x = image.shape[2]

    step = roi_size - overlap

    step_counts_x = (image_x - overlap) // step
    step_counts_y = (image_y - overlap) // step
    step_counts_z = (image_z - overlap) // step

    x_crop = image_x - step * step_counts_x - overlap
    y_crop = image_y - step * step_counts_y - overlap
    z_crop = image_z - step * step_counts_z - overlap

    # image_cropped = image[:-z_crop, :-y_crop, :-x_crop]
    image_cropped = image[z_crop:, y_crop:, x_crop:]

    print("image volume is cropped for equal dicing. crop sizes are: {}".format((z_crop, y_crop, x_crop)))
    return image_cropped



