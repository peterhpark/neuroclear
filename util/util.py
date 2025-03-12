"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
import math
import ntpath
from tifffile import imwrite

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

        # NOTE Notice that we assume the data range to be (0,1). Be carefull if you set it to (-1,1)
        if imtype == np.uint8:
            image_numpy = np.clip(image_numpy, 0, 1)
            image_numpy *= (2 ** 8 * 1.0 - 1)
            image_numpy = np.clip(image_numpy, 0, 255)

        elif imtype == np.uint16:
            image_numpy = np.clip(image_numpy, 0, 1)
            image_numpy *= (2 ** 16 * 1.0 - 1)
            image_numpy = np.clip(image_numpy, 0, 2**16-1)
        elif imtype == float:
            pass
    else:  # if it is a numpy array, do nothing

        if imtype == np.uint8:
            image_numpy = np.clip(input_image, 0, 1)
            image_numpy *= (2 ** 8 * 1.0 - 1)
            image_numpy = np.clip(image_numpy, 0, 255)
        elif imtype == np.uint16:
            image_numpy = np.clip(input_image, 0, 1)
            image_numpy *= (2 ** 16 * 1.0 - 1)
            image_numpy = np.clip(image_numpy, 0, 2**16-1)
        elif imtype == float:
            pass
        # image_numpy = input_image
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


def save_images(visuals, save_dir, name = ""):
    """Save images to the disk,

    Parameters:
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width

    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """

    img_name = ntpath.basename(name[0])

    for label, im_data in visuals.items():
        image_numpy = tensor2im(im_data)

        label_image_dir = save_dir+'/'+label+'/'
        if not os.path.exists(label_image_dir):
            os.makedirs(label_image_dir)

        file_name = '%s_%s.tif' % (img_name, label)
        save_path = os.path.join(label_image_dir, file_name)
        image_numpy = image_numpy.squeeze()
        imwrite(save_path, image_numpy)

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
    source = source.astype(float)

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



# Patch-wise operations for 3D volumes
def extract_3d_patches(volume, patch_size=(64, 64, 32), stride=(32, 32, 16)):
    """
    Extracts overlapping 3D patches from a given volume.
    
    Args:
        volume (np.ndarray): 3D input volume of shape (D, H, W).
        patch_size (tuple): Size of the extracted patches (depth, height, width).
        stride (tuple): Step size for the sliding window (depth, height, width).
    
    Returns:
        patches (list): List of 3D patches.
        indices (list): List of coordinates corresponding to patches.
    """
    D, H, W = volume.shape
    d_step, h_step, w_step = stride
    d_size, h_size, w_size = patch_size
    
    patches = []
    indices = []
    
    for d in range(0, D - d_size + 1, d_step):
        for h in range(0, H - h_size + 1, h_step):
            for w in range(0, W - w_size + 1, w_step):
                patch = volume[d:d+d_size, h:h+h_size, w:w+w_size]
                patches.append(patch)
                indices.append((d, h, w))
    
    return np.array(patches), indices


# reconstruct_from patches 
def reconstruct_from_patches(patch_preds, indices, volume_shape, patch_size, stride):
    """
    Reconstructs the full 3D volume from overlapping patches using average weighting.
    
    Args:
        patch_preds (np.ndarray): Array of predicted patches.
        indices (list): List of patch starting coordinates.
        volume_shape (tuple): Shape of the original 3D volume (D, H, W).
        patch_size (tuple): Size of each patch.
        stride (tuple): Stride used during patch extraction.
    
    Returns:
        np.ndarray: Reconstructed 3D volume.
    """
    D, H, W = volume_shape
    d_size, h_size, w_size = patch_size
    
    output_volume = np.zeros(volume_shape)
    weight_map = np.zeros(volume_shape)
    
    for patch, (d, h, w) in zip(patch_preds, indices):
        output_volume[d:d+d_size, h:h+h_size, w:w+w_size] += patch
        weight_map[d:d+d_size, h:h+h_size, w:w+w_size] += 1
    
    # Avoid division by zero
    weight_map[weight_map == 0] = 1
    return output_volume / weight_map

# create a gansisan mask 
def gaussian_3d(shape, sigma_scale=0.5):
    """
    Generates a 3D Gaussian weight mask.
    
    Args:
        shape (tuple): The shape of the patch (depth, height, width).
        sigma_scale (float): Scaling factor for Gaussian standard deviation.
    
    Returns:
        np.ndarray: A 3D Gaussian weight mask.
    """
    d, h, w = shape
    dz, dy, dx = np.meshgrid(
        np.linspace(-1, 1, d),
        np.linspace(-1, 1, h),
        np.linspace(-1, 1, w),
        indexing="ij"
    )
    
    # Compute squared distance from the center
    distance = dx**2 + dy**2 + dz**2
    sigma = sigma_scale  # Controls spread of Gaussian
    gaussian_mask = np.exp(-distance / (2 * sigma**2))
    
    return gaussian_mask

def reconstruct_from_patches_gaussian(patch_preds, indices, volume_shape, patch_size, stride):
    """
    Reconstructs a 3D volume using Gaussian-weighted blending of patches.
    
    Args:
        patch_preds (np.ndarray): Predicted patches.
        indices (list): List of patch coordinates.
        volume_shape (tuple): Shape of the original 3D volume (D, H, W).
        patch_size (tuple): Size of each patch (depth, height, width).
        stride (tuple): Stride used for patch extraction.
    
    Returns:
        np.ndarray: Reconstructed 3D volume.
    """
    D, H, W = volume_shape
    d_size, h_size, w_size = patch_size
    
    output_volume = np.zeros(volume_shape)
    weight_map = np.zeros(volume_shape)
    
    # Generate the Gaussian weight mask
    gaussian_mask = gaussian_3d(patch_size)

    for patch, (d, h, w) in zip(patch_preds, indices):
        # Apply Gaussian weighting
        weighted_patch = patch * gaussian_mask

        # Add weighted patch to volume
        output_volume[d:d+d_size, h:h+h_size, w:w+w_size] += weighted_patch
        weight_map[d:d+d_size, h:h+h_size, w:w+w_size] += gaussian_mask  # Track weight contributions

    # Normalize the volume by total weights to avoid over-representation
    weight_map[weight_map == 0] = 1  # Avoid division by zero
    return output_volume / weight_map
