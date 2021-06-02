from __future__ import print_function
import math
import os
import random
import copy
import scipy
import imageio
import string
import numpy as np
from skimage.transform import resize
try:  # SciPy >= 0.19
    from scipy.special import comb
except ImportError:
    from scipy.misc import comb

from scipy.ndimage import affine_transform
from math import pi
from transforms3d import affines, euler

def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """

    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i

def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.

       Control points should be a list of lists, or list of tuples
       such as [ [1,1], 
                 [2,3], 
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000

        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)   ])
    
    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals

def data_augmentation(x, y, prob=0.5):
    # augmentation by flipping
    cnt = 3
    while random.random() < prob and cnt > 0:
        degree = random.choice([0, 1, 2])
        x = np.flip(x, axis=degree)
        y = np.flip(y, axis=degree)
        cnt = cnt - 1

    return x, y

def nonlinear_transformation(x, prob=0.5):
    if random.random() >= prob:
        return x
    points = [[0, 0], [random.random(), random.random()], [random.random(), random.random()], [1, 1]]
    xpoints = [p[0] for p in points]
    ypoints = [p[1] for p in points]
    xvals, yvals = bezier_curve(points, nTimes=100000)
    if random.random() < 0.5:
        # Half change to get flip
        xvals = np.sort(xvals)
    else:
        xvals, yvals = np.sort(xvals), np.sort(yvals)
    nonlinear_x = np.interp(x, xvals, yvals)
    return nonlinear_x

def local_pixel_shuffling(x, prob=0.5):
    if random.random() >= prob:
        return x
    image_temp = copy.deepcopy(x)
    orig_image = copy.deepcopy(x)
    _, img_rows, img_cols, img_deps = x.shape
    num_block = 10000
    for _ in range(num_block):
        block_noise_size_x = random.randint(1, img_rows//10)
        block_noise_size_y = random.randint(1, img_cols//10)
        block_noise_size_z = random.randint(1, img_deps//10)
        noise_x = random.randint(0, img_rows-block_noise_size_x)
        noise_y = random.randint(0, img_cols-block_noise_size_y)
        noise_z = random.randint(0, img_deps-block_noise_size_z)
        window = orig_image[0, noise_x:noise_x+block_noise_size_x, 
                               noise_y:noise_y+block_noise_size_y, 
                               noise_z:noise_z+block_noise_size_z,
                           ]
        window = window.flatten()
        np.random.shuffle(window)
        window = window.reshape((block_noise_size_x, 
                                 block_noise_size_y, 
                                 block_noise_size_z))
        image_temp[0, noise_x:noise_x+block_noise_size_x, 
                      noise_y:noise_y+block_noise_size_y, 
                      noise_z:noise_z+block_noise_size_z] = window
    local_shuffling_x = image_temp

    return local_shuffling_x

def image_in_painting(x):
    _, img_rows, img_cols, img_deps = x.shape
    cnt = 5
    while cnt > 0 and random.random() < 0.95:
        block_noise_size_x = random.randint(img_rows//6, img_rows//3)
        block_noise_size_y = random.randint(img_cols//6, img_cols//3)
        block_noise_size_z = random.randint(img_deps//6, img_deps//3)
        noise_x = random.randint(3, img_rows-block_noise_size_x-3)
        noise_y = random.randint(3, img_cols-block_noise_size_y-3)
        noise_z = random.randint(3, img_deps-block_noise_size_z-3)
        x[:, 
          noise_x:noise_x+block_noise_size_x, 
          noise_y:noise_y+block_noise_size_y, 
          noise_z:noise_z+block_noise_size_z] = np.random.rand(block_noise_size_x, 
                                                               block_noise_size_y, 
                                                               block_noise_size_z, ) * 1.0
        cnt -= 1
    return x

def image_out_painting(x):
    _, img_rows, img_cols, img_deps = x.shape
    image_temp = copy.deepcopy(x)
    x = np.random.rand(x.shape[0], x.shape[1], x.shape[2], x.shape[3], ) * 1.0
    block_noise_size_x = img_rows - random.randint(3*img_rows//7, 4*img_rows//7)
    block_noise_size_y = img_cols - random.randint(3*img_cols//7, 4*img_cols//7)
    block_noise_size_z = img_deps - random.randint(3*img_deps//7, 4*img_deps//7)
    noise_x = random.randint(3, img_rows-block_noise_size_x-3)
    noise_y = random.randint(3, img_cols-block_noise_size_y-3)
    noise_z = random.randint(3, img_deps-block_noise_size_z-3)
    x[:, 
      noise_x:noise_x+block_noise_size_x, 
      noise_y:noise_y+block_noise_size_y, 
      noise_z:noise_z+block_noise_size_z] = image_temp[:, noise_x:noise_x+block_noise_size_x, 
                                                       noise_y:noise_y+block_noise_size_y, 
                                                       noise_z:noise_z+block_noise_size_z]
    cnt = 4
    while cnt > 0 and random.random() < 0.95:
        block_noise_size_x = img_rows - random.randint(3*img_rows//7, 4*img_rows//7)
        block_noise_size_y = img_cols - random.randint(3*img_cols//7, 4*img_cols//7)
        block_noise_size_z = img_deps - random.randint(3*img_deps//7, 4*img_deps//7)
        noise_x = random.randint(3, img_rows-block_noise_size_x-3)
        noise_y = random.randint(3, img_cols-block_noise_size_y-3)
        noise_z = random.randint(3, img_deps-block_noise_size_z-3)
        x[:, 
          noise_x:noise_x+block_noise_size_x, 
          noise_y:noise_y+block_noise_size_y, 
          noise_z:noise_z+block_noise_size_z] = image_temp[:, noise_x:noise_x+block_noise_size_x, 
                                                           noise_y:noise_y+block_noise_size_y, 
                                                           noise_z:noise_z+block_noise_size_z]
        cnt -= 1
    return x
                

def img_interpolate(img, mask, target_size):
    x = np.linspace(1, img.shape[0], img.shape[0])
    y = np.linspace(1, img.shape[1], img.shape[1])
    z = np.linspace(1, img.shape[2], img.shape[2])
    img_interpolator = rgi((x,y,z), img, method='linear')
    mask_interpolator = rgi((x,y,z), mask, method='nearest')
    xi = np.linspace(1, img.shape[0], target_size[0])
    yi = np.linspace(1, img.shape[1], target_size[1])
    zi = np.linspace(1, img.shape[2], target_size[2])
    xq, yq, zq = np.meshgrid(xi, yi, zi)
    pts = list(zip(xq.flatten(),yq.flatten(),zq.flatten()))
    return img_interpolator(pts).reshape(target_size), mask_interpolator(pts).reshape(target_size)

def img_pad(img, mask, pad_dims):
    """Pad input image vol to given voxel dims"""
    x_pad0, x_pad1, y_pad0, y_pad1, z_pad0, z_pad1 = 0,0,0,0,0,0
    dims = img.shape[1:]
    if dims[0] < pad_dims[0]:
        x_pad0 = (pad_dims[0] - dims[0]) // 2
        x_pad1 = pad_dims[0] - dims[0] - x_pad0
    if dims[1] < pad_dims[1]:
        y_pad0 = (pad_dims[1] - dims[1]) // 2
        y_pad1 = pad_dims[1] - dims[1] - y_pad0
    if dims[2] < pad_dims[2]:
        z_pad0 = (pad_dims[2] - dims[2]) // 2
        z_pad1 = pad_dims[2] - dims[2] - z_pad0

    padding = ((0, 0), (x_pad0, x_pad1), (y_pad0, y_pad1), (z_pad0, z_pad1))

    img  = np.pad(img,  padding, 'constant', constant_values=0)
    mask = np.pad(mask, padding, 'constant', constant_values=0)
    return img, mask

def img_unpad(img, dims):
    """Unpad image vol back to original input dimensions"""
    pad_dims = img.shape
    xmin, ymin, zmin = 0, 0, 0
    if pad_dims[0] > dims[0]:
        xmin = (pad_dims[0] - dims[0]) // 2
    if pad_dims[1] > dims[1]:
        ymin = (pad_dims[1] - dims[1]) // 2
    if pad_dims[2] > dims[2]:
        zmin = (pad_dims[2] - dims[2]) // 2
    return img[xmin : xmin + dims[0],
               ymin : ymin + dims[1],
               zmin : zmin + dims[2]]

def img_crop(img, mask, crop_size, lesion_frac=0.8, cb_frac=0.05, edge_frac=0.05, outlier_frac=0.05):
    """Sample a random image subvol/patch from larger input vol"""
    # Pick the location for the patch centerpoint
    good_inds = (mask.squeeze() > 0.05).nonzero() # sample all lesion voxels
    n_pos = good_inds[0].size
    if n_pos >= 100:
        pass
    else:
        lesion_frac = n_pos/(n_pos + 100)
    rand = np.random.random()
    img_sq = np.squeeze(img)
    if rand < lesion_frac:
        good_inds = (mask.squeeze() != 0).nonzero() # sample all lesion voxels
    # Sample edges
    elif rand < lesion_frac + edge_frac:
        img_bin = np.zeros(img_sq.shape)
        img_bin = np.where(img_sq > 0, 1, 0)
        grads = np.array(np.gradient(img_bin))
        good_inds = np.max(grads, axis=0).nonzero()
    # Sample cerebellum
    elif rand < lesion_frac + edge_frac + cb_frac:
        good_inds = np.array((img[0,] != 0).nonzero()) # sample all brain voxels
        # Take indices from most inferior third
        idx = np.where(good_inds[2] <= np.max(good_inds[2])/4)
        good_inds = tuple(np.squeeze(good_inds[:,idx]))
    elif rand < lesion_frac + edge_frac + cb_frac + outlier_frac:
        good_inds = np.array((img[0,] < (np.mean(img[0,] - np.std(img[0,])*2))).nonzero()) # sample low intensity voxels
    else:
        good_inds = (img[0,] != 0).nonzero() # sample all brain voxels
    if not good_inds[0].size > 0: # If no lesion present or some other error
        good_inds = (img[0,] != 0).nonzero() # sample all brain voxels
    i_center = np.random.randint(good_inds[0].size)
    if good_inds[0].size > 1:
        xmin = good_inds[0][i_center] - crop_size[0] // 2
        ymin = good_inds[1][i_center] - crop_size[1] // 2
        zmin = good_inds[2][i_center] - crop_size[2] // 2
    else:
        xmin = good_inds[0] - crop_size[0] // 2
        ymin = good_inds[1] - crop_size[1] // 2
        zmin = good_inds[2] - crop_size[2] // 2

    # Make sure centerpoint is not too small
    if xmin < 0: xmin = 0
    if ymin < 0: ymin = 0
    if zmin < 0: zmin = 0

    # Make sure centerpoint is not too big
    max_sizes = np.array(img.shape[1:]) - crop_size
    if xmin > max_sizes[0]: xmin = max_sizes[0]
    if ymin > max_sizes[1]: ymin = max_sizes[1]
    if zmin > max_sizes[2]: zmin = max_sizes[2]
    
    img  =  img[:, xmin : xmin + crop_size[0], 
                   ymin : ymin + crop_size[1],
                   zmin : zmin + crop_size[2]]
    mask = mask[:, xmin : xmin + crop_size[0], 
                   ymin : ymin + crop_size[1],
                   zmin : zmin + crop_size[2]]
    return img, mask

def img_warp(img, mask, theta_max=15, offset_max=0, scale_max=1.1, shear_max=0.1):
    """Training data augmentation with random affine transformation"""
    # Rotation
    vec = np.random.normal(0, 1, 3)
    vec /= np.sqrt(np.sum(vec ** 2))
    theta = np.random.uniform(- theta_max, theta_max, 1) * pi / 180
    R = euler.axangle2mat(vec, theta)
    
    # Scale/zoom
    sign = -1 if np.random.random() < 0.5 else 1
    Z = np.ones(3) * np.random.uniform(1, scale_max, 1) ** sign
    
    # Translation
    c_in = np.array(img.shape[1:]) // 2
    offset = np.random.uniform(- offset_max, offset_max, 3)
    T = - (c_in).dot((R * Z).T) + c_in + offset
    
    # Shear
    S = np.random.uniform(- shear_max, shear_max, 3)
    
    # Compose affine
    mat = affines.compose(T, R, Z, S)
    
    # Apply warp
    img_warped  = np.zeros_like(img)
    mask_warped = np.zeros_like(mask)
    for i in range(len(img)):
        img_warped[i,] = affine_transform(img[i,], mat, order=1) # Trilinear
    mask_warped[0,] = affine_transform(mask[0,], mat, order=0)   # Nearest neighbor
    
    return img_warped, mask_warped

def standardize(image):
    if len(image.shape) < 4:
        image = image[np.newaxis]
    out = np.zeros_like(image)
    for i, img in enumerate(image):
        img_voxels = img[img>0]
        if img_voxels.size > 0:
            img = (img-np.mean(img_voxels))/(np.std(img_voxels))
        out[i,] = img
    return out

def normalize(image):
    if len(image.shape) < 4:
        image = image[np.newaxis]
    out = np.zeros_like(image)
    for i, img in enumerate(image):
        img = (img-np.min(img))/(np.max(image)-np.min(image))
        out[i,] = img
    return out

def get_input_patch_shape(model):
    return model.layers[0].input_shape[2:]
