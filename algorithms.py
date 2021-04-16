import os
import sys

import scipy.ndimage
from skimage.io import imread

import matplotlib.pyplot as plt

import numpy as np

from PIL import Image

def guassian_blur(img, img_mask, sigma=5):
    mask = img_mask[:,:,:1].astype(np.float)
    filter = scipy.ndimage.filters.gaussian_filter(img*mask, sigma=(sigma, sigma, 0))
    weights = scipy.ndimage.filters.gaussian_filter(mask, sigma=(sigma, sigma, 0))
    filter /= weights + 0.001

    filter = filter.astype(np.uint8)
    inv_mask = (mask < 1.0)
    filter -= filter*inv_mask
    img = (img*inv_mask)
    img += filter
    img = img.astype(np.uint8)
    return img

def pixelization(img, mask_img):
    dim_x, dim_y = img.shape[0], img.shape[1]
    img = Image.fromarray(img)
    inv_mask = (mask_img < 1.0)
    imgSmall = img.resize((dim_x//16,dim_y//16),resample=Image.BILINEAR)
    imgSmall = imgSmall.resize(img.size,Image.NEAREST)
    imgSmall -= imgSmall*inv_mask
    img = (img*inv_mask)
    img += imgSmall
    img = img.astype(np.uint8)
    return np.array(img)

def pixel_sort(img, img_mask):
    # Stores beginning and end of row
    selected_row = [-1,-1]
    # Sort pixels horizontally
    for row in range(len(img_mask)):
        for col in range(len(img_mask[row])):
            val = img_mask[row][col][0]
            if val == 255:
                if selected_row[0] == -1:
                    selected_row[0] = col
            else:
                if selected_row[0] != -1:
                    selected_row[1] = col
                    np.random.shuffle(img[row][selected_row[0]:selected_row[1]])
                    selected_row = [-1, -1]
        selected_row = [-1, -1]
    return img

def fill_in(img, img_mask):

    sumPixels = np.array([0, 0, 0]) #RGB
    pixels = []
    N = [0] #Number of pixels in group

    for row in range(len(img_mask)):
        for col in range(len(img_mask[row])):
            val = img_mask[row][col][0]
            if val == 255:
                fill_in_dfs(col, row, pixels, img, img_mask, sumPixels, N)
                avgPixel = sumPixels / N[0]
                for i in pixels:
                    img[i[0]][i[1]][0] = avgPixel[0]
                    img[i[0]][i[1]][1] = avgPixel[1]
                    img[i[0]][i[1]][2] = avgPixel[2]

                # Reset data structres
                sumPixels = np.array([0, 0, 0])
                pixels = []
                N[0] = 0

    return img

def fill_in_dfs(col, row, pixels, img, img_mask, sumPixels, N):

    sumPixels = np.add(sumPixels, img[row][col])
    print(sumPixels)
    N[0] += 1
    pixels.append((row, col))
    img_mask[row][col][0] = 0

    #left
    if col > 0 and img_mask[row][col-1][0] == 255:
        fill_in_dfs(col-1, row, pixels, img, img_mask, sumPixels, N)
    #top
    if row > 0 and img_mask[row-1][col][0] == 255:
        fill_in_dfs(col, row-1, pixels, img, img_mask, sumPixels, N)
    #right
    if col < len(img_mask[0])-1 and img_mask[row][col+1][0] == 255:
        fill_in_dfs(col+1, row, pixels, img, img_mask, sumPixels, N)
    #bottom
    if row > len(img_mask)-1 and img_mask[row+1][col][0] == 255:
        fill_in_dfs(col, row+1, pixels, img, img_mask, sumPixels, N)