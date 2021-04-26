# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 19:49:35 2021

@author: swimc
"""

import numpy as np
import astropy.io.fits as fits
import scipy as sp
import glob
import scipy.signal
import sys

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import mpl_toolkits.axes_grid1

def mkfolder(suffix = ""):
    import os
    """    
    Parameters
    ----------
    suffix : str, optional
        The default is "".

    Returns
    -------
    str ( script name + suffix )
    """
    filename = os.path.basename(__file__)
    filename = filename.replace(".py", "") + suffix
    folder = "mkfolder/" + filename + "/"
    os.makedirs(folder, exist_ok=True)
    return folder

def fits_2darray(path):
    f = fits.open(path)
    pic = f[0]
    header = pic.header
    data = pic.data
    return data

def fits_interpolate(array_2d, magnification): 
    range_max = len(array_2d)
    
    x_arr = y_arr = np.arange(range_max)
    xx, yy = np.meshgrid(x_arr, y_arr)
    x_old = xx.flatten()
    y_old = yy.flatten()
    xy_old = np.stack([x_old, y_old], axis=1)
    z_old = array_2d.flatten()
    
    x_new = y_new = np.linspace(0, range_max, range_max*magnification)
    xx_new, yy_new = np.meshgrid(x_new, y_new)
    zz_new = sp.interpolate.griddata(xy_old, z_old, (xx_new, yy_new), "cubic")
    return zz_new    
    
def argmax2d(ndim_array):
    idx = np.unravel_index(np.argmax(ndim_array), ndim_array.shape)
    return idx, str(idx)

def image_plot(fig, title, position, c, c_scale, min_per, max_per, cbar_title):
    cmap1 = cm.jet
    fs = 10
    c_scale = np.sort(c_scale.flatten())
    cbar_min = c_scale[round(len(c_scale)*min_per/100)]
    cbar_max = c_scale[round(len(c_scale)*max_per/100)-1]
    
    ax = fig.add_subplot(position)
    ax.imshow(c, interpolation="nearest", cmap=cmap1, vmin=cbar_min, vmax=cbar_max, origin="lower")
    ax.set_title(title, fontsize=fs)
  
    divider = mpl_toolkits.axes_grid1.make_axes_locatable(ax)
    cax = divider.append_axes('right', '5%', pad='3%')
    norm = Normalize(vmin=cbar_min, vmax=cbar_max)
    mappable = cm.ScalarMappable(norm = norm, cmap = cmap1)
    cbar = fig.colorbar(mappable, ax=ax, cax=cax)
    cbar.set_label(cbar_title, fontsize=fs)
        
    return ax

if __name__ == '__main__':
    
    name = ["-500", "+500"]
    px_v, px_h = 384, 512
    mgn = 10 # magnification subpixelまで細かくする時の、データ数の倍率

    data_mean = []
    data_limb = []
    data_center = []
    
    
    for i in range(0, 2):
        path = "raw_data/210421/ex04_act09_" + name[i] + "/*.FIT"
        
        path_list = glob.glob(path)
        
        if len(path_list) == 0: # globで何もマッチしなかったときに終了する
            print("path_list is empty!")
            sys.exit()
        
        data_mean_temp = np.empty((px_v, px_h))
        
        for path in path_list:
            data = fits_2darray(path)
            data_mean_temp = data + data_mean_temp
        
        data_mean_temp = data_mean_temp / len(path_list)
        
        data_mean.append(data_mean_temp)
        data_limb.append(data_mean_temp[125:275, 275:425])
        data_center.append(data_mean_temp[100:300, 0:200])
        
    ## interpolate ---------------------------------------------------------------  
    ip_limb = [fits_interpolate(data_limb[0], mgn), fits_interpolate(data_limb[1], mgn)]
    ip_center = [fits_interpolate(data_center[0], mgn), fits_interpolate(data_center[1], mgn)]
    
    data_diff = data_mean[1] - data_mean[0]
    
    
    """
    ## correlate func --------------------------------------------------------
    corr_limb = scipy.signal.correlate2d(data_limb[0]-data_limb[0].mean(), data_limb[1]-data_limb[1].mean())
    corr_limb = corr_limb[100:300, 100:300]
    corr_center = scipy.signal.correlate2d(data_center[0]-data_center[0].mean(), data_center[1]-data_center[1].mean())
    corr_center = corr_center[100:300, 100:300]
    """
    
    
    ## for plot --------------------------------------------------------------
    fig = plt.figure(figsize=(10,15))
    gs = fig.add_gridspec(5, 2)
    
    ax_diff = image_plot(fig, path[16:26], gs[0, 0:2], data_diff, data_diff, 0, 100, "")
    ax_limb_0 = image_plot(fig, name[0]+argmax2d(data_limb[0])[1], gs[1, 1], data_limb[0], data_limb[0], 0, 100, "")
    ax_limb_1 = image_plot(fig, name[1]+argmax2d(data_limb[1])[1], gs[3, 1], data_limb[1], data_limb[0], 0, 100, "")
    ax_center_0 = image_plot(fig, name[0]+argmax2d(data_center[0])[1], gs[1, 0], data_center[0], data_limb[0], 0, 100, "")
    ax_center_1 = image_plot(fig, name[1]+argmax2d(data_center[1])[1], gs[2, 0], data_center[1], data_limb[0], 0, 100, "")
    
    fig.tight_layout()
    
    picname = mkfolder("/"+path[9:15]) + path[16:26] + "_" + name[0][:2] + "_" + name[1][:2] + ".png"
    fig.savefig(picname)