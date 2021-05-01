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
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.cm as cm

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

def displace(X, param):
    """
    X = (dx, dy) で与えられたpx分 data_1 をずらし、差分とって標準偏差を計算

    Parameters
    ----------
    X : tuple (dx, dy)
        ずらすpx数を指定
    param : TYPE
        list [data_list, magnification, dlim]

    Returns
    -------
    std : float
        ずらす前後の差分についての標準偏差
    """
    dx = round(X[0])
    dy = round(X[1])
    data = param[0]
    magnification = param[1]
    dlim = param[2]
    
    data_0 = data[0]
    data_1 = data[1]
    
    ## ずらした部分にnp.nanが入らないように、ずらす最大値の分だけ周りを切り取る
    s0x = s0y = slice( int(dlim), int(len(data_0)-(dlim+magnification)) )
    ## dx, dyずらす を dx, dyずらして切り出す で対応
    s1x = slice( int(dlim+dx), int(len(data_0)+dx-(dlim+magnification)) )
    s1y = slice( int(dlim+dy), int(len(data_0)+dy-(dlim+magnification)) )
    
    cut_0 = data_0[s0x, s0y]
    cut_1 = data_1[s1x, s1y]
    diff = cut_1 - cut_0
    return diff#, s1x, s1y, data_0, data_1, s0x, s0y

def std_func(X, param):
    """
    X = (dx, dy) で与えられたpx分 data_1 をずらし、差分とって標準偏差を計算

    Parameters
    ----------
    X : tuple (dx, dy)
        ずらすpx数を指定
    param : TYPE
        list [data_list, magnification, dlim]

    Returns
    -------
    std : float
        ずらす前後の差分についての標準偏差
    """
    diff = displace(X, param)
    std = np.std(diff)
    return std

def px2urad(Res_minimize):
    """
    minimizeのresultのpx数を棒と垂直・水平な平面内での角度[urad]に変換し、更にplot用のtitleを作成
    Parameters
    ----------
    Res_minimize : Object
        output of sp.optimize.minimize()

    Returns
    -------
    Tilt_para : float64
        Angle [micro rad] included in the plane parallel to the autocollimetor stick
    Tilt_perp : float64
        Angle [micro rad] included in the plane perpendicular to the autocollimetor stick

    """
    F = 500e3 # 望遠鏡の焦点距離 [um]
    Zwopx = 2.4 # zwo183 : 2.4um per 1px

    Px_xy = Res_minimize["x"]/10
    Physical_length = Px_xy * Zwopx # 検出器位置での物理的距離
    Theta = np.arctan(Physical_length / F) 
    Tilt_urad_x, Tilt_urad_y = Theta / 2 * 1e6 # 反射するので鏡の傾きの2倍, microradなので1e6 
    Tilt_para = Tilt_urad_x # 棒に平行な平面内での角度
    Tilt_perp = Tilt_urad_y # 棒に垂直な平面内での角度
    
    Str_para = str(round(Tilt_para, 2))
    Str_perp = str(round(Tilt_perp, 2))
    Title = r"( $\Delta$para, $\Delta$perp ) = ( " + Str_para + " , " + Str_perp + " ) [micro rad]"
    return Tilt_para, Tilt_perp, Title

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
  
    from matplotlib.colors import Normalize
    import mpl_toolkits.axes_grid1
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
    px_old = 150 # 切り出すpx幅
    mgn = 10 # magnification subpixelまで細かくする時の、データ数の倍率
    px_lim = int(30*mgn)

    act_list = ["06", "07", "08", "09", "10", "11", "13", "14", "15", "16", "17", "19", "20", "21", "22"]
    #act_list = ["17"]
    df_cols = ["act", "para_e", "perp_e", "para_c", "perp_c"]
    df_res = pd.DataFrame(index=[], columns=df_cols)
    
    for act_num in act_list:
        print(act_num)
        data_mean = []
        data_limb = []
        data_center = []
        
        for i in range(0, 2):
            folder_path = "raw_data/210423/ex10_act" + act_num + "_" + name[i] + "/*.FIT"
            
            path_list = glob.glob(folder_path)
            
            if len(path_list) == 0: # globで何もマッチしなかったときに終了する
                print("path_list is empty!")
                sys.exit()
            
            data_mean_temp = np.empty((px_v, px_h))
            
            for path in path_list:
                data = fits_2darray(path)
                data_mean_temp = data + data_mean_temp
            
            data_mean_temp = data_mean_temp / len(path_list)
            
            data_mean.append(data_mean_temp)
            data_limb.append(data_mean_temp[100:100+px_old, 250:250+px_old])
            data_center.append(data_mean_temp[100:100+px_old, 0:0+px_old])
            
        ## interpolate ---------------------------------------------------------------  
        ip_limb = [fits_interpolate(data_limb[0], mgn), fits_interpolate(data_limb[1], mgn)]
        ip_center = [fits_interpolate(data_center[0], mgn), fits_interpolate(data_center[1], mgn)]
        
        data_diff = data_mean[1] - data_mean[0]
        
        ## minimize ----------------------------------------------------------------
        param_limb = [ip_limb, mgn, px_lim]
        res_limb = sp.optimize.minimize(fun=std_func, x0=(0,0), args=(param_limb,), method="Powell")
        diff_limb = displace(res_limb["x"], param_limb)
        angle_limb = px2urad(res_limb)
        
        param_center = [ip_center, mgn, px_lim]
        res_center = sp.optimize.minimize(fun=std_func, x0=(0,0), args=(param_center,), method="Powell")
        diff_center = displace(res_center["x"], param_center)
        angle_center = px2urad(res_center)
        
        ## for plot --------------------------------------------------------------
        fig = plt.figure(figsize=(10,15))
        gs = fig.add_gridspec(4, 2)
        
        ax_diff = image_plot(fig, path[16:26], gs[0, 0:2], data_diff, data_diff, 0, 100, "")
        ax_limb_0 = image_plot(fig, name[0]+argmax2d(data_limb[0])[1], gs[1, 1], data_limb[0], data_limb[0], 0, 100, "")
        ax_limb_1 = image_plot(fig, name[1]+argmax2d(data_limb[1])[1], gs[2, 1], data_limb[1], data_limb[0], 0, 100, "")
        ax_center_0 = image_plot(fig, name[0]+argmax2d(data_center[0])[1], gs[1, 0], data_center[0], data_limb[0], 0, 100, "")
        ax_center_1 = image_plot(fig, name[1]+argmax2d(data_center[1])[1], gs[2, 0], data_center[1], data_limb[0], 0, 100, "")
        ax_res_limb = image_plot(fig, angle_limb[2], gs[3, 1], diff_limb, diff_limb, 0, 100, "")
        ax_res_center = image_plot(fig, angle_center[2], gs[3, 0], diff_center, diff_center, 0, 100, "")
        
        
        fig.tight_layout()
        
        picname = mkfolder("/"+folder_path[9:15]) + folder_path[16:26] + "_" + name[0] + "_" + name[1] + ".png"
        fig.savefig(picname)
        
        record = pd.Series([act_num, angle_limb[0], angle_limb[1], angle_center[0], angle_center[1]], index=df_res.columns)        
        df_res = df_res.append(record, ignore_index=True)
    
    df_res.to_csv(mkfolder("/"+folder_path[9:15])+folder_path[16:20]+".csv")