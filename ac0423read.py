# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 19:49:35 2021

@author: swimc

0423のac実験データ用
"""

import numpy as np
import astropy.io.fits as fits
import scipy as sp
import glob
import scipy.signal
import sys
import pandas as pd
import time

import matplotlib.pyplot as plt
import matplotlib.cm as cm


def mkfolder(Suffix = ""):
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
    Filename = os.path.basename(__file__)
    Filename = Filename.replace(".py", "") + Suffix
    Folder = "mkfolder/" + Filename + "/"
    os.makedirs(Folder, exist_ok=True)
    return Folder


def fits_2darray(path):
    f = fits.open(path)
    pic = f[0]
    header = pic.header
    data = pic.data
    return data

def data_clip(Data, Vmin, Hmin, Width):
    """
    データをcenter と edge に切り取る

    Parameters
    ----------
    Data : 2d_array of float
        DESCRIPTION.
    Vmin : int
        vartical (y-axis) px min
    Hmin : int
        horizontal (x-axis) px min
    Width : int
        clip width

    Returns
    -------
    Cliped : 2d_array of float
        size : Width * Width

    """
    Vmax = Vmin + Width
    Hmax = Hmin + Width
    Cliped = Data[Vmin:Vmax, Hmin:Hmax]
    return Cliped

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
    zz_new_nandrop = zz_new[:-magnification, :-magnification]
    return zz_new_nandrop


def displace(Dx, Dy, Param):
    """
    

    Parameters
    ----------
    Dx : int
        subpx
    Dy : int
        subpx
    Param : TYPE
        DESCRIPTION.

    Returns
    -------
    Diff : TYPE
        2d_array

    """
    Dx = int(np.round(Dx))
    Dy = int(np.round(Dy))
    Data_0 = Param[0]
    Data_1 = Param[1]
    Subpx_lim = Param[2]
    
    if abs(Dx) > Subpx_lim:
        Dx = Subpx_lim
    if abs(Dy) > Subpx_lim:
        Dy = Subpx_lim
    
    S0min = Subpx_lim
    S0max = int(len(Data_0) - Subpx_lim) 
    
    S1xmin = int(S0min - Dx)
    S1xmax = int(S0max - Dx)
    S1ymin = int(S0min - Dy)
    S1ymax = int(S0max - Dy)
    
    Cut_0 = Data_0[S0min:S0max, S0min:S0max]
    Cut_1 = Data_1[S1xmin:S1xmax, S1ymin:S1ymax]
    Diff = Cut_1 - Cut_0
    return Diff

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
    diff = displace(X[0], X[1], param)
    std = np.std(diff)
    return std

def error_xy_loop(OptimizeResult, Param, Err_size, Err_mgn, Std_noise):
    Xopt, Yopt = OptimizeResult["x"]
    Std_opt = OptimizeResult["fun"]
    Xerr = np.empty(Err_size)
    Yerr = np.empty(Err_size)
    for i in range(Err_size):
        Xi = Xopt + i - Err_size/2
        Yi = Yopt + i - Err_size/2
        Xerr[i] = np.std(displace(Xi, Yopt, Param))
        Yerr[i] = np.std(displace(Xopt, Yi, Param))
    
    Threshold = Std_opt + (Std_opt - Std_noise) * Err_mgn
    Xeb = np.sum(np.where(Xerr<Threshold, 1, 0))
    Yeb = np.sum(np.where(Yerr<Threshold, 1, 0))
    Result = np.array([Xopt, Xeb, Yopt, Yeb])
    
    return Xerr, Yerr, Result

def subpx2urad(Subpx):
    """
    minimizeのresultのsubpx数を、主鏡のrocalな角度[urad]に変換

    Parameters
    ----------
    Subpx : float
        output of sp.optimize.minimize()["x"][0] or [1] (not tuple)

    Returns
    -------
    Tilt_urad : float
        Angle [micro rad]

    """
    F = 500e3 # 望遠鏡の焦点距離 [um]
    Zwopx = 2.4 # zwo183 : 2.4um per 1px

    Px = Subpx / 10
    Physical_length = Px * Zwopx # 検出器位置での物理的距離
    Theta = np.arctan(Physical_length / F) 
    Tilt_urad = Theta / 2 * 1e6 # 反射するので鏡の傾きの2倍, microradなので1e6
    return Tilt_urad

def urad2title(Tilt_urad_x, Tilt_urad_y):
    """
    x,y方向pxから変換した [urad] を、figureのtitle用にテキストに変換
    para : Angle [micro rad] included in the plane parallel to the autocollimetor stick
    perp : Angle [micro rad] included in the plane perpendicular to the autocollimetor stick
    
    Parameters
    ----------
    Tilt_urad_x : float
        Angle [micro rad] (x)
    Tilt_urad_y : float
        Angle [micro rad] (y)

    Returns
    -------
    Title : str
        DESCRIPTION.

    """
    Tilt_para = Tilt_urad_x # 棒に平行な平面内での角度
    Tilt_perp = Tilt_urad_y # 棒に垂直な平面内での角度
    
    Str_para = str(round(Tilt_para, 2))
    Str_perp = str(round(Tilt_perp, 2))
    Title = r"( $\Delta$para, $\Delta$perp ) = ( " + Str_para + " , " + Str_perp + " ) [micro rad]"
    return Title

def image_plot(fig, title, position, c, c_scale, min_per=0, max_per=100, cbar_title=""):
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

def err_plot(Fig, Title, Position, Xopt, Yarr, Std_opt, Std_noise, Err_mgn):
    Size = len(Yarr)
    Xmin = Xopt - Size/2
    Xmax = Xopt + Size/2
    Xarr = subpx2urad(np.linspace(Xmin, Xmax, num=Size))
    Threshold_eb = Std_opt + (Std_opt - Std_noise) * Err_mgn
    Threshold_ymax = Std_opt + (Std_opt - Std_noise) * 0.5
    
    Ax = Fig.add_subplot(Position)
    Ax.plot(Xarr, Yarr)
    Ax.set_xlabel("[urad]")
    Ax.set_ylim(Yarr.min(), Threshold_ymax)
    Ax.set_title(Title)
    
    Ax.vlines(subpx2urad(Xopt), Yarr.min(), Yarr.max())
    Ax.hlines(Threshold_eb, Xarr.min(), Xarr.max())
    
    return Ax

if __name__ == '__main__':
    start = time.time()
    name = ["-500", "+500"]
    px_v, px_h = 384, 512
    px_clip_width = 250 # 切り出すpx幅
    px_lim = 50
    mgn = 10 # magnification subpixelまで細かくする時の、データ数の倍率
    subpx_lim = int(px_lim * mgn)
    
    act_list = ["06", "07", "08", "09", "10", "11", "13", "14", "15", "16", "17", "19", "20", "21", "22"]
    #act_list = ["17"]
    df_cols = ["act",
               "para_e", "para_e_eb",
               "perp_e", "perp_e_eb", 
               "para_c", "para_c_eb", 
               "perp_c", "perp_c_eb",
               "e_std", "c_std", "noise"]
    
    df_res = pd.DataFrame(index=[], columns=df_cols)
    
    for act_num in act_list:
        print(act_num)
        data_mean = []
        data_e = []
        data_c = []
        data_noise = []
        data_noise_std = []
        
        for i in range(0, 2):
            folder_path = "raw_data/210423/ex10_act" + act_num + "_" + name[i] + "/*.FIT"
            
            path_list = glob.glob(folder_path)
            
            if len(path_list) == 0: # globで何もマッチしなかったときに終了する
                print("path_list is empty!")
                sys.exit()
            
            data_mean_temp = np.zeros((px_v, px_h))
            
            for path in path_list:
                data = fits_2darray(path)
                data_mean_temp = data + data_mean_temp
            
            data_mean_temp = data_mean_temp / len(path_list)
            
            data_e_temp = data_clip(data_mean_temp, 50, 200, px_clip_width)
            data_c_temp = data_clip(data_mean_temp, 75, 0, px_clip_width)
            
            data_mean.append(data_mean_temp)
            data_e.append(data_e_temp)
            data_c.append(data_c_temp)
            
            ## read out noise -----------------------------------------------
            data_noise_temp = np.zeros((px_v, px_h))
            for path in path_list:
                data = fits_2darray(path)
                data_noise_temp = data_noise_temp + (data -data_mean_temp)**2
            
            data_noise_temp = np.sqrt( data_noise_temp / len(path_list) )
            data_noise.append(data_noise_temp)
            data_noise_std.append(np.std(data_noise_temp))
        
        data_noise_std.append(np.sqrt(data_noise_std[0]**2+data_noise_std[1]**2))
        
        ## interpolate ---------------------------------------------------------------  
        ip_e = [fits_interpolate(data_e[0], mgn), fits_interpolate(data_e[1], mgn)]
        ip_c = [fits_interpolate(data_c[0], mgn), fits_interpolate(data_c[1], mgn)]
        
        data_diff = data_mean[1] - data_mean[0]
        
        ## minimize ----------------------------------------------------------------
        param_e = ip_e + [subpx_lim]
        param_c = ip_c + [subpx_lim]
        
        res_e = sp.optimize.minimize(fun=std_func, x0=(0,0), args=(param_e), method="Powell") 
        res_c = sp.optimize.minimize(fun=std_func, x0=(0,0), args=(param_c), method="Powell")        
       
        diff_e = displace(res_e["x"][0], res_e["x"][1], param_e)
        diff_c = displace(res_c["x"][0], res_c["x"][1], param_c)
        print(time.time() - start)
        
        ## error_bar ------------------------------------------------------------
        
        err_size, err_mgn = 20, 0.05
        x_err_e, y_err_e, eb_e_px = error_xy_loop(res_e, param_e, err_size, err_mgn, data_noise_std[2])
        x_err_c, y_err_c, eb_c_px = error_xy_loop(res_c, param_c, err_size, err_mgn, data_noise_std[2])
        
        eb_e_urad = subpx2urad(eb_e_px)
        eb_c_urad = subpx2urad(eb_c_px)

        angle_e = urad2title(eb_e_urad[0], eb_e_urad[2])
        angle_c = urad2title(eb_c_urad[0], eb_c_urad[2])

        record = np.concatenate([np.atleast_1d(int(act_num)), eb_e_urad, eb_c_urad, np.array([res_e["fun"], res_c["fun"], data_noise_std[2]])])
        
        df_res = df_res.append(pd.Series(record, index = df_res.columns), 
                               ignore_index=True)    

        print(time.time()-start)
        
        ## for plot --------------------------------------------------------------
        fig = plt.figure(figsize=(10,15))
        gs = fig.add_gridspec(5,2)
        fig.suptitle(folder_path[9:15] + " act" + act_num)
        
        ax_5 = image_plot(fig, "-500", gs[0, 0], data_mean[0], data_mean[0])
        ax_0 = image_plot(fig, "+500", gs[0, 1], data_mean[1], data_mean[0])
        ax_diff = image_plot(fig, "diff {+500} - {-500}", gs[1,0:2], data_diff, data_diff)
        ax_res_e = image_plot(fig, angle_e, gs[2,1], diff_e, data_diff)
        ax_res_c = image_plot(fig, angle_c, gs[2,0], diff_c, data_diff)
        
        ax_err_xe = err_plot(fig, "xe", gs[3, 1], res_e["x"][0], x_err_e, res_e["fun"], data_noise_std[2], err_mgn)
        ax_err_ye = err_plot(fig, "ye", gs[4, 1], res_e["x"][1], y_err_e, res_e["fun"], data_noise_std[2], err_mgn)
        ax_err_xc = err_plot(fig, "xc", gs[3, 0], res_e["x"][0], x_err_c, res_c["fun"], data_noise_std[2], err_mgn)
        ax_err_yc = err_plot(fig, "yc", gs[4, 0], res_e["x"][1], y_err_c, res_c["fun"], data_noise_std[2], err_mgn)
        
        fig.tight_layout()
        
        picname = mkfolder("/"+folder_path[9:15]) + folder_path[16:26] + "_" + name[0] + "_" + name[1] + ".png"
        fig.savefig(picname)
    
        record = pd.Series(np.concatenate([np.atleast_1d(int(act_num)), eb_e_urad, eb_c_urad, np.array([res_e["fun"], res_c["fun"], data_noise_std[2]])]),
                           index = df_res.columns)
        
        df_res = df_res.append(record, ignore_index=True)
        
    df_res.to_csv(mkfolder("/"+folder_path[9:15])+folder_path[16:20]+".csv")