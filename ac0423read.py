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
    os.makedirs(folder, exist_ok = True)
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

def displace(Dx, Dy, Param):
    """
    (Dx, Dy) で与えられたpx分 data_1 をずらし、差分とって標準偏差を計算

    Parameters
    ----------
    Dx : int
        x方向にずらすpx数を指定
    Dy : int
        y方向にずらすpx数を指定
    Param : list
        list [data_list, magnification, dlim]

    Returns
    -------
    std : float
        ずらす前後の差分についての標準偏差
    """
    Dx = np.round(Dx)
    Dy = np.round(Dy)
    Data = Param[0]
    Magnification = Param[1]
    Dlim = Param[2]
    
    Data_0 = Data[0]
    Data_1 = Data[1]
    
    ## ずらした部分にnp.nanが入らないように、ずらす最大値の分だけ周りを切り取る
    S0x = S0y = slice( int(Dlim), int(len(Data_0)-(Dlim+Magnification)) )
    ## dx, dyずらす を dx, dyずらして切り出す で対応
    S1x = slice( int(Dlim+Dx), int(len(Data_0)+Dx-(Dlim+Magnification)) )
    S1y = slice( int(Dlim+Dy), int(len(Data_0)+Dy-(Dlim+Magnification)) )
    
    Cut_0 = Data_0[S0x, S0y]
    Cut_1 = Data_1[S1x, S1y]
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

def error_x_func(X, Param_res):
    """
    paramで与えられた y は固定で、Xをずらして標準偏差を計算

    Parameters
    ----------
    X : float (not tuple)
        x方向にずらすpx数を指定
    Param_res : list [[data_list, magnification, dlim], OptimizeResult, sigma_mgn]
        sigma_mgnは標準偏差に対するエラーバーの長さ

    Returns
    -------
    Std : TYPE
        DESCRIPTION.

    """
    
    Param = Param_res[0]
    Opresult = Param_res[1]
    Sigma_mgn = Param_res[2]
    
    Diff = displace(X, Opresult["x"][1], Param)
    Std = np.std(Diff)
    Result = abs( Std - Sigma_mgn * Opresult["fun"])
    return Result

def error_y_func(Y, Param_res):
    Param = Param_res[0]
    Opresult = Param_res[1]
    Sigma_mgn = Param_res[2]
    Diff = displace(Opresult["x"][0], Y, Param)
    Std = np.std(Diff)
    Result = abs( Std - Sigma_mgn * Opresult["fun"])
    return Result

def error_bar_bounds(Param, OptimizeResult, Sigma_mgn):
    """
    x, yについて error barの最大値と最小値を、制約付き最小化で求めるためのラッパー
    OptimizeResultで求めた標準偏差の最小値 ["fun"] に対して、 fun*sigma_mgn を閾値として、標準偏差がfun*sigma_mgnまでの範囲をエラーバーとする
    例） OptimizeResult["fun"] = 796, OptimizeResult["x"] = (4.9, -67.7), Sigma_mgn = 2であるとき
    x方向のエラーバー -> y=-67.7に固定して fun = 796*2 になる x を探す
    この時、エラーバーには上限と下限があるので、x を探す範囲に x > 4.9 or x < 4.9 の制約をつけてそれぞれについて fun = 796*2 を探す
    
    Parameters
    ----------
    Param : list [data_list, magnification, dlim]
        DESCRIPTION.
    OptimizeResult : Object
        DESCRIPTION.
    Sigma_mgn : float
        標準偏差に対するエラーバーの長さの計算用、標準偏差 * sigma_mgnがエラーバーの幅を決定

    Returns
    -------
    Result : list [x下限, x最小値, x上限, y下限, y最小値, y上限]
        x : parallel方向, y : perpendicular方向
        "最小値"は、inputの OptimizeResultで計算済みの、"fun"を最小にするような x ,y
    """
    
    Param_res = [Param, OptimizeResult, Sigma_mgn]
    
    Xmin = sp.optimize.minimize(fun = error_x_func,
                                x0 = ( OptimizeResult["x"][0] - 1 ), 
                                args = (Param_res),
                                constraints = {"type" : "ineq", "fun" : lambda x: - ( x - OptimizeResult["x"][0] ) },
                                method = "COBYLA")
 
    Xmax = sp.optimize.minimize(fun = error_x_func, 
                                x0 = ( OptimizeResult["x"][0] + 1 ), 
                                args = (Param_res),
                                constraints = {"type" : "ineq", "fun" : lambda x: + ( x - OptimizeResult["x"][0] ) },
                                method = "COBYLA")
    
    Ymin = sp.optimize.minimize(fun = error_y_func,
                                x0 = ( OptimizeResult["x"][1] - 1 ), 
                                args = (Param_res),
                                constraints = {"type" : "ineq", "fun" : lambda x: - ( x - OptimizeResult["x"][1] ) },
                                method = "COBYLA")
        
    Ymax = sp.optimize.minimize(fun = error_y_func, 
                                x0 = ( OptimizeResult["x"][1] + 1 ), 
                                args = (Param_res),
                                constraints = {"type" : "ineq", "fun" : lambda x: + ( x - OptimizeResult["x"][1] ) },
                                method = "COBYLA")
    
    Result = np.stack([Xmin["x"], OptimizeResult["x"][0], Xmax["x"], Ymin["x"], OptimizeResult["x"][1], Ymax["x"]])
    return Result

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

def argmax2d(ndim_array):
    idx = np.unravel_index(np.argmax(ndim_array), ndim_array.shape)
    return idx, str(idx)

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

if __name__ == '__main__':
    
    name = ["-500", "+500"]
    px_v, px_h = 384, 512
    px_old = 150 # 切り出すpx幅
    mgn = 10 # magnification subpixelまで細かくする時の、データ数の倍率
    px_lim = int(30*mgn)

    act_list = ["06", "07", "08", "09", "10", "11", "13", "14", "15", "16", "17", "19", "20", "21", "22"]
    #act_list = ["17"]
    
    df_cols = ["act",
               "para_e_min", "para_e", "para_e_max",
               "perp_e_min", "perp_e", "perp_e_max", 
               "para_c_min", "para_c", "para_c_max", 
               "perp_c_min", "perp_c", "perp_c_max" ]
    
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
        diff_limb = displace(res_limb["x"][0], res_limb["x"][1], param_limb)
        
        param_center = [ip_center, mgn, px_lim]
        res_center = sp.optimize.minimize(fun=std_func, x0=(0,0), args=(param_center,), method="Powell")
        diff_center = displace(res_center["x"][0], res_center["x"][1], param_center)
        ## error_bar ------------------------------------------------------------
        
        eb_c_px = error_bar_bounds(param_center, res_center, 2)
        eb_e_px = error_bar_bounds(param_limb, res_limb, 2)
        eb_c_urad = subpx2urad(eb_c_px)
        eb_e_urad = subpx2urad(eb_e_px)
        angle_center = urad2title(eb_c_urad[1], eb_c_urad[4])
        angle_limb = urad2title(eb_e_urad[1], eb_e_urad[4])
    
        ## for plot --------------------------------------------------------------
        fig = plt.figure(figsize=(10,15))
        gs = fig.add_gridspec(4, 2)
        
        ax_diff = image_plot(fig, path[16:26], gs[0, 0:2], data_diff, data_diff, 0, 100, "")
        ax_limb_0 = image_plot(fig, name[0]+argmax2d(data_limb[0])[1], gs[1, 1], data_limb[0], data_limb[0], 0, 100, "")
        ax_limb_1 = image_plot(fig, name[1]+argmax2d(data_limb[1])[1], gs[2, 1], data_limb[1], data_limb[0], 0, 100, "")
        ax_center_0 = image_plot(fig, name[0]+argmax2d(data_center[0])[1], gs[1, 0], data_center[0], data_limb[0], 0, 100, "")
        ax_center_1 = image_plot(fig, name[1]+argmax2d(data_center[1])[1], gs[2, 0], data_center[1], data_limb[0], 0, 100, "")
        ax_res_limb = image_plot(fig, angle_limb, gs[3, 1], diff_limb, diff_limb, 0, 100, "")
        ax_res_center = image_plot(fig, angle_center, gs[3, 0], diff_center, diff_center, 0, 100, "")
        
        
        fig.tight_layout()
        
        picname = mkfolder("/"+folder_path[9:15]) + folder_path[16:26] + "_" + name[0] + "_" + name[1] + ".png"
        fig.savefig(picname)
    
        
        record = pd.Series(np.concatenate([np.atleast_1d(int(act_num)), eb_e_urad, eb_c_urad]), index = df_res.columns)
        
        df_res = df_res.append(record, ignore_index=True)
    df_res.to_csv(mkfolder("/"+folder_path[9:15])+folder_path[16:20]+".csv")