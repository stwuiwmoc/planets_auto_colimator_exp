# -*- coding: utf-8 -*-
"""
Created on Tue May  4 13:55:36 2021

@author: swimc

0507 ExWH22のac実験用
"""

import ac0423read as ac

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

if __name__ == '__main__':
    start = time.time()
    px_v, px_h = 384, 512
    px_clip_width = 250 
    px_lim = 50
    mgn = 10 # magnification subpixelまで細かくする時の、データ数の倍率
    subpx_lim = int(px_lim * mgn)

    df_cols = ["act",
               "para_e", "para_e_eb",
               "perp_e", "perp_e_eb", 
               "para_c", "para_c_eb", 
               "perp_c", "perp_c_eb",
               "e_std", "c_std", "noise"]
    
    df_res = pd.DataFrame(index=[], columns=df_cols)
    
    folder_path = "raw_data/210507/ExWH22/**"
    folder_list_raw = glob.glob(folder_path, recursive=False)
    
    
    """
    ExWH19に76のディレクトリがありますが、順番に
    1) WH全部0で画像取得x2回　(x2)
    2) WH01を+500で画像取得 (x1)
    3) WH01を+000で画像取得 (x1)
    4) 2-3をWH02-WH36まで繰り返す (2x35=70)
    5) WH全部0で画像取得x2回　(x2)
    のデータになっています。
    """
    folder_list = folder_list_raw[2:-2]
    
    for i in range(36):
        act_num = str(i+1).zfill(2)
        print(act_num)
        
        data_mean = []
        data_e = [] # edge
        data_c = [] # center
        ip_e = []
        ip_c = []
        data_noise = []
        data_noise_std = []
        
        
        for j in range(2): # j=0 : +500 / j=1 : -500
            
            path_list = glob.glob(folder_list[int(i*2+j)]+"/*.FIT")
            data_mean_temp = np.zeros((px_v, px_h))
            
            for path in path_list:
                data = ac.fits_2darray(path)
                data_mean_temp = data + data_mean_temp
            
            ## 10回の撮像を平均し、edgeとcenterを切り出し --------------------------
            data_mean_temp = data_mean_temp / len(path_list)
            data_e_temp = data_mean_temp[100:300, 250:450]
            data_c_temp = data_mean_temp[50:300, 50:300]
            
            data_mean.append(data_mean_temp)
            data_e.append(data_e_temp)
            data_c.append(data_c_temp)
            
            ## read out noise -----------------------------------------------
            data_noise_temp = np.zeros((px_v, px_h))
            
            for path in path_list:
                data = ac.fits_2darray(path)
                data_noise_temp = data_noise_temp + (data -data_mean_temp)**2
            
            data_noise_temp = np.sqrt( data_noise_temp / len(path_list) )
            data_noise.append(data_noise_temp)
            data_noise_std.append(np.std(data_noise_temp))
            
            ## interpolate ---------------------------------------------------
            ip_e.append(ac.fits_interpolate(data_e_temp, mgn))
            ip_c.append(ac.fits_interpolate(data_c_temp, mgn))
        
        data_diff = data_mean[1] - data_mean[0]
        data_noise_std.append(np.sqrt(data_noise_std[0]**2+data_noise_std[1]**2))
        
        ## minimize ----------------------------------------------------------
        param_e = ip_e + [subpx_lim]
        param_c = ip_c + [subpx_lim]
        
        res_e = sp.optimize.minimize(fun=ac.std_func, x0=(0,0), args=(param_e, ), method="Powell")
        res_c = sp.optimize.minimize(fun=ac.std_func, x0=(0,0), args=(param_c, ), method="Powell")
        
        diff_e = ac.displace(res_e["x"][0], res_e["x"][1], param_e)
        diff_c = ac.displace(res_c["x"][0], res_c["x"][1], param_c)
       
        print(time.time()-start)
        ## error_bar ---------------------------------------------------------
        err_size, err_mgn = 20, 0.05
        x_err_e, y_err_e, eb_e_px = ac.error_xy_loop(res_e, param_e, err_size, err_mgn, data_noise_std[2])
        x_err_c, y_err_c, eb_c_px = ac.error_xy_loop(res_c, param_c, err_size, err_mgn, data_noise_std[2])
        
        eb_e_urad = ac.subpx2urad(eb_e_px)
        eb_c_urad = ac.subpx2urad(eb_c_px)

        angle_e = ac.urad2title(eb_e_urad[0], eb_e_urad[2])
        angle_c = ac.urad2title(eb_c_urad[0], eb_c_urad[2])

        record = np.concatenate([np.atleast_1d(int(act_num)), eb_e_urad, eb_c_urad, np.array([res_e["fun"], res_c["fun"], data_noise_std[2]])])
        
        df_res = df_res.append(pd.Series(record, index = df_res.columns), 
                               ignore_index=True)

        print(time.time()-start)
        
        ## plot --------------------------------------------------------------
        fig = plt.figure(figsize=(10,15))
        gs = fig.add_gridspec(5,2)
        fig.suptitle(folder_path[9:15] + " act" + act_num)
        
        ax_5 = ac.image_plot(fig, "+500", gs[0, 0], data_mean[0], data_mean[0])
        ax_0 = ac.image_plot(fig, "-500", gs[0, 1], data_mean[1], data_mean[0])
        ax_diff = ac.image_plot(fig, "diff {+500} - {-500}", gs[1,0:2], data_diff, data_diff)
        ax_res_e = ac.image_plot(fig, angle_e, gs[2,1], diff_e, data_diff)
        ax_res_c = ac.image_plot(fig, angle_c, gs[2,0], diff_c, data_diff)
        
        ax_err_xe = ac.err_plot(fig, "xe", gs[3, 1], res_e["x"][0], x_err_e, res_e["fun"], data_noise_std[2], err_mgn)
        ax_err_ye = ac.err_plot(fig, "ye", gs[4, 1], res_e["x"][1], y_err_e, res_e["fun"], data_noise_std[2], err_mgn)
        ax_err_xc = ac.err_plot(fig, "xc", gs[3, 0], res_e["x"][0], x_err_c, res_c["fun"], data_noise_std[2], err_mgn)
        ax_err_yc = ac.err_plot(fig, "yc", gs[4, 0], res_e["x"][1], y_err_c, res_c["fun"], data_noise_std[2], err_mgn)
        
        fig.tight_layout()
        
        picname = mkfolder("/"+folder_path[9:15]) + "act" + act_num + ".png"
        fig.savefig(picname)
        fig.clf()
    
    csvname = mkfolder("/"+folder_path[9:15]) + "act01_36.csv"
    if i == 1:
        pass
    else:    
        df_res.to_csv(csvname, index=False)
    