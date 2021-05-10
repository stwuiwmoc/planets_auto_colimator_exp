# -*- coding: utf-8 -*-
"""
Created on Tue May  4 13:55:36 2021

@author: swimc

0430のac実験用
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

if __name__ == '__main__':
    start = time.time()
    px_v, px_h = 384, 512
    mgn = 10 # magnification subpixelまで細かくする時の、データ数の倍率
    px_lim = int(30*mgn)

    df_cols = ["act", "para_e", "perp_e", "para_c", "perp_c"]
    df_res = pd.DataFrame(index=[], columns=df_cols)
    
    folder_path = "raw_data/210430/**"
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
    
    for i in range(1):
        act_num = str(i+1).zfill(2)
        print(act_num)
        
        data_mean = []
        data_e = [] # edge
        data_c = [] # center
        ip_e = []
        ip_c = []
        
        for j in range(2): # j=0 : +500 / j=1 : +000
            
            path_list = glob.glob(folder_list[int(i*2+j)]+"/*.FIT")
            data_mean_temp = np.zeros((px_v, px_h))
            
            for path in path_list:
                data = ac.fits_2darray(path)
                data_mean_temp = data + data_mean_temp
            
            ## 10回の撮像を平均し、edgeとcenterを切り出し --------------------------
            data_mean_temp = data_mean_temp / len(path_list)
            data_e_temp = data_mean_temp[50:300, 250:500]
            data_c_temp = data_mean_temp[25:275, 50:300]
            
            data_mean.append(data_mean_temp)
            data_e.append(data_e_temp)
            data_c.append(data_c_temp)
            
            ## interpolate ---------------------------------------------------
            ip_e.append(ac.fits_interpolate(data_e_temp, mgn))
            ip_c.append(ac.fits_interpolate(data_c_temp, mgn))
        
        data_diff = data_mean[1] - data_mean[0]
        
        ## minimize ----------------------------------------------------------
        param_e = [ip_e, mgn, px_lim]
        res_e = sp.optimize.minimize(fun=ac.std_func, x0=(0,0), args=(param_e, ), method="Powell")
        diff_e = ac.displace(res_e["x"], param_e)
        angle_e = ac.px2urad(res_e)
        
        param_c = [ip_c, mgn, px_lim]
        res_c = sp.optimize.minimize(fun=ac.std_func, x0=(0,0), args=(param_c, ), method="Powell")
        diff_c = ac.displace(res_c["x"], param_c)
        angle_c = ac.px2urad(res_c)
        
        record = pd.Series([act_num, angle_e[0], angle_e[1], angle_c[0], angle_c[1]], index=df_res.columns)        
        df_res = df_res.append(record, ignore_index=True)
    
        
        ## plot --------------------------------------------------------------
        fig = plt.figure(figsize=(10,15))
        gs = fig.add_gridspec(4,2)
        fig.suptitle(folder_path[9:15] + " act" + act_num)
        
        ax_5 = ac.image_plot(fig, "+500", gs[0, 0:2], data_mean[0], data_mean[0])
        ax_0 = ac.image_plot(fig, "+000", gs[1, 0:2], data_mean[1], data_mean[0])
        ax_diff = ac.image_plot(fig, "diff {+500} - {+000}", gs[2,0:2], data_diff, data_diff)
        ax_res_e = ac.image_plot(fig, angle_e[2], gs[3,0], diff_e, data_diff)
        ax_res_c = ac.image_plot(fig, angle_c[2], gs[3,1], diff_c, data_diff)
        
        fig.tight_layout()
        
        picname = mkfolder("/"+folder_path[9:15]) + "act" + act_num + ".png"
        fig.savefig(picname)
    
    csvname = mkfolder("/"+folder_path[9:15]) + "act01_36.csv"
    #df_res.to_csv(csvname, index=False)
    print(time.time()-start)