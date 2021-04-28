# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 14:48:33 2021

@author: swimc
"""

import numpy as np
import pandas as pd
import os
import proper as pr
import PIL
import fits_correlation as corr
import scipy as sp

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import mpl_toolkits.axes_grid1

def mkfolder(suffix = ""):
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

def read(filename):
    #skip行数の設定
    skip = 0
    with open(filename) as f:
        while True:
            line = f.readline()
            if line[0] == '%':
                skip += 1
            else:
                break
            #エラーの処理
            if len(line) == 0:
                break

    #データの読み出し
    df = pd.read_csv(filename, sep='\s+', skiprows=skip, header=None) #\s+...スペース数に関わらず区切る
    df.columns = ["x", "y", "z", "color", "dx", "dy", "dz" ]
    df = df * 10**3 # [m] -> [mm]
    return df

def fem_interpolate(df_0, dfxx):
    # input [mm] -> output [mm]    
    #mesh型のデータを格子点gridに補完
    x_old = dfxx["x"]
    y_old = dfxx["y"]
    dw_old = dfxx["dz"] - df_0["dz"]
    
    xy_old = np.stack([x_old, y_old], axis=1)
    dw_new = sp.interpolate.griddata(xy_old, dw_old, (xx, yy), method="linear", fill_value=0)
    return dw_new
    
def rotation(array_2d, angle_deg, mask_tf):
    img = PIL.Image.fromarray(array_2d)
    img_rotate = img.rotate(angle_deg)
    return img_rotate * mask_tf

def tangent_line(x, y_surf, edge):
    y = y_surf
    radi = x.max()
    
    idx = abs( x - (radi-edge)).argmin()
    
    tilt = ( y[idx+15] - y[idx-15] ) / ( x[idx+15] - x[idx-15] )
    tilt_rad = np.arctan(tilt) # 角度rad
    y_line = tilt * (x - x[idx]) + y[idx]
    return y_line, tilt_rad, idx

def calc_zwopx(tilt_rad):
    f = 500e3 # 望遠鏡の焦点距離 [um]
    zwopx = 2.4 # zwo183 : 2.4um per 1px
    
    theta = 2 * tilt_rad # 反射するので鏡の傾きの2倍
    physical_length = f * np.tan(theta) # 検出器位置での物理的距離
    px_num = physical_length / zwopx
    return px_num

def stick_plot(fig, title, position, x, y_surf, edge):
    px = len(x)
    radi = x.max()
    
    y_ct, tilt_ct, idx_ct = tangent_line(x_arr, y_surf, m1_radi)
    y_eg, tilt_eg, idx_eg = tangent_line(x_arr, y_surf, edge)
    tilt_ct_mrad = str(round(tilt_ct * 1e6, 3))
    tilt_eg_mrad = str(round(tilt_eg * 1e6, 3))

    ## plot
    ax = fig.add_subplot(position)
    ax.plot(x[1:-1], y_surf[1:-1], linewidth=5, label="")
    ax.plot(x, y_ct, label=tilt_ct_mrad + " [micro rad]")
    ax.plot(x, y_eg, label=tilt_eg_mrad + " [micro rad]")
    ax.scatter([x[idx_c], x[idx_e]], [y_surf[idx_c], y_surf[idx_e]], s=200, c="red", label="")
    
    ax.set_ylim(y_surf.min(), y_surf.max())
    ax.set_ylabel
    ax.legend()
    return ax

def table_plot(fig, title, position, data_list, col, row):
    data_list = data_list.round(2).values
    ax = fig.add_subplot(position)
    ax.table(cellText = data_list,
             cellLoc = "right",
             colLabels = col,
             colLoc = "center",
             rowLabels = row,
             rowLoc = "center")
    ax.axis("off")
    return ax

def text_plot(fig, title, position, text_list):
    row = len(text_list)
    fs = 15
    ax = fig.add_subplot(position)
    height = np.arange(start=0.1, stop=0.2+0.1*row, step=0.1)
    
    for i in range(row):
        ax.text(0.1, height[i], text_list[i], ha="left", fontsize=fs)
    
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.set_ylim(1,0)
    return ax

if __name__ == '__main__':
    px = 1025
    m1_radi = 1850/2
    stick_angle = 22.5 # アルミ棒のx軸に対する角度deg
    edge_length = 40 # フチから斜め鏡までの距離
    # fem では0.05Nm のトルク、実物の+-500は板バネ+-5mm相当、ばね定数5.78N/mm、腕の長さ0.25m
    
    #WH番号　[0,1,6,7,12,13,18,19,24,25,30,31]+1 にはx37
    #WH番号 [2,3,8,9,14,15,20,21,26,27,32,33]+1 にはx20
    #WH番号 [4,10,16,22,28,34]+1 にはx-38
    #WH番号　[5,11,17,23,29,35]+1 にはx38
    act_tuning = np.array([ 37.,  37.,  20.,  20., -38.,  38.,  37.,  37.,  20.,  20., -38.,
        38.,  37.,  37.,  20.,  20., -38.,  38.,  37.,  37.,  20.,  20.,
       -38.,  38.,  37.,  37.,  20.,  20., -38.,  38.,  37.,  37.,  20.,
        20., -38.,  38.])
    
    fname_res = "mkfolder/fits_correlation/210423/ex10.csv"
    df_res = pd.read_csv(fname_res, index_col=0)
    
    x_arr = y_arr = np.linspace(-m1_radi, m1_radi, px)
    xx, yy = np.meshgrid(x_arr, y_arr)
    
    df0 = read("_Fxx/PM3.5_36ptAxWT06_F00.smesh.txt") 
    tf = np.where(xx**2+yy**2<=m1_radi**2, True, False)
    
    for i in range(0, len(df_res)):
    #for i in range(0, 1):
        act_num = "F" + str(df_res["act"][i]).zfill(2)
        print(act_num)
        
        #fem は0.5Nm, モーター100step(1mm)への変換係数がact_tuning, 0423の計測は+-500stepなので更に10倍
        fem2act = 10 * act_tuning[df_res["act"][i]-1]
    
        fname = "_Fxx/PM3.5_36ptAxWT06_" + act_num + ".smesh.txt"
        dfxx = read(fname)
        diff =  tf * fem_interpolate(df0, dfxx)
        
        diff_rotate = rotation(diff, stick_angle, tf) * fem2act
        stick_line = diff_rotate[round(px/2), :]
        
        y_c, tilt_c, idx_c = tangent_line(x_arr, stick_line, m1_radi)
        y_e, tilt_e, idx_e = tangent_line(x_arr, stick_line, edge_length)
        
        ## calc tilt_angle to zwo183 px ------------------------------------------
        zwopx_c = calc_zwopx(tilt_c) 
        zwopx_e = calc_zwopx(tilt_e)
        
        
        ## for plot ------------------------------------------------------------
        text = ["zwopx_center = " + str(zwopx_c.round(3)) + "[px]",
                fname_res[-15:-4] + " = " + str(round(df_res["dvert_c"][i], 3)) + "[px]",
                "",
                "zwopx_edge = " + str(zwopx_e.round(3)) + " [px]",
                fname_res[-15:-4] + " = " + str(round(df_res["dvert_l"][i], 3)) + "[px]",
                "",
                "zwopx_diff = " + str((zwopx_e-zwopx_c).round(3)) + " [px]",
                fname_res[-15:-4] + " = " + str(round(df_res["dvert_l"][i]-df_res["dvert_c"][i], 3)) + "[px]",
                ]
        
        #title_diff = act_dict[act_num] + " in FEM model"
        title_diff = "act" + act_num[1:] + " in FEM"
        title_rotate = "act" + act_num[1:] + " ( FEM x " + str(fem2act) + " ), " + str(stick_angle)+" deg"
        
        fig = plt.figure(figsize=(10,10))
        gs = fig.add_gridspec(2,2)
        
        ax_diff = corr.image_plot(fig, title_diff, gs[0,0], diff, diff, 0, 100, "mm")
        ax_rotate = corr.image_plot(fig, title_rotate, gs[0,1], diff_rotate, diff_rotate, 0, 100, "mm")
        ax_rotate.hlines(round(px/2), 0, px-1, linewidth=5, colors = "white")
        #ax_table = table_plot(fig, "", gs[1,0], table_list, column_list, row_list)
        ax_text = text_plot(fig, "", gs[1,0], text)
        ax_stick = stick_plot(fig, "", gs[1, 1], x_arr, stick_line, edge_length)
        fig.tight_layout()
        
        picname = mkfolder() + "act_" + act_num + ".png"
        fig.savefig(picname)