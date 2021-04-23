# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 14:48:33 2021

@author: swimc
"""

import numpy as np
import pandas as pd
import os
import pykrige as krg
import proper as pr
import PIL
import fits_correlation as corr

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

def kriging(df_0, dfxx):
    # input [mm] -> output [mm]    
    #mesh型のデータを格子点gridに補完
    df_0 = df_0
    dfxx = dfxx
    
    x = dfxx["x"]
    y = dfxx["y"]
    dw = dfxx["dz"] - df_0["dz"]
    
    ok_module = krg.ok.OrdinaryKriging(x, y, dw)
    z, sigmasq = ok_module.execute("grid", x_arr, y_arr) #格子点にdwをfitting
    return z

def rotation(array_2d, angle_deg, mask_tf):
    img = PIL.Image.fromarray(array_2d)
    img_rotate = img.rotate(angle_deg)
    return img_rotate * mask_tf

def tangent_line(x, y_surf, edge):
    y = y_surf
    radi = x.max()
    
    idx = abs( x - (radi-edge)).argmin()
    
    tilt = ( y[idx+1] - y[idx-1] ) / ( x[idx+1] - x[idx-1] )
    tilt_deg = np.rad2deg(np.arctan(tilt)) # 角度deg
    y_line = tilt * (x - x[idx]) + y[idx]
    return y_line, tilt_deg, idx

def calc_zwopx(tilt_deg):
    f = 500e3 # 望遠鏡の焦点距離 [um]
    zwopx = 2.4 # zwo183 : 2.4um per 1px
    
    theta = 2 * np.deg2rad(tilt_deg) # 反射するので鏡の傾きの2倍
    physical_length = f * np.tan(theta) # 検出器位置での物理的距離
    px_num = physical_length / zwopx
    return px_num

def stick_plot(fig, title, position, x, y_surf, y_center, y_edge, edge):
    px = len(x)
    radi = x.max()
    
    ## center
    idx_c = round(px/2)
    
    ## edge
    idx_e = abs( x - (radi-edge)).argmin()
    
    ## plot
    ax = fig.add_subplot(position)
    ax.plot(x, y_surf, linewidth=5)
    ax.plot(x, y_center)
    ax.plot(x, y_edge)
    ax.scatter([x[idx_c], x[idx_e]], [y_surf[idx_c], y_surf[idx_e]], s=200, c="red")
    
    ax.set_ylim(y_surf.min(), y_surf.max())
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
    px = 257
    m1_radi = 1850/2
    stick_angle = 22.5 # アルミ棒のx軸に対する角度deg
    edge_length = 40 # フチから斜め鏡までの距離
    
    x_arr = y_arr = np.linspace(-m1_radi, m1_radi, px)
    xx, yy = np.meshgrid(x_arr, y_arr)
    
    df0 = read("_Fxx/PM3.5_36ptAxWT03_F00.smesh.txt") 
    tf = np.where(xx**2+yy**2<=m1_radi**2, True, False)
    
    ## 現実でのF09 -> モデルでのF08
    ## 現実でのF11 -> モデルでのF07
    act_dict = {"F09":"F08", "F11":"F07"} # {act_num:fem_num}
    act_num = "F09"
    fname = "_Fxx/PM3.5_36ptAxWT03_" + act_dict[act_num] + ".smesh.txt"
    
    dfxx = read(fname)
    diff = tf * kriging(df0, dfxx)
    
    diff_rotate = rotation(diff, stick_angle, tf) 
    stick_line = diff_rotate[128, :]
    
    y_c, tilt_c, idx_c = tangent_line(x_arr, stick_line, m1_radi)
    y_e, tilt_e, idx_e = tangent_line(x_arr, stick_line, edge_length)
    
    ## calc tilt_angle to zwo183 px ------------------------------------------
    zwopx_c = calc_zwopx(tilt_c)
    zwopx_e = calc_zwopx(tilt_e)
    
    
    ## for plot ------------------------------------------------------------
    text = ["tilt_center = " + str(tilt_c.round(9)) + " [deg]",
            "zwopx_center = " + str(zwopx_c.round(5)) + "[px]",
            "tilt_edge = " + str(tilt_e.round(9)) + " [deg]",
            "zwopx_edge = " + str(zwopx_e.round(5)) + " [px]",
            ]
    title_diff = "act " + act_num + " ( "+ act_dict[act_num] + " in FEM model)"
    fig = plt.figure(figsize=(10,10))
    gs = fig.add_gridspec(2,2)
    
    ax_diff = corr.image_plot(fig, title_diff, gs[0,0], diff, diff, 0, 100, "mm")
    ax_rotate = corr.image_plot(fig, str(stick_angle)+" deg", gs[0,1], diff_rotate, diff, 0, 100, "mm")
    ax_rotate.hlines(round(px/2), 0, px-1, linewidth=5, colors = "white")
    #ax_table = table_plot(fig, "", gs[1,0], table_list, column_list, row_list)
    ax_text = text_plot(fig, "", gs[1,0], text)
    ax_stick = stick_plot(fig, "", gs[1, 1], x_arr, stick_line, y_c, y_e, edge_length)
    fig.tight_layout()
    
    picname = mkfolder() + "act_" + act_num + ".png"
    fig.savefig(picname)