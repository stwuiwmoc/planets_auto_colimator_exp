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
import ac0423read as ac
import scipy as sp

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import mpl_toolkits.axes_grid1

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

def read(Filename):
    #skip行数の設定
    Skip = 0
    with open(Filename) as F:
        while True:
            Line = F.readline()
            if Line[0] == '%':
                Skip += 1
            else:
                break
            #エラーの処理
            if len(Line) == 0:
                break

    #データの読み出し
    Df = pd.read_csv(Filename, sep='\s+', skiprows=Skip, header=None) #\s+...スペース数に関わらず区切る
    Df.columns = ["x", "y", "z", "color", "dx", "dy", "dz" ]
    Df = Df * 10**3 # [m] -> [mm]
    return Df

def make_act_tuning():
    """
    fem は0.5Nmでの計算結果, モーター100step(1mm)への変換係数がact_tuning
    
    WH番号　[0,1,6,7,12,13,18,19,24,25,30,31]+1 にはx37
    WH番号 [2,3,8,9,14,15,20,21,26,27,32,33]+1 にはx20
    WH番号 [4,10,16,22,28,34]+1 にはx-38
    WH番号　[5,11,17,23,29,35]+1 にはx38

    Returns
    -------
    None.

    """
    Act_tuning = np.array([ 37.,  37.,  20.,  20., -38.,  38.,  37.,  37.,  20.,  20., -38.,
        38.,  37.,  37.,  20.,  20., -38.,  38.,  37.,  37.,  20.,  20.,
       -38.,  38.,  37.,  37.,  20.,  20., -38.,  38.,  37.,  37.,  20.,
        20., -38.,  38.])
    return Act_tuning

def fem_interpolate(Df_0, Dfxx, XX_new, YY_new):
    # input [mm] -> output [mm]    
    #mesh型のデータを格子点gridに補完
    X_old = Dfxx["x"]
    Y_old = Dfxx["y"]
    Dw_old = Dfxx["dz"] - Df_0["dz"]
    
    XY_old = np.stack([X_old, Y_old], axis=1)
    Dw_new = sp.interpolate.griddata(XY_old, Dw_old, (XX_new, YY_new), method="linear", fill_value=0)
    return Dw_new
    
def rotation(array_2d, angle_deg, mask_tf):
    img = PIL.Image.fromarray(array_2d)
    img_rotate = img.rotate(angle_deg)
    return img_rotate * mask_tf

def perp_line(X, Array_2d, Edge):
    Radi = X.max()
    Idx = abs( X - (Radi-Edge)).argmin()
    
    Perp_line = Array_2d[:, Idx]
    return Perp_line

def tangent_line(X, Z_surf, Edge):
    """
    

    Parameters
    ----------
    X : 1d-array
        
    Z_surf : 1d-array
        stick_line
    Edge : float
        length from M1 edge

    Returns
    -------
    Y_line : 1d-array
        tangent line
    Angle : float[micro rad]
        
    Idx : int
        DESCRIPTION.

    """
    Y = Z_surf
    Radi = X.max()
    Idx = abs( X - (Radi-Edge)).argmin()
    
    Tilt = ( Y[Idx+15] - Y[Idx-15] ) / ( X[Idx+15] - X[Idx-15] )
    Angle = np.arctan(Tilt) * 1e6 # 角度 microrad
    
    Y_line = Tilt * (X - X[Idx]) + Y[Idx]
    return Y_line, Angle, Idx

def calc_zwopx(tilt_rad):
    f = 500e3 # 望遠鏡の焦点距離 [um]
    zwopx = 2.4 # zwo183 : 2.4um per 1px
    
    theta = 2 * tilt_rad # 反射するので鏡の傾きの2倍
    physical_length = f * np.tan(theta) # 検出器位置での物理的距離
    px_num = physical_length / zwopx
    return px_num

def stick_plot(Fig, Title, Position, X, Z_surf, Edge, M1_radi):
    Px = len(Z_surf)
    Horizontal = np.linspace(-M1_radi, M1_radi, Px) # 地面に水平な面上での距離
    Z_c, Tilt_c_mrad, Idx_c = tangent_line(Horizontal, Z_surf, M1_radi)
    Z_e, Tilt_e_mrad, Idx_e = tangent_line(Horizontal, Z_surf, Edge)
    
    Tilt_c_str = str(round(Tilt_c_mrad, 2))
    Tilt_e_str = str(round(Tilt_e_mrad, 2))
    
    Z_surf_drop = np.where(Z_surf==0, np.nan, Z_surf)
    
    ## plot
    Ax = Fig.add_subplot(Position)
    Ax.plot(X, Z_surf_drop, linewidth=5, label="")
    Ax.plot(X, Z_c, label=Tilt_c_str + " [micro rad]")
    Ax.plot(X, Z_e, label=Tilt_e_str + " [micro rad]")
    Ax.scatter([X[Idx_c], X[Idx_e]], [Z_surf[Idx_c], Z_surf[Idx_e]], s=200, c="red", label="")
    
    Ax.set_title(Title)
    #Ax.set_ylim(np.nanmin(Z_surf_drop), np.nanmax(Z_surf_drop))
    Ax.set_ylim(Z_surf.min(), Z_surf.max())
    Ax.set_ylabel("Surfaace figure z [mm]")
    Ax.legend()
    return Ax

if __name__ == '__main__':
    px = 1025
    m1_radi = 1850/2
    stick_angle = 30 # アルミ棒のx軸に対する角度deg
    edge_length = 40 # フチから斜め鏡までの距離
    # fem では0.05Nm のトルク、実物の+-500は板バネ+-5mm相当、ばね定数5.78N/mm、腕の長さ0.25m
    
    act_tuning = make_act_tuning()
    
    fname_res = "mkfolder/ac0430read/210430/act01_36.csv"
    df_res = pd.read_csv(fname_res)
    
    x_arr = y_arr = np.linspace(-m1_radi, m1_radi, px)
    xx, yy = np.meshgrid(x_arr, y_arr)
    
    df0 = read("_Fxx/PM3.5_36ptAxWT06_F00.smesh.txt") 
    tf = np.where(xx**2+yy**2<=m1_radi**2, True, False)
    
    df_cols = ["act", "para_e", "perp_e", "para_c", "perp_c"]
    df_res_fem = pd.DataFrame(index=[], columns=df_cols)
    
    for i in range(0, len(df_res)):
    #for i in range(0, 3):
        act_num = "F" + str(int(df_res["act"][i])).zfill(2)
        print(act_num)
        
        #fem は0.5Nm, モーター100step(1mm)への変換係数がact_tuning
        fem2act = 5 * act_tuning[int(df_res["act"][i]-1)]
    
        fname = "_Fxx/PM3.5_36ptAxWT06_" + act_num + ".smesh.txt"
        dfxx = read(fname)
        diff =  tf * fem_interpolate(df0, dfxx, xx, yy)
        
        diff_rotate = rotation(diff, stick_angle, tf) * fem2act
        para_line = diff_rotate[round(px/2), :]
        perp_line_c = perp_line(y_arr, diff_rotate, m1_radi)
        perp_line_e = perp_line(y_arr, diff_rotate, edge_length)
        
        para_c = tangent_line(x_arr, para_line, m1_radi)
        para_e = tangent_line(x_arr, para_line, edge_length)
        perp_c = tangent_line(y_arr, perp_line_c, m1_radi)
        perp_e = tangent_line(y_arr, perp_line_e, m1_radi)
        
        
        ## for plot ------------------------------------------------------------
        title_diff = "act" + act_num[1:] + " in FEM"
        title_rotate = "act" + act_num[1:] + " ( FEM x " + str(fem2act) + " ), " + str(stick_angle)+" deg"
        
        fig = plt.figure(figsize=(10,15))
        gs = fig.add_gridspec(3,2)
        
        ax_diff = ac.image_plot(fig, title_diff, gs[0,0], diff, diff, 0, 100, "Surfaace figure z [mm]")
        ax_rotate = ac.image_plot(fig, title_rotate, gs[1,0], diff_rotate, diff_rotate, 0, 100, "Surfaace figure z [mm]")
        ax_rotate.hlines(round(px/2), 0, px-1, linewidth=5, colors = "white")
        ax_rotate.vlines([para_c[2], para_e[2]], 0, px-1, linewidth=3, colors="black")
        ax_para = stick_plot(fig, "Parallel to Optical path", gs[2, 0], x_arr, para_line, edge_length, m1_radi)
        
        ax_perp_c = stick_plot(fig, "Perpendicular in Optical path in Center", gs[1, 1], y_arr, perp_line_c, m1_radi, m1_radi)
        ax_perp_e = stick_plot(fig, "Perpendicular in Optical path in Edge", gs[2, 1], y_arr, perp_line_e, m1_radi, m1_radi)
        fig.tight_layout()
        
        picname = mkfolder() + "act_" + act_num + ".png"
        fig.savefig(picname)
        
        record = pd.Series([int(act_num[1:]), para_e[1], perp_e[1], para_c[1], perp_c[1]], index=df_res_fem.columns)        
        df_res_fem = df_res_fem.append(record, ignore_index=True)
    
    df_res_fem.to_csv(mkfolder()+"fem_angle.csv")