# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 16:49:52 2021

@author: swimc
"""


import numpy as np
import pandas as pd
import os
import proper as pr
import PIL
import scipy as sp

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import mpl_toolkits.axes_grid1

import ac0423read as ac
import stick_pass_angle as spa

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

def make_set():
    """
    0507ex23
    ExWH23の10のディレクトリは順番に
    1) WH全部0で撮像(x2)
    2) パターン1で符号入れ替えて1回ずつ
    3) パターン2で符号入れ替えて1回ずつ
    4) パターン3で符号入れ替えて1回ずつ
    5) WH全部0で撮像(x2)
    
    パターン1 -> act05, 11 の120°対象 計6つ +500
    パターン2 -> act03, 04 の120°対象 計6つ +500
    パターン3 -> act03, 04 の120°対象 計6つ +500 と
               act05, 11 の120°対象 計6つ -500
    
    Returns
    -------
    Pattern1 : TYPE
        DESCRIPTION.
    Pattern2 : TYPE
        DESCRIPTION.
    Pattern3 : TYPE
        DESCRIPTION.

    """
    
    Pattern1 = np.empty(36)
    for Idx in range(36):
        Num = Idx + 1
        if Num == 5 or Num == 11 or Num == 17 or Num == 23 or Num == 29 or Num == 35:
            Pattern1[Idx] = 1
        else:
            Pattern1[Idx] = 0
    
    Pattern2 = np.empty(36)
    for Idx in range(36):
        Num = Idx + 1
        if Num == 3 or Num == 4 or Num == 15 or Num == 16 or Num == 27 or Num == 28:
            Pattern2[Idx] = 1
        else:
            Pattern2[Idx] = 0
    
    Pattern3 = np.empty(36)
    for Idx in range(36):
        Num = Idx + 1
        if Num == 3 or Num == 4 or Num == 15 or Num == 16 or Num == 27 or Num == 28:
            Pattern3[Idx] = 1
        elif Num == 5 or Num == 11 or Num == 17 or Num == 23 or Num == 29 or Num == 35:
            Pattern3[Idx] = -1
        else:
            Pattern3[Idx] = 0
    
    return Pattern1, Pattern2, Pattern3
    
if __name__ == '__main__':
    px = 1025
    m1_radi = 1850/2
    stick_angle = 30 # アルミ棒のx軸に対する角度deg
    edge_length = 40 # フチから斜め鏡までの距離
    
    x_arr = y_arr = np.linspace(-m1_radi, m1_radi, px)
    xx, yy = np.meshgrid(x_arr, y_arr)
    tf = np.where(xx**2+yy**2<=m1_radi**2, True, False)
    
    set_list = make_set()
    act_tuning = spa.make_act_tuning()
    
    df0 = spa.read("_Fxx/PM3.5_36ptAxWT06_F00.smesh.txt")
    
    df_cols = ["act", "para_e", "perp_e", "para_c", "perp_c"]
    df_res_fem = pd.DataFrame(index=[], columns=df_cols)
    
    for set_num in range(3):
        set_str = str(int(set_num+1))
        diff_sum = np.zeros((px,px))
        
        for act_num in range(36):
            act_str = str(int(act_num + 1)).zfill(2)
            print("F" + act_str)
            
            if set_list[set_num][act_num] == 0:
                pass
            else:
                fem2act = 5 * act_tuning[act_num] * set_list[set_num][act_num]
                
                dfxx = spa.read("_Fxx/PM3.5_36ptAxWT06_F" + act_str + ".smesh.txt")
                diff = tf * fem2act * spa.fem_interpolate(df0, dfxx, xx, yy)
                
                diff_sum = diff_sum + diff
        
        diff_rotate = spa.rotation(diff_sum, stick_angle, tf)
        para_line = diff_rotate[round(px/2), :]
        perp_line_c = spa.perp_line(y_arr, diff_rotate, m1_radi)
        perp_line_e = spa.perp_line(y_arr, diff_rotate, edge_length)
        
        para_c = spa.tangent_line(x_arr, para_line, m1_radi)
        para_e = spa.tangent_line(x_arr, para_line, edge_length)
        perp_c = spa.tangent_line(y_arr, perp_line_c, m1_radi)
        perp_e = spa.tangent_line(y_arr, perp_line_e, m1_radi)
        
        ## for plot ------------------------------------------------------------
        fig = plt.figure(figsize=(10,15))
        fig.suptitle("Pattern" + set_str)
        gs = fig.add_gridspec(3,2)
        
        title_rotate = str(stick_angle) + "[deg]"
        
        ax_diff = ac.image_plot(fig, "", gs[0,0], diff_sum, diff_sum, 0, 100, "mm")
        ax_rotate = ac.image_plot(fig, "", gs[1,0], diff_rotate, diff_rotate, 0, 100, "mm")
        ax_rotate.hlines(round(px/2), 0, px-1, linewidth=5, colors = "white")
        ax_rotate.vlines([para_c[2], para_e[2]], 0, px-1, linewidth=3, colors="black")
        ax_para = spa.stick_plot(fig, "", gs[2, 0], x_arr, para_line, edge_length, m1_radi)
        
        ax_perp_c = spa.stick_plot(fig, "", gs[1, 1], y_arr, perp_line_c, m1_radi, m1_radi)
        ax_perp_e = spa.stick_plot(fig, "", gs[2, 1], y_arr, perp_line_e, m1_radi, m1_radi)
        fig.tight_layout()
        
        picname = mkfolder() + "pattern" + set_str + ".png"
        fig.savefig(picname)
        fig.show()
        
        record = pd.Series([set_str, para_e[1], perp_e[1], para_c[1], perp_c[1]], index=df_res_fem.columns)        
        df_res_fem = df_res_fem.append(record, ignore_index=True)
    
    df_res_fem.to_csv(mkfolder()+"fem_multiact_angle.csv")