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
    print(Pattern1)
    
    Pattern2 = np.empty(36)
    for Idx in range(36):
        Num = Idx + 1
        if Num == 3 or Num == 4 or Num == 15 or Num == 16 or Num == 27 or Num == 28:
            Pattern2[Idx] = 1
        else:
            Pattern2[Idx] = 0
    print(Pattern2)
    
    Pattern3 = np.empty(36)
    for Idx in range(36):
        Num = Idx + 1
        if Num == 3 or Num == 4 or Num == 15 or Num == 16 or Num == 27 or Num == 28:
            Pattern3[Idx] = 1
        elif Num == 5 or Num == 11 or Num == 17 or Num == 23 or Num == 29 or Num == 35:
            Pattern3[Idx] = -1
        else:
            Pattern3[Idx] = 0
    print(Pattern3)
    
    return Pattern1, Pattern2, Pattern3
    
if __name__ == '__main__':
    m1_radi = 1850/2
    stick_angle = 30 # アルミ棒のx軸に対する角度deg
    edge_length = 40 # フチから斜め鏡までの距離
    
    set1, set2, set3 = make_set()
    act_tuning = spa.make_act_tuning()