# -*- coding: utf-8 -*-
"""
Created on Sat May  1 15:35:51 2021

@author: swimc
"""

import numpy as np
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
def act_plot(Fig, Title, Position, Black=False, Blue=False, Red=False, Orange=False, Green=False):
    
    Color_list = ["black", "blue", "red", "orangered", "green"]
    Data_list = [Black, Blue, Red, Orange, Green]

    Ax = Fig.add_subplot(Position)
    
    for i in range(5):
        Data = Data_list[i]
        Color = Color_list[i]
        
        if not Data == False:
            Df, Key = Data
            
            if len(Df.columns) == 7: # つまり df_fem を使っている場合
                Ax.plot(Df["act"], Df[Key], c=Color)
            
            else: # つまり df_ac を使っている場合
                Key_min = Key + "_ebmin"
                Key_max = Key + "_ebmax"
                Ax.errorbar(Df["act"], Df[Key], c=Color, 
                            yerr=[Df[Key_min].values, Df[Key_max].values])
    
    Ax.set_title(Title)
    Ax.set_ylabel("Tilt [micro rad]")
    Ax.set_xlabel("Act01 ~ Act36")
    Ax.grid(axis="y")
    return Ax
        

if __name__ == '__main__':
    #fname_ac = "mkfolder/fits_correlation/210423/ex10.csv"
    #fname_ac = "mkfolder/ac0430read/210430/act01_36.csv"
    fname_ac = "mkfolder/ac0507_exwh22read/210507/act01_36.csv"
    fname_fem = "mkfolder/stick_pass_angle/fem_angle.csv"
    df_ac = pd.read_csv(fname_ac)
    df_fem = pd.read_csv(fname_fem, index_col=0)
    
    # -500, +500 の場合
    df_fem.iloc[:, df_fem.columns!="act"] = df_fem.iloc[:, df_fem.columns!="act"] * 2
    
    df_ac["para_d"] = df_ac["para_e"] - df_ac["para_c"]
    df_ac["perp_d"] = df_ac["perp_e"] - df_ac["perp_c"]
    
    df_fem["para_d"] = df_fem["para_e"] - df_fem["para_c"]
    df_fem["perp_d"] = df_fem["perp_e"] - df_fem["perp_c"]
  
    for i in ["para", "perp"]:
        df_ac[i+"_d_ebmin"] = np.sqrt( df_ac[i+"_e_ebmin"]**2 + df_ac[i+"_c_ebmin"]**2 )
        df_ac[i+"_d_ebmax"] = np.sqrt( df_ac[i+"_e_ebmax"]**2 + df_ac[i+"_c_ebmax"]**2 )
    
    
    fig2 = plt.figure(figsize=(10,10))
    gs2 = fig2.add_gridspec(3,2)
    fig2.suptitle("Left : Perpendicular to Optical Path  |  Right : Parallel to Optical Path\n")
    
    ax_acpp2 = act_plot(fig2, "Blue : AC(center), Red : AC(edge)\nGreen : dAC (Edge - Center)", gs2[0,0]
                       , False, [df_ac, "perp_c"], [df_ac, "perp_e"], False, [df_ac, "perp_d"])
    ax_acpr2 = act_plot(fig2, "Blue : AC(center), Red : AC(edge)\nGreen : dAC (Edge - Center)", gs2[0,1]
                       , False, [df_ac, "para_c"], [df_ac, "para_e"], False, [df_ac, "para_d"])
    
    ax_fempp2 = act_plot(fig2, "Blue : FEM(center), Red : FEM(edge)\nOrange : dFEM (Edge - Center)", gs2[1,0]
                        , False, [df_fem, "perp_c"], [df_fem, "perp_e"], [df_fem, "perp_d"])
    ax_fempr2 = act_plot(fig2, "Blue : FEM(center), Red : FEM(edge)\nOrange : dFEM (Edge - Center)", gs2[1,1]
                        , False, [df_fem, "para_c"], [df_fem, "para_e"], [df_fem, "para_d"])
    
    ax_ppd2 = act_plot(fig2, "Green : dAC, Orange : dFEM", gs2[2,0]
                      , False, False, False, [df_fem, "perp_d"], [df_ac, "perp_d"])
    ax_prd2 = act_plot(fig2, "Green : dAC, Orange : dFEM", gs2[2,1]
                      , False, False, False, [df_fem, "para_d"], [df_ac, "para_d"])
    
    fig2.tight_layout()
    picname2 = mkfolder() + fname_ac[-19:-13] + "ver2.png"
    fig2.savefig(picname2)
    print("finish")