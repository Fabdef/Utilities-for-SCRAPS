#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 19 17:57:58 2018

@author: ubadmin
"""

import os
import tkinter as tk
from tkinter import filedialog
from shutil import copyfile
import numpy as np
import sys
sys.path.append('/home/ubadmin/Documents/Caltech/local_python_library/')
#sys.path.append('C:/Users/DELL E7270/Documents/Documents/Travail/Caltech/local_python_library/')
import scraps as scr
import matplotlib.pyplot as plt

def create_folder_tree(dev_name):
    dev_dir = dev_name + '/'
    if not os.path.exists(dev_dir):
        print(dev_dir + ' directory has been created') 
        os.makedirs(dev_dir)

    folders1 = ['Data/', 'Figures/']
    folders2 = ['Full_freq_sweep/', 'Nb_reso/', 'TiN_reso/']
    folders3 = [['Raw_data/'], ['Pwr_sweep/', 'Temp_sweep/'], ['Fit_MB_params/', 'Fit_reso_params/', 'Raw_data/', 'Split_data/', 'Fit_reso_objects/', 'Select_Fit_reso_objects'], ['Plots/', 'Reports/'], ['Plots/', 'Reports/'], ['Plots/', 'Reports/']]
    folders4 = [['MB_Fit_fres/', 'MB_Fit_Qi/', 'Qi_Qc_fres/'], ['Fit_MB/', 'Fit_reso/']]
    folders5 = [['Raw_data/', 'Fit_reso_params/', 'Fit_reso_objects/', 'Fit_tand_f0_params/'], ['Pwr_sweep/', 'Temp_sweep/']]
   
    for i in range(len(folders1)):
        dir_name = dev_dir + folders1[i]
        if not os.path.exists(dir_name):
            print(dir_name + ' directory has been created') 
            os.makedirs(dir_name)
            
        for k in range(len(folders2)):
            dir2_name = dir_name + folders2[k]
            if not os.path.exists(dir2_name):
                print(dir2_name + ' directory has been created') 
                os.makedirs(dir2_name)

            for m in range(len(folders3[k+i*len(folders2)])):
                dir3_name = dir2_name + folders3[k+i*len(folders2)][m]
                if not os.path.exists(dir3_name):
                    print(dir3_name + ' directory has been created') 
                    os.makedirs(dir3_name)

    for i in range(len(folders4)):
        for k in range(len(folders4[i])):
            dir4_name = dev_dir + folders1[1] + folders2[2] + folders3[5][i] + folders4[i][k]
            if not os.path.exists(dir4_name):
                print(dir4_name + ' directory has been created') 
                os.makedirs(dir4_name)

    for i in range(len(folders5)):
        tmp = [1, 4]
        for k in range(len(folders5[i])):
            dir5_name1 = dev_dir + folders1[i] + folders2[1] + folders3[tmp[i]][0] + folders5[i][k]
            dir5_name2 = dev_dir + folders1[i] + folders2[1] + folders3[tmp[i]][1] + folders5[i][k]
            if not os.path.exists(dir5_name1):
                print(dir5_name1 + ' directory has been created') 
                os.makedirs(dir5_name1)
            if not os.path.exists(dir5_name2):
                print(dir5_name2 + ' directory has been created') 
                os.makedirs(dir5_name2)
   
    
                
def format_datafile_name():   
    root = tk.Tk()
    root.withdraw()
    input_dir = filedialog.askdirectory(title = "Select the input directory (ex.txt_data/)") # Open a dialog window to select the file
    
    list_files = os.listdir(input_dir)
    N = len(list_files) 
    data_dir = os.path.dirname(input_dir) + '/txt_data_formated_name/'
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
               
    if not os.listdir(data_dir):
        for i in range(N):
            filename = input_dir + "/" + list_files[i]
            filename_short = filename.split('/')[-1]
    #        fileName = fileName.split('/')[-1]
            pwr_str = filename_short.split('_')[2][:-2] # power in dBm
            temp_str = filename_short.split('_')[1] # Temp in mK
            str_res = filename_short.split('_')[4].split('-')[-1][:-4] # Name
            devname = filename_short.split('_')[4].split('-')[0]
    #            date = filename_short.split('_')[3]
            if float(temp_str)>10:
                raw_datafile = data_dir + 'RES-' + devname + '-' + str_res + '_' + pwr_str + '_DBM_TEMP_' + '%.3f' %(int(temp_str)/1000) + '.txt'
                copyfile(filename, raw_datafile)
            else:
                raw_datafile = data_dir + 'RES-' + devname + '-' + str_res + '_' + pwr_str + '_DBM_TEMP_' + '%.3f' %(float(temp_str)) + '.txt'
                copyfile(filename, raw_datafile)
                
                
                
def sort_fit_data_files(raw_data_dir):
    list_files = os.listdir(raw_data_dir)
    N= len(list_files)
    pwr, temp, str_res = [], [], []
    for i in range(N):
        filename_short = list_files[i]
        pwr.append(int(filename_short.split('_')[1])) # power in dBm
        temp.append(float(filename_short.split('_')[4][:-4])) # Temp in K
        str_res.append(filename_short.split('_')[0].split('-')[2]) # Name
        devname = filename_short.split('_')[0].split('-')[1] # Name
    
    res_uniq = np.unique(str_res)
    str_res = np.array(str_res)
    pwr = np.array(pwr)
    temp = np.array(temp)

    res_list = []
    for res_name in res_uniq:
        i_res = np.where(str_res == res_name)[0]
        i_pwr = np.argsort(pwr[i_res])
        uniqPwr = np.unique(pwr[i_res[i_pwr]])
        res_pwrsweep = []
        for p in uniqPwr:
            i_temp = np.where(pwr[i_res[i_pwr]] == p)[0]
            i_temp_sorted = np.argsort(temp[i_res[i_pwr[i_temp]]])
            i3 = i_res[i_pwr[i_temp[i_temp_sorted]]]
            res_tempsweep = []
            for t in i3:
                print('Fitting file: ' + list_files[t] + '...')
                res_filename = raw_data_dir + "/" + list_files[t]
                fileDataDict = scr.process_file(res_filename)
                res = scr.makeResFromData(fileDataDict)
                res.name = res_name
                res.load_params(scr.cmplxIQ_params)
                res.do_lmfit(scr.cmplxIQ_fit)
                res.devname = devname
                res_tempsweep.append(res)
            res_pwrsweep.append(res_tempsweep)
        res_list.append(res_pwrsweep)
    return res_list



def plot_freq_sweeps():
    root = tk.Tk()
    root.withdraw()
    width_screen = root.winfo_screenwidth()
    height_screen = root.winfo_screenheight()
    mydpi = 100
    root = tk.Tk()
    root.withdraw()
    input_dir = filedialog.askdirectory(title = "Select the input directory with files to plot inside") # Open a dialog window to select the file
    listfiles = os.listdir(input_dir)
#    listfiles = filedialog.askopenfile(title = 'File to plot').name
    
    fig10, ax10 = plt.subplots(figsize=(width_screen/mydpi, height_screen/mydpi), dpi=mydpi)
    fig10.canvas.set_window_title('Frequency sweep') 
    plt.subplots_adjust(bottom=0.11, top=0.94, right=0.97, left=0.11)
    
    if np.size(listfiles) > 1:
        for f in listfiles:
            data = np.loadtxt(input_dir + '/' + f)
            f = data[:,0]* 1e-9
            z = 10*np.log10(data[:,1]**2 + data[:,2]**2)
            plt.plot(f, z, 'C0', linewidth = 2)
    else:
        data = np.loadtxt(listfiles)
        f = data[:,0]* 1e-9
        z = 10*np.log10(data[:,1]**2 + data[:,2]**2)
        plt.plot(f, z, 'C0', linewidth = 2)
        
    plt.xlabel('Frequency [GHz]', fontsize=30)
    plt.ylabel(r'$|S21|^2$ [dB]', fontsize=30)
    plt.xticks(color='k', size=28)
    plt.yticks(color='k', size=28)
    plt.grid()
    plt.title('Be180306bl: VNA Pwr = -55 dBm, T = 234 mK',fontsize = 30)