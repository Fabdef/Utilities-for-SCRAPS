#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 15:01:04 2018

@author: ubadmin
"""

import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from matplotlib.widgets import Slider, Button
import os
import sys
#sys.path.append('/home/ubadmin/Documents/Caltech/local_python_library/scraps-master/')
sys.path.append('C:/Users/DELL E7270/Documents/Documents/Travail/Caltech/local_python_library/')
import scraps as scr
import glob
from scipy.interpolate import interp1d
from pyswarm import pso
import scipy.special as sp
from matplotlib.backends.backend_pdf import PdfPages
import pickle


# Constants
kb = 8.6173423e-5 # eV/K
N0 = 3.9e10 # eV ^-1 um^-3 for TiN (A.E. Lowitz, E.M. Barrentine, S.R. Golwala, P.T. Timbie, LTD 2013)
h = 4.13567e-15 # =2 pi hbar in eV s
hbar= h/(2 * np.pi)
win_size_ini = 0.07 # Initial size of the zoomed bottom plot 
picker = 12 # define the radius of the circle around the point where it can be selected
atten = 30 # attenuation inside the cryostat (dB)
# Get the sizr of the sreen to plot full-screen plots
root = tk.Tk()
root.withdraw()
width_screen = root.winfo_screenwidth()
height_screen = root.winfo_screenheight()
mydpi = 100

root = tk.Tk()
root.withdraw()
device_dir = filedialog.askdirectory(title = "Select the device directory (ex. Be170227bl)") # Open a dialog window to select the device directory
device_dir = device_dir + '/'

# Try to find the folder Data/TiN_reso/Fit_reso_objects*/ because it is where I would usually save the fits of the resonances with a different program (split_and_fit_KIDs_reso)
# If it does not exist, the programs asks you where to look at
if glob.glob(device_dir + 'Data/TiN_reso/Fit_reso_objects*/'):
    obj_load_dir = glob.glob(device_dir + 'Data/TiN_reso/Fit_reso_objects*/')[0]
else:
    root = tk.Tk()
    root.withdraw()
    obj_load_dir = filedialog.askdirectory(title = "Select the Fit params objects directory (ex. Fit_reso_object)") # Open a dialog window to select the file

# Try to find the folder Data/TiN_reso/Select_Fit_reso_objects*/ because it is where I would usually save the good fits of the resonances with a different program (split_and_fit_KIDs_reso)
# (The fits can be visually checked by the user and only the good ones are selected)
# If it does not exist, the programs asks you where to look at
if glob.glob(device_dir + 'Data/TiN_reso/Select_Fit_reso_objects*/'):
    obj_save_dir = glob.glob(device_dir + 'Data/TiN_reso/Select_Fit_reso_objects*/')[0]
else:
    root = tk.Tk()
    root.withdraw()
    obj_save_dir = filedialog.askdirectory(title = "Select the Fit params objects directory (ex. Fit_reso_object)") # Open a dialog window to select the file

# Creation of various directories
plot_dir = device_dir + 'Figures/TiN_reso/Plots/'
if not os.path.exists(plot_dir):
    print(plot_dir + ' directory has been created') 
    os.makedirs(plot_dir)
    
MB_fit_dir = device_dir + 'Data/TiN_reso/Fit_MB_params/'
if not os.path.exists(MB_fit_dir):
    print(MB_fit_dir + ' directory has been created') 
    os.makedirs(MB_fit_dir)

# 
pickle_files = np.sort(os.listdir(obj_load_dir)) # Need to sort the directory where the pickle files are to have them ordered by temperature
N_files = len(pickle_files)
fileName = pickle_files[0]
str_pwr = fileName.split('_')[1] # power in dBm
str_temp = fileName.split('_')[4][:-2] # Temp in K
str_name = device_dir.split('/')[-2][0:-1] # Device Name

temp, pwr = np.zeros(N_files), np.zeros(N_files)
fres_data, T_data, qi_data, qc_data, fit_ok_ind_data, res_data = [], [], [], [], [], []

# Open the pickle files, read the different parameters of the resonance fits (from SCRAPS)
for i in range(N_files):
    res_list = pickle.load(open(obj_load_dir + pickle_files[i],'rb'))
    res_data.append(res_list)
    fres = [a.lmfit_result.params['f0'].value for a in res_list]
    qi = [a.lmfit_result.params['qi'].value for a in res_list]
    qc = [a.lmfit_result.params['qc'].value for a in res_list]
    fit_ok_ind = np.where([a.fit_ok for a in res_list])[0] # new parameter added in split_and_fit_KIDs_reso to tell if the resonance fit is good or not
    fres_data.append(np.array(fres))
    qi_data.append(np.array(qi))
    qc_data.append(np.array(qc))
    fit_ok_ind_data.append(np.array(fit_ok_ind))
    N = len(fres_data[i])
    temp[i] = pickle_files[i].split('_')[4][:-2]
    pwr[i] = pickle_files[i].split('_')[1]
    T_data.append(np.ones(N)*temp[i])
    # new parameter to say if the Mattis Bardeen fit is good. For now this param is set to fit_ok (only the resonances corresctly fitted are used for the MB fit)
    # Later the user will be able to choose what resonance is good to be fitted for Mattis Bardeen and MB_fit_ok will become more restrictive
    if not hasattr(res_list[0], 'MB_fit_ok'): 
        for k in range(len(res_list)): res_list[k].MB_fit_ok = res_list[k].fit_ok
    
N_data = [len(a) for a in fit_ok_ind_data]

T_data_arr = np.zeros([max(N_data),N_files])*np.NaN
fres_data_arr = np.zeros([max(N_data),N_files])*np.NaN
qi_data_arr = np.zeros([max(N_data),N_files])*np.NaN
qc_data_arr = np.zeros([max(N_data),N_files])*np.NaN

# Select the data with fit_ok = 1 before plotting them
for i in range(N_files):
    T_data_arr[0:N_data[i],i] = T_data[i][fit_ok_ind_data[i]]
    fres_data_arr[0:N_data[i],i] = fres_data[i][fit_ok_ind_data[i]]
    qi_data_arr[0:N_data[i],i] = qi_data[i][fit_ok_ind_data[i]]
    qc_data_arr[0:N_data[i],i] = qc_data[i][fit_ok_ind_data[i]]

# Choose the right unit for the data
f_min = np.nanmin(fres_data_arr)*0.98 
f_max = np.nanmax(fres_data_arr)*1.02 
pow10 = np.floor(np.log10((f_max-f_min)/2))
plot_unit = 10**(np.floor(pow10/3)*3)
unit_str = ['Hz', 'kHz', 'MHz', 'GHz', 'THz']
plot_unit_str = unit_str[int(np.log10(plot_unit)/3)]
f_start = (np.floor(f_min/10**(pow10))*10**(pow10))/plot_unit
f_stop = (np.ceil(f_max/10**(pow10))*10**(pow10))/plot_unit

# Plot the figure with 2 plots with the Temp VS resonance frequency of all theresonances (1 main one and one zoomed one where the user can select the "Good" resonances)
fig = plt.figure()
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.subplots_adjust(bottom=0.2, top=0.96, right=0.96, left=0.08)

ax1 = fig.add_subplot(211, facecolor='#FFFFCC')
ax1.plot(fres_data_arr/plot_unit,T_data_arr*1e3, 'C0o')
plt.ylabel('Temperature [mK]', fontsize=14)
plt.xticks(color='k', size=14)
plt.yticks(color='k', size=14)
ax1.set_xlim(f_start, f_stop)

ax2 = fig.add_subplot(212, facecolor='#FFFFCC')
ax2.plot(fres_data_arr/plot_unit,T_data_arr*1e3, 'C0o', picker=picker)
plt.ylabel('Temperature [mK]', fontsize=14)
plt.xlabel('Frequency [' + plot_unit_str + ']', fontsize=14)
plt.xticks(color='k', size=14)
plt.yticks(color='k', size=14)
ax2.set_xlim(f_start, f_stop)

ax_color = 'lightgoldenrodyellow' # Color of the sliders
ax_move_win = plt.axes([0.12, 0.09, 0.50, 0.03], facecolor=ax_color)
s_move_win = Slider(ax_move_win, 'Window position', 0, 1, valinit = 0)
s_move_win.label.set_fontsize(14)
s_move_win.valtext.set_fontsize(14)

ax_win_width = plt.axes([0.12, 0.03, 0.50, 0.03], facecolor=ax_color)
s_win_width = Slider(ax_win_width, 'Window size', 0, 1, valinit = win_size_ini)
s_win_width.label.set_fontsize(14)
s_win_width.valtext.set_fontsize(14)

ax_export = plt.axes([0.73, 0.09, 0.1, 0.04])
b_export = Button(ax_export, 'Export params', color=ax_color, hovercolor='0.975')
b_export.label.set_fontsize(14)

ax_plot_Qi_Qc = plt.axes([0.73, 0.03, 0.1, 0.04])
b_plot_Qi_Qc = Button(ax_plot_Qi_Qc, 'Plot Qi & Qc', color=ax_color, hovercolor='0.975')
b_plot_Qi_Qc.label.set_fontsize(14)

ax_fit_Qi_f0 = plt.axes([0.855, 0.03, 0.1, 0.04])
b_fit_Qi_f0 = Button(ax_fit_Qi_f0, 'Fit Qi & f0 (MB)', color=ax_color, hovercolor='0.975')
b_fit_Qi_f0.label.set_fontsize(14)

ax_pdf = plt.axes([0.855, 0.09, 0.1, 0.04])
b_pdf = Button(ax_pdf, 'Generate pdf', color=ax_color, hovercolor='0.975')
b_pdf.label.set_fontsize(14)


class analyze_data(object):
    
    def __init__(self):
        self.win_width = win_size_ini*(f_stop - f_start)
        self.pos_Norm = 0
        self.win_width_Norm = self.win_width/(f_stop - f_start)
        self.win = ax1.axvspan(f_start, f_start + self.win_width, facecolor='r', alpha=0.5)
        self.indx = np.array([], dtype=int)
        self.indy = np.array([], dtype=int)
        self.selected, = ax2.plot([], [], 'o', ms=15, alpha=0.5, color='red', visible=False)
        self.device_dir = device_dir
        self.fres_data_arr_cleaned = fres_data_arr
        self.T_data_arr_cleaned = T_data_arr
        
    # Define the frequency limits of the bottom plot (this plot is a zoom of the main top plot, and it is possible to move the zoom area)
    def move_win(self, val):
        win_width = self.win_width
        win = self.win
        pos_Norm = s_move_win.val
        pos = f_start + win_width/2+ (f_stop-f_start-win_width)*pos_Norm
        xmin = pos - win_width/2
        xmax = pos + win_width/2
        ax2.set_xlim(xmin, xmax)
        self.update_win(win, xmin, xmax)
        fig.canvas.draw_idle()
        self.pos_Norm = pos_Norm
        self.win = win
        
    # Change the width of the zoomed area
    def change_win_width(self, val):
        pos_Norm = self.pos_Norm
        win = self.win
        win_width_Norm = s_win_width.val
        win_width = (f_stop-f_start)*win_width_Norm
        pos = f_start + win_width/2+ (f_stop-f_start-win_width)*pos_Norm
        xmin = pos - win_width/2
        xmax = pos + win_width/2
        ax2.set_xlim(xmin, xmax)
        self.update_win(win, xmin, xmax)    
        fig.canvas.draw_idle()
        self.win_width = win_width
        self.win_width_Norm = win_width_Norm
        
    # Use the right and left arrows to move the zoomed area in the bottom plot (more convenient)
    def press(self, event):
        pos_Norm = self.pos_Norm
        win_width_Norm = self.win_width_Norm
        sys.stdout.flush()
        if event.key == 'right':
            pos_Norm = pos_Norm + win_width_Norm/5
            s_move_win.set_val(pos_Norm)
            fig.canvas.draw()
        elif event.key == 'left':
            pos_Norm = pos_Norm - win_width_Norm/5
            s_move_win.set_val(pos_Norm)
            fig.canvas.draw()
            
    # Update the red rectangle location on top plot corresponding to the zoomed area of the bottom plot
    def update_win(self, polygon, xmin, xmax):
        _ndarray = polygon.get_xy()
        _ndarray[:, 0] = [xmin, xmin, xmax, xmax, xmin]
        polygon.set_xy(_ndarray)
        
    # Select the data when clicking on the data points. (and unselect when clicking again)
    # The list of selected data will be used for the Mattis Bardeen fits
    def select_data(self, event):
        N = len(event.ind)
        if not N:
            return True
        # the click locations
        x = event.mouseevent.xdata
        y = event.mouseevent.ydata
        indy = int(np.nanargmin(abs(y-T_data_arr[0]*1e3)))
        indx = int(np.nanargmin(abs(x-fres_data_arr[:,indy]/plot_unit)))
        if (indx in self.indx) & (indy in self.indy):
            indx_remove = np.where(self.indx == indx)[0]
            indy_remove = np.where(self.indy == indy)[0]
            ind_remove = [a for a in indx_remove if a in [b for b in indy_remove]]
            if ind_remove:
                print(indx, indy)
                print(res_data[indy][fit_ok_ind_data[indy][indx]].lmfit_result.params['f0'].value)
                res_data[indy][fit_ok_ind_data[indy][indx]].MB_fit_ok = 1
                self.indx = np.delete(self.indx, ind_remove)
                self.indy = np.delete(self.indy, ind_remove)
                self.update_selection()
                return True
        
        res_data[indy][fit_ok_ind_data[indy][indx]].MB_fit_ok = 0 
        print(indx, indy)
        print(self.indx, self.indy)
        print(res_data[indy][fit_ok_ind_data[indy][indx]].lmfit_result.params['f0'].value)
        self.indx = np.append(self.indx, indx)
        self.indy = np.append(self.indy, indy)
        self.update_selection()

    # Update the selected list of data points
    def update_selection(self):
        indx = self.indx
        indy = self.indy
        fres_data_arr_cleaned = self.fres_data_arr_cleaned
        T_data_arr_cleaned = self.T_data_arr_cleaned
        self.selected.set_visible(True)
        self.selected.set_data(fres_data_arr_cleaned[indx, indy]/plot_unit, T_data_arr_cleaned[indx, indy]*1e3)
        fig.canvas.draw()
        self.fres_data_arr_cleaned = fres_data_arr_cleaned
        self.T_data_arr_cleaned = T_data_arr_cleaned
        
        
# end of cleaned part of the code. The rest is in construction        
        
        
    def export_cleaned_params(self, var):
        fres_tmp = np.array(fres_data_arr)
        fres_tmp[self.indx, self.indy] = np.NaN
        for i in range(len(res_data)):
            print()
            res_list_save_file = obj_save_dir + 'pickle_' + str_pwr + '_DBM_TEMP_' + '%.3f' %(temp[i]) + '.p'
            print(res_list_save_file)
            pickle.dump(res_data[i], open(res_list_save_file,'wb'))
            print(len(res_data[i]))
        for i in range(np.size(fres_tmp,1)):
            ind = np.where(~np.isnan(fres_tmp[:,i]))[0]
            ind_cleaned.append(ind)
        if len(np.unique([len(a) for a in ind_cleaned])) > 1:
            print('Bad cleaning')
            print([len(a) for a in ind_cleaned])
#            print([fres_tmp[a] for a in ind_cleaned])
#            plt.figure()
#            for t in range(8):
#            t=6
#            N = [len(a) for a in ind_cleaned][t]
#            ind_cleaned_arr = np.array(ind_cleaned).T
#            print(ind_cleaned_arr[t], t*np.ones(N,dtype=int))
#            plt.plot(T_data_arr[ind_cleaned_arr[t], t*np.ones(N,dtype=int)], fres_data_arr[ind_cleaned_arr[t], t*np.ones(N,dtype=int)], 'o')
                
        else:
            N = [len(a) for a in ind_cleaned][0]
            ind_cleaned_arr = np.array(ind_cleaned).T
            ind_temp = np.meshgrid(np.arange(0, N_files), np.ones(N))[0]
            fres_cleaned_arr = fres_data_arr[ind_cleaned_arr, ind_temp]
            T_cleaned_arr = T_data_arr[ind_cleaned_arr, ind_temp]
            qi_cleaned_arr = qi_data_arr[ind_cleaned_arr, ind_temp]
            qc_cleaned_arr = qc_data_arr[ind_cleaned_arr, ind_temp]
            
#            if not os.path.exists(cleaned_reso_dir):
#                print(cleaned_reso_dir + ' directory has been created') 
#                os.makedirs(cleaned_reso_dir)
        
#            for i in range(N_files):
#                Output_params_file = cleaned_reso_dir + 'cleaned-fit-params_' + str_pwr + '_DBM_TEMP_' + '%.3f' %(T_cleaned_arr[0,i]) + '.txt'
#                np.savetxt(Output_params_file, np.c_[ind_cleaned_arr[:,i], T_cleaned_arr[:,i]*1e3, fres_cleaned_arr[:,i], qi_cleaned_arr[:,i], qc_cleaned_arr[:,i]], delimiter='\t', fmt='%.0f', header='Ind  Temp (mK)  Freq (Hz)   Qi    Qc')


    def read_clean_files(self):
        ind_cleaned, fres_data, T_data, qi_data, qc_data, res_data, MB_ok_ind_data = [], [], [], [], [], [], []
#        cleaned_reso_dir = self.cleaned_reso_dir
#        
#        if os.path.isdir(cleaned_reso_dir):
#            cleaned_param_files = os.listdir(cleaned_reso_dir)
#            
#        else:
#            root = tk.Tk()
#            root.withdraw()
#            cleaned_reso_dir = filedialog.askopenfilename() # Open a dialog window to select the file
#            cleaned_param_files = os.listdir(cleaned_reso_dir)
        
        pickle_files = np.sort(os.listdir(obj_load_dir))

        for filename in pickle_files:
#            data = np.loadtxt(cleaned_reso_dir + cleaned_param_files[i], delimiter ='\t')
            res_list = pickle.load(open(obj_save_dir + filename,'rb'))
            res_data.append(res_list)
            fres = np.array([a.lmfit_result.params['f0'].value for a in res_list])
            qi = np.array([a.lmfit_result.params['qi'].value for a in res_list])
            qc = np.array([a.lmfit_result.params['qc'].value for a in res_list])
            MB_ok_ind = np.where([a.MB_fit_ok for a in res_list])[0]
            fit_ok_ind = np.where([a.fit_ok for a in res_list])[0]
            MB_ok_ind_data.append(np.array(MB_ok_ind))
            temp = filename.split('_')[4][:-2]
            fres_data.append(fres[MB_ok_ind])
            qi_data.append(qi[MB_ok_ind])
            qc_data.append(qc[MB_ok_ind])
            N = len(fres_data[0])
            T_data.append(np.ones(N)*float(temp))
    
#            ind_cleaned.append(data[:,0])
#            fres_data.append(data[:,2])
#            T_data.append(data[:,1])
#            qi_data.append(data[:,3])
#            qc_data.append(data[:,4])
        
#        argsorted = np.argsort([a[0] for a in T_data])
#        fres_data =  [fres_data[i] for i in argsorted]
#        T_data = [T_data[i] for i in argsorted]
#        qi_data =  [qi_data[i] for i in argsorted]
#        qc_data =  [qc_data[i] for i in argsorted]
#        ind_cleaned =  [ind_cleaned[i] for i in argsorted]
        self.fres_data = fres_data
        self.qi_data = qi_data
        self.qc_data = qc_data
        self.T_data = T_data
        self.MB_ok_ind_data = MB_ok_ind_data
        
    
    def plot_Qi_Qc(self, var):
        self.read_clean_files()
        qi_data = self.qi_data
        qc_data = self.qc_data
        T_data = self.T_data
        
        fig1, ax1 = plt.subplots(figsize=(width_screen/mydpi, height_screen/mydpi), dpi=mydpi)
        fig1.canvas.set_window_title('Qi VS Temp') 
        plt.subplots_adjust(bottom=0.1, top=0.96, right=0.97, left=0.1)
        med = [np.median(a) for a in qi_data]
        for i in range(len(T_data[0])):
            T = [a[i] for a in T_data]
            Qi = [a[i] for a in qi_data]
            plt.plot(np.array(T)*1e3, Qi ,'o-')
            
        min_Qi = max([min([min(a) for a in qi_data])*0.85, min(med)/3])
        max_Qi = min([max([max(a) for a in qi_data])*1.15, max(med)*3])
        plt.yscale('log')
        plt.xlabel('Temperature [mK]', fontsize=26)
        plt.ylabel('Qi', fontsize=26)
        plt.xticks(color='k', size=24)
        plt.yticks(color='k', size=24)
        plt.ylim(min_Qi, max_Qi)
        plt.tick_params(axis='y', which='minor', labelsize=20)
        plt.grid(True, which="both") 
        plt.show()

        fig2, ax2 = plt.subplots(figsize=(width_screen/mydpi, height_screen/mydpi), dpi=mydpi)
        fig2.canvas.set_window_title('Histogram Qi') 
        plt.subplots_adjust(bottom=0.11, top=0.95, right=0.97, left=0.07)
        med = [np.median(a) for a in qi_data]
        min_Qi = max([min([min(a) for a in qi_data])*0.95, min(med)/3])
        max_Qi = min([max([max(a) for a in qi_data])*1.05, max(med)*3])
#        plt.hist( qi_data)
#        ax.xaxis.set_major_locator(plt.MaxNLocator(15))
#        Nbins = [(a>plt.xlim()[0]) & (a<plt.xlim()[1]) for a in plt.xticks()[0]].count(True)
#        plt.clf()
        plt.hist( qi_data, alpha = 0.7, bins=12)
#        plt.xscale('log')
        plt.xlabel('Qi', fontsize=26)
        plt.ylabel('Number of resonances', fontsize=26)
        plt.xticks(color='k', size=24)
        plt.yticks(color='k', size=24)
        plt.legend([str(int(a[0]*1e3)) + ' mK' for a in T_data], fontsize=20)
        plt.xlim(min_Qi, max_Qi)
        ax2.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
        tx = ax2.xaxis.get_offset_text()
        tx.set_fontsize(22)
#        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        plt.grid()     
        plt.show()

        fig3, ax3 = plt.subplots(figsize=(width_screen/mydpi, height_screen/mydpi), dpi=mydpi)
        fig3.canvas.set_window_title('Qc VS Temp') 
        plt.subplots_adjust(bottom=0.1, top=0.96, right=0.97, left=0.1)
        for i in range(len(T_data[0])):
            T = [a[i] for a in T_data]
            Qc = [a[i] for a in qc_data]
            plt.plot(np.array(T)*1e3, Qc ,'o-')
        
        med = [np.median(a) for a in qc_data]
        min_Qc = max([min([min(a) for a in qc_data])*0.85, min(med)/5])
        max_Qc = min([max([max(a) for a in qc_data])*1.15, max(med)*5])
        plt.yscale('log')
        plt.xlabel('Temperature [mK]', fontsize=26)
        plt.ylabel('Qc', fontsize=26)
        plt.xticks(color='k', size=24)
        plt.yticks(color='k', size=24)
        plt.ylim(min_Qc, max_Qc)
        plt.tick_params(axis='y', which='minor', labelsize=20)
        plt.grid(True, which="both") 
        plt.show()
        
        fig4, ax4 = plt.subplots(figsize=(width_screen/mydpi, height_screen/mydpi), dpi=mydpi)
        fig4.canvas.set_window_title('Histogram Qc') 
        plt.subplots_adjust(bottom=0.11, top=0.95, right=0.97, left=0.07)
        med = [np.median(a) for a in qc_data]
        min_Qc = max([min([min(a) for a in qc_data])*0.95, min(med)/5])
        max_Qc = min([max([max(a) for a in qc_data])*1.05, max(med)*5])        
#        plt.hist( qc_data, alpha = 0)
#        ax4.xaxis.set_major_locator(plt.MaxNLocator(12))
#        Nbins = [(a>plt.xlim()[0]) & (a<plt.xlim()[1]) for a in plt.xticks()[0]].count(True)
#        xticks = plt.xticks()[0][np.where([(a>plt.xlim()[0]) & (a<plt.xlim()[1]) for a in plt.xticks()[0]])[0]]
#        ax4.cla()
        plt.hist( qc_data, alpha = 0.7, bins=12)
#        plt.xticks(xticks)
#        ax4.set_xticks(xticks)
##        ax.xaxis.set_major_locator(plt.MaxNLocator(10))
        plt.xlabel('Qc', fontsize=26)
        plt.ylabel('Number of resonances', fontsize=26)
        plt.xticks(color='k', size=24)
        plt.yticks(color='k', size=24)
        plt.grid()
        plt.legend([str(int(a[0]*1e3)) + ' mK' for a in T_data], fontsize=20)
        plt.xlim(min_Qc, max_Qc)
        ax4.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
        tx = ax4.xaxis.get_offset_text()
        tx.set_fontsize(22)
        plt.show()

        Q_plots_dir = plot_dir + 'Qi_Qc_fres/'
        filename = list([Q_plots_dir + str_name + "_Qi_VS_temp"])
        filename.append(Q_plots_dir + str_name + "_Qi_hist")
        filename.append(Q_plots_dir + str_name + "_Qc_VS_temp")
        filename.append(Q_plots_dir + str_name + "_Qc_hist")
        fig_num = [fig1, fig2, fig3, fig4]
        for t in range(len(filename)):
            fig_num[t].savefig(filename[t] + '.svg', dpi=fig.dpi, bbox_inches = 'tight')
            fig_num[t].savefig(filename[t] + '.png', dpi=plt.gcf().dpi, bbox_inches = 'tight')
            fig_num[t].savefig(filename[t] + '.pdf', dpi=fig.dpi, bbox_inches = 'tight')
            plt.close(fig_num[t])


    def fit_Qi_f0(self,var):
        self.read_clean_files()
        fres_data = self.fres_data
        qi_data = self.qi_data
#        qc_data = self.qc_data
        T_data = self.T_data
        MB_ok_ind_data = self.MB_ok_ind_data
        
#        def get_nqp(T, Delta):
#            N = np.size(T)  
#            if N>1:
#                nqp = np.zeros(N)
#                for m in range(N):
##                    Delta = Delta0*((1-np.sqrt(2*np.pi*kb*T[m]/Delta0)*np.exp(-(Delta0-mu)/(kb*T[m]))))
#                    nqp[m] = quad(lambda E: E/((1 + np.exp(E/(kb*T[m]))) * np.real(np.sqrt(E**2-(Delta+0*1j)**2))), Delta, Delta*10, epsabs=1e-16)[0]*4*N0
#            else:
##                delta = Delta0*((1-np.sqrt(2*np.pi*kb*T/Delta0)*np.exp(-(Delta0-mu)/(kb*T))))
#                nqp = quad(lambda E: E/((1 + np.exp(E/(kb*T[m]))) * np.real(np.sqrt(E**2-(Delta+0*1j)**2))), Delta, Delta*10, epsabs=1e-16)[0]*4*N0
#            return nqp
#
#        def MB_calc_f(T, fres, alpha, Delta, f0):
#            nqp = 2 * N0 * np.sqrt(2 * np.pi * kb * T * Delta) * np.exp(-Delta/(kb * T))
##            nqp = get_nqp(T, Delta)
#            zeta = (hbar * np.pi * fres)/(kb * T)
#            k2 = 1 / (2 * N0 * Delta) * (1+ np.sqrt(2 * Delta/(np.pi * kb * T)) * np.exp(-zeta) * sp.iv(0, zeta))
#            df_f0 = -alpha/2 * k2 * nqp
#            return (df_f0) * f0 + f0
#        
#        def MB_calc_Qi(T, fres, Qi, alpha, Delta, Qi0):
#            nqp = 2 * N0 * np.sqrt(2 * np.pi * kb * T * Delta) * np.exp(-Delta/(kb * T))
##            nqp = get_nqp(T, Delta)
#            zeta = (hbar * np.pi * fres)/(kb * T)
#            k1 = 1 / (np.pi * N0 * Delta) * np.sqrt(2 * Delta/(np.pi * kb * T)) * np.sinh(zeta) * sp.kn(0, zeta)
#            d_1VSQi = alpha * k1 * nqp
#            return 1/((d_1VSQi) + 1/Qi0)
#        
#        def minimize_f_for_pso(params, *args):
#            alpha, Delta, f0 = params
#            T, fres = args
#            return np.sum(abs(MB_calc_f(T, fres, alpha, Delta, f0) - fres)**2)
#        
#        def minimize_Qi_for_pso(params, *args):
#            alpha, Delta, Qi0 = params
#            T, fres, Qi = args
#            return np.sum(abs(MB_calc_Qi(T, fres, Qi, alpha, Delta, Qi0) - Qi)**2)
#


        def MB_calc_fbis(T, fres, alpha, Delta, f0):
            nqp = 2 * N0 * np.sqrt(2 * np.pi * kb * T * Delta) * np.exp(-Delta/(kb * T))
#            nqp = get_nqp(T, Delta)
            zeta = (hbar * np.pi * fres)/(kb * T)
            k2 = 1 / (2 * N0 * Delta) * (1+ np.sqrt(2 * Delta/(np.pi * kb * T)) * np.exp(-zeta) * sp.iv(0, zeta))
            df_f0 = -alpha/2 * k2 * nqp
            return (df_f0) * f0 + f0
        
        def MB_calc_Qibis(T, fres, Qi, alpha, Delta, Qi0):
            nqp = 2 * N0 * np.sqrt(2 * np.pi * kb * T * Delta) * np.exp(-Delta/(kb * T))
#            nqp = get_nqp(T, Delta)
            zeta = (hbar * np.pi * fres)/(kb * T)
            k1 = 1 / (np.pi * N0 * Delta) * np.sqrt(2 * Delta/(np.pi * kb * T)) * np.sinh(zeta) * sp.kn(0, zeta)
            d_1VSQi = alpha * k1 * nqp
            return 1/((d_1VSQi) + 1/Qi0)
        
        def minimize_f_for_psobis(params, *args):
            Delta, f0 = params
            alpha, T, fres = args
            return np.sum(abs(MB_calc_fbis(T, fres, alpha, Delta, f0) - fres)**2)
        
        def minimize_Qi_for_psobis(params, *args):
            Delta, Qi0 = params
            alpha, T, fres, Qi = args
            return np.sum(abs(MB_calc_Qibis(T, fres, Qi, alpha, Delta, Qi0) - Qi)**2)



        alphaFres_arr, DeltaFres_arr, errFres_arr, f0_arr, freslist_arr, Tlist_arr = [], [], [], [], [], []
        alphaQi_arr, DeltaQi_arr, errQi_arr, Qi0_arr, Qilist_arr = [], [], [], [], []


        for l in range(len(fres_data[0])):
            freslist = np.array([a[l] for a in fres_data])
            Tlist = np.array([a[l] for a in T_data])
            Qilist = np.array([a[l] for a in qi_data])
            
#            lbFres = np.array([0.05, 1e-7, freslist[0]*1])
#            ubFres = np.array([5, 1e-2, freslist[0]*1.03])
#            lbQi = np.array([0.05, 1e-7, Qilist[0]*0.5])
#            ubQi = np.array([5, 1e-2, Qilist[0]*2])
#            
#            xoptFres, foptFres = pso(minimize_f_for_pso, lbFres, ubFres, ieqcons=[], f_ieqcons=None, args=(Tlist, freslist), kwargs={},
#                swarmsize=2000, omega=0.5, phip=0.5, phig=0.5, maxiter=200, minstep=1e-8,
#                minfunc=1e-8, debug=False)
#            
#            xoptQi, foptQi = pso(minimize_Qi_for_pso, lbQi, ubQi, ieqcons=[], f_ieqcons=None, args=(Tlist, freslist, Qilist), kwargs={},
#                swarmsize=2000, omega=0.5, phip=0.5, phig=0.5, maxiter=200, minstep=1e-8,
#                minfunc=1e-8, debug=False)
#            
#            alphaFres, DeltaFres, f0 = xoptFres
#            alphaQi, DeltaQi, Qi0 = xoptQi
            
            
            lbFres = np.array([1e-7, freslist[0]*1])
            ubFres = np.array([1e-2, freslist[0]*1.03])
            lbQi = np.array([1e-7, Qilist[0]*0.5])
            ubQi = np.array([1e-2, Qilist[0]*2])
            
            xoptFres, foptFres = pso(minimize_f_for_psobis, lbFres, ubFres, ieqcons=[], f_ieqcons=None, args=(0.9, Tlist, freslist), kwargs={},
                swarmsize=2000, omega=0.5, phip=0.5, phig=0.5, maxiter=200, minstep=1e-8,
                minfunc=1e-8, debug=False)
            
            xoptQi, foptQi = pso(minimize_Qi_for_psobis, lbQi, ubQi, ieqcons=[], f_ieqcons=None, args=(0.9, Tlist, freslist, Qilist), kwargs={},
                swarmsize=2000, omega=0.5, phip=0.5, phig=0.5, maxiter=200, minstep=1e-8,
                minfunc=1e-8, debug=False)
            
#            alphaFres, DeltaFres, f0 = xoptFres
#            alphaQi, DeltaQi, Qi0 = xoptQi
            DeltaFres, f0 = xoptFres
            DeltaQi, Qi0 = xoptQi
            alphaFres = 0.9
            alphaQi = 0.9
            
            
            alphaFres_arr.append(alphaFres)
            DeltaFres_arr.append(DeltaFres)
            f0_arr.append(f0)
            errFres_arr.append(foptFres*1e-6)
            alphaQi_arr.append(alphaQi)
            DeltaQi_arr.append(DeltaQi)
            Qi0_arr.append(Qi0)
            errQi_arr.append(foptQi*1e-6)
            freslist_arr.append(freslist)
            Tlist_arr.append(Tlist)
            Qilist_arr.append(Qilist)

#        err_arr = np.array(err_arr)
#        alpha_arr = np.array(alpha_arr)
#        Delta_arr = np.array(Delta_arr)
#        f0_arr = np.array(f0_arr)
        
        filenameFres = MB_fit_dir + str_name + "_MB_fres-fit_params.txt"
        fmt = ['%d','%.3f','%.3e','%.2f'] + ['%d' for a in np.ones(len(freslist_arr[0]))] + ['%.3f' for a in np.ones(len(Tlist_arr[0]))] 
        header = 'f0 [Hz]    Alpha    Delta [eV]     error [AU]           fres [Hz] (for each temp)            Temp [K]'
        np.savetxt(filenameFres, np.c_[f0_arr, alphaFres_arr, DeltaFres_arr, errFres_arr, freslist_arr, Tlist_arr], delimiter='\t', fmt=fmt, header=header)

        filenameQi = MB_fit_dir + str_name + "_MB_Qi-fit_params.txt"
        fmt = ['%d','%.3f','%.3e','%.2f'] + ['%d' for a in np.ones(len(freslist_arr[0]))] + ['%.3f' for a in np.ones(len(Tlist_arr[0]))]  + ['%.3e' for a in np.ones(len(Qilist_arr[0]))] 
        header = 'Qi0         Alpha    Delta [eV]     error [AU]           fres [Hz] (for each temp)            Temp [K]          Qi'
        np.savetxt(filenameQi, np.c_[Qi0_arr, alphaQi_arr, DeltaQi_arr, errQi_arr, freslist_arr, Tlist_arr, Qilist_arr], delimiter='\t', fmt=fmt, header=header)



        MB_fres_fit_params = np.loadtxt(filenameFres)
        MB_Qi_fit_params = np.loadtxt(filenameQi)
        f0_arr = np.array([a[0] for a in MB_fres_fit_params])
        alphaFres_arr = np.array([a[1] for a in MB_fres_fit_params])
        DeltaFres_arr = np.array([a[2] for a in MB_fres_fit_params])
        errFres_arr = np.array([a[3] for a in MB_fres_fit_params])
        Qi0_arr = np.array([a[0] for a in MB_Qi_fit_params])
        alphaQi_arr = np.array([a[1] for a in MB_Qi_fit_params])
        DeltaQi_arr = np.array([a[2] for a in MB_Qi_fit_params])
        errQi_arr = np.array([a[3] for a in MB_Qi_fit_params])
        Nfres = int((len(MB_fres_fit_params[0])-4)/2)
        freslist_arr = [a[4:Nfres+4] for a in MB_fres_fit_params]
        Tlist_arr = [a[Nfres+4:] for a in MB_fres_fit_params]
        
        fig10, ax10 = plt.subplots(figsize=(width_screen/mydpi, height_screen/mydpi), dpi=mydpi)
        fig10.canvas.set_window_title('df/f0 VS Temp') 
        plt.subplots_adjust(bottom=0.09, top=0.96, right=0.97, left=0.09)
        
        for i in range(len(freslist_arr)):
            dfvsf0_fit = (MB_calc_fbis(Tlist_arr[i], freslist_arr[i], alphaFres_arr[i], DeltaFres_arr[i], f0_arr[i])-f0_arr[i])/f0_arr[i]

#            dfvsf0_fit = (MB_calc_f(Tlist_arr[i], freslist_arr[i], alphaFres_arr[i], DeltaFres_arr[i], f0_arr[i])-f0_arr[i])/f0_arr[i]
            dfvsf0 = (freslist_arr[i] - f0_arr[i]) / f0_arr[i]
            Tnew = np.linspace(Tlist_arr[i][0],Tlist_arr[i][-1],100)
            dfvsf0_fit_smooth = interp1d(Tlist_arr[i], dfvsf0_fit, kind='cubic')
            color = plt.cm.tab20(i%20)
            plt.plot(Tnew*1e3,dfvsf0_fit_smooth(Tnew)*1e3,  color=color)
            plt.plot(Tlist_arr[i]*1e3, dfvsf0*1e3, 'o', markersize = 8,  color=color)
            plt.plot(Tlist_arr[i]*1e3, dfvsf0_fit*1e3, 'x', markersize = 8,  color=color)
        #    str_leg.append(r"$\alpha$ = %.3f, $f_0$ = %.4f MHz, $\Delta$ = %.3e, err = %.2f" %(alpha, f0*1e-6, Delta, fopt*1e-6))

        #plt.legend(str_leg, fontsize = 14, loc=3)
        plt.xlabel('Temperature [mK]', fontsize=26)
        plt.ylabel(r'[f(T) - f(0)] / f(0)    $\times 10^3$ ', fontsize=26)
        plt.xticks(color='k', size=24)
        plt.yticks(color='k', size=24)
        plt.grid()


        fig20, ax20 = plt.subplots(figsize=(width_screen/mydpi, height_screen/mydpi), dpi=mydpi)
        fig20.canvas.set_window_title('d(1/Qi) VS Temp') 
        plt.subplots_adjust(bottom=0.09, top=0.96, right=0.97, left=0.09)
        
        for i in range(len(Qilist_arr)):
            d_1vsQi_fit = 1/MB_calc_Qibis(Tlist_arr[i], freslist_arr[i], Qilist_arr[i], alphaQi_arr[i], DeltaQi_arr[i], Qi0_arr[i]) - 1/Qi0_arr[i]

#            d_1vsQi_fit = 1/MB_calc_Qi(Tlist_arr[i], freslist_arr[i], Qilist_arr[i], alphaQi_arr[i], DeltaQi_arr[i], Qi0_arr[i]) - 1/Qi0_arr[i]
            d_1vsQi = 1/Qilist_arr[i] - 1/Qi0_arr[i]
            Tnew = np.linspace(Tlist_arr[i][0],Tlist_arr[i][-1],100)
            d_1vsQi_fit_smooth = interp1d(Tlist_arr[i],d_1vsQi_fit, kind='cubic')
            color = plt.cm.tab20(i%20)
            plt.plot(Tnew*1e3,d_1vsQi_fit_smooth(Tnew)*1e3,  color=color)
            plt.plot(Tlist_arr[i]*1e3, d_1vsQi*1e3, 'o', markersize = 8,  color=color)
            plt.plot(Tlist_arr[i]*1e3, d_1vsQi_fit*1e3, 'x', markersize = 8,  color=color)
        #    str_leg.append(r"$\alpha$ = %.3f, $f_0$ = %.4f MHz, $\Delta$ = %.3e, err = %.2f" %(alpha, f0*1e-6, Delta, fopt*1e-6))
            
        #plt.legend(str_leg, fontsize = 14, loc=3)
        plt.xlabel('Temperature [mK]', fontsize=26)
        plt.ylabel(r'[1/Qi(T) - 1/Qi(0)]   $\times 10^3$ ', fontsize=26)
        plt.xticks(color='k', size=24)
        plt.yticks(color='k', size=24)
        plt.grid()





#        for i in range(len(exp_ind)):
#            fit_data_file = peaks_files_fit_dir + 'RES-' + '%04d' %(i+1) + '_' + str_pwr + '_DBM_TEMP_' + '%.3f' %(int(str_temp)/1000) + '.txt'
#            np.savetxt(fit_data_file, np.c_[res_list[exp_ind[i]].freq, res_list[exp_ind[i]].I, res_list[exp_ind[i]].Q, res_list[exp_ind[i]].resultI, res_list[exp_ind[i]].resultQ], delimiter='\t', fmt=('%.0f','%.5f','%.5f','%.5f','%.5f'), header='Freq (Hz)    I      Q      Fit I     Fit Q')

#        alphaQi_arr, DeltaQi_arr, errQi_arr, Qi0_arr = [], [], [], []

#        plt.figure()
#        figManager = plt.get_current_fig_manager()
#        figManager.window.showMaximized()
#        alphaQi_arr, DeltaQi_arr, errQi_arr, Qi0_arr = [], [], [], []
#        for l in range(len(f0_data[0])):
#            f0list = np.array([a[l] for a in f0_data])
#            Tlist = np.array([a[l] for a in T_data])*1e-3
#            Qilist = np.array([a[l] for a in qi_data])
#            
#            lbQi = np.array([0.05, 1e-7, Qilist[0]*0.5])
#            ubQi = np.array([5, 1e-2, Qilist[0]*2])
#                   
#            xoptQi, foptQi = pso(minimize_Qi_for_pso, lbQi, ubQi, ieqcons=[], f_ieqcons=None, args=(Tlist, f0list, Qilist), kwargs={},
#                swarmsize=2000, omega=0.5, phip=0.5, phig=0.5, maxiter=200, minstep=1e-8,
#                minfunc=1e-8, debug=False)
#                        
#            alphaQi, DeltaQi, Qi0 = xoptQi
#            
#            alphaQi_arr.append(alphaQi)
#            DeltaQi_arr.append(DeltaQi)
#            Qi0_arr.append(Qi0)
#            errQi_arr.append(foptQi*1e-6)
#
#            d_1vsQi_fit = 1/MB_calc_Qi(Tlist, f0list, Qilist, alphaQi, DeltaQi, Qi0) - 1/Qi0
#            d_1vsQi = 1/Qilist - 1/Qi0
#            Tnew = np.linspace(Tlist[0],Tlist[-1],100)
#            d_1vsQi_fit_smooth = interp1d(Tlist,d_1vsQi_fit, kind='cubic')
#            color = plt.cm.tab20(l%20)
#            plt.plot(Tnew*1e3,d_1vsQi_fit_smooth(Tnew)*1e3,  color=color)
#            plt.plot(Tlist*1e3, d_1vsQi*1e3, 'o', markersize = 8,  color=color)
#            plt.plot(Tlist*1e3, d_1vsQi_fit*1e3, 'x', markersize = 8,  color=color)
#            
#                       
#        #plt.legend(str_leg, fontsize = 14, loc=3)
#        plt.xlabel('Temperature [mK]', fontsize=18)
#        plt.ylabel(r'[1/Qi(T) - 1/Qi(0)]   $\times 10^3$ ', fontsize=18)
#        plt.xticks(color='k', size=16)
#        plt.yticks(color='k', size=16)
#        plt.grid()
#           
#        
#        errQi_arr = np.array(errQi_arr)
#        alphaQi_arr = np.array(alphaQi_arr)
#        DeltaQi_arr = np.array(DeltaQi_arr)
#        Qi0_arr = np.array(Qi0_arr)



        
        alpha_hist, alpha_bins = np.histogram(alphaFres_arr, bins = 15)
        alpha_x = (alpha_bins[1:] + alpha_bins[0:-1])/2
        ind, err_arr_sorted = [], []
        for n in range(len(alpha_bins)-1):
            ind.append(np.where((alphaFres_arr>=alpha_bins[n]) & (alphaFres_arr<=alpha_bins[n+1]))[0])
            err_arr_sorted.append(np.mean(errFres_arr[ind[n]]))
        
        err_sorted_scaled = err_arr_sorted/max(err_arr_sorted)*max(alpha_hist)
        
        fig11, ax11 = plt.subplots(figsize=(width_screen/mydpi, height_screen/mydpi), dpi=mydpi)
        plt.subplots_adjust(bottom=0.1, top=0.96, right=0.97, left=0.09)
        width = (max(alpha_x) - min(alpha_x))/20
        plt.bar(alpha_x, alpha_hist, width = width)
        plt.bar(alpha_x+width/2, err_sorted_scaled, width = width/5)
        plt.xlabel(r'$\alpha$ (Fres fit)', fontsize=26)
        plt.ylabel('Number of resonances', fontsize=26)
        plt.xticks(color='k', size=24)
        plt.yticks(color='k', size=24)
        plt.legend([r'$\alpha$','Fit error [a.u.]'], fontsize = 20)
        plt.grid()
        
        
        Delta_hist, Delta_bins = np.histogram(DeltaFres_arr, bins = 15)
        Delta_x = (Delta_bins[1:] + Delta_bins[0:-1])/2
        ind, err_arr_sorted = [], []
        for n in range(len(Delta_bins)-1):
            ind.append(np.where((DeltaFres_arr>=Delta_bins[n]) & (DeltaFres_arr<=Delta_bins[n+1]))[0])
            err_arr_sorted.append(np.mean(errFres_arr[ind[n]]))
        
        err_sorted_scaled = err_arr_sorted/max(err_arr_sorted)*max(Delta_hist)
        
        fig12, ax12 = plt.subplots(figsize=(width_screen/mydpi, height_screen/mydpi), dpi=mydpi)
        plt.subplots_adjust(bottom=0.1, top=0.96, right=0.97, left=0.09)
        width = (max(Delta_x) - min(Delta_x))/20*1e3
        plt.bar(Delta_x*1e3, Delta_hist, width = width)
        plt.bar(Delta_x*1e3+width/2, err_sorted_scaled, width = width/5)
        plt.xlabel(r'$\Delta$ (Fres fit) [meV]', fontsize=26)
        plt.ylabel('Number of resonances', fontsize=26)
        plt.xticks(color='k', size=24)
        plt.yticks(color='k', size=24)
        plt.legend([r'$\Delta$','Fit error [a.u.]'], fontsize = 20)
        plt.grid()
        

#        alphaQi_hist, alphaQi_bins = np.histogram(alphaQi_arr, bins = 15)
       
#        max_alphaQi = min([np.median(alphaQi_arr)*3, max(alphaQi_arr)*1.15, np.percentile(alphaQi_arr,98)])
#        min_alphaQi = max([np.median(alphaQi_arr)/5, min(alphaQi_arr)*0.85, np.percentile(alphaQi_arr,1)])
#        bins = np.linspace(min_alphaQi, max_alphaQi, 15)
        alphaQi_hist, alphaQi_bins = np.histogram(alphaQi_arr, bins = 15)
#        alphaQi_hist, alphaQi_bins = np.histogram(alphaQi_arr, bins = [0, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.15, 1.2, 100])
#        alphaQi_x = np.array([0.575, 0.625, 0.675, 0.725, 0.775, 0.825, 0.875, 0.925, 0.975, 1.025, 1.075, 1.125, 1.175, 1.225])

        alphaQi_x = (alphaQi_bins[1:] + alphaQi_bins[0:-1])/2
        ind, errQi_arr_sorted = [], []
        for n in range(len(alphaQi_bins)-1):
            ind.append(np.where((alphaQi_arr>=alphaQi_bins[n]) & (alphaQi_arr<=alphaQi_bins[n+1]))[0])
            errQi_arr_sorted.append(np.mean(errQi_arr[ind[n]]))
        
        errQi_sorted_scaled = errQi_arr_sorted/max(errQi_arr_sorted)*max(alphaQi_hist)

        plt.figure()
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.subplots_adjust(bottom=0.1, top=0.96, right=0.97, left=0.09)
        width = (max(alphaQi_x) - min(alphaQi_x))/20
        plt.bar(alphaQi_x, alphaQi_hist, width = width)
        plt.bar(alphaQi_x+width/2, errQi_sorted_scaled, width = width/5)
        plt.xlabel(r'$\alpha$ (Qi fit)', fontsize=26)
        plt.ylabel('Number of resonances', fontsize=26)
        plt.xticks(color='k', size=24)
        plt.yticks(color='k', size=24)
#        plt.xlim(0.4, 1.2)
#        labels = [item.get_text() for item in ax.get_xticklabels()]
#        labels[0] = '0'
#        labels[-1] = '100'
#        ax.set_xticklabels(labels)
        plt.legend([r'$\alpha$','Fit error [a.u.]'], fontsize = 20)
        plt.grid()


#        max_deltaQi = min([np.median(DeltaQi_arr)*3, max(DeltaQi_arr)*1.15, np.percentile(DeltaQi_arr,98)])
#        min_deltaQi = max([np.median(DeltaQi_arr)/3, min(DeltaQi_arr)*0.85, np.percentile(DeltaQi_arr,1)])
#        bins = np.linspace(min_deltaQi, max_deltaQi, 15)
        bins = np.linspace(2e-4, 3e-4, 15)

        DeltaQi_hist, DeltaQi_bins = np.histogram(DeltaQi_arr, bins = bins)
#        DeltaQi_hist, DeltaQi_bins = np.histogram(DeltaQi_arr, bins = 1e-3*np.array([0, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 100]))
        DeltaQi_x = (DeltaQi_bins[1:] + DeltaQi_bins[0:-1])/2
#        DeltaQi_x = np.array([ 0.135, 0.145,   0.155,   0.165,   0.175,   0.185,   0.195,   0.205,  0.215,   0.225,   0.235,   0.245,   0.255,   0.265])*1e-3
        
        ind, errQi_arr_sorted = [], []

        for n in range(len(DeltaQi_bins)-1):
            ind.append(np.where((DeltaQi_arr>=DeltaQi_bins[n]) & (DeltaQi_arr<=DeltaQi_bins[n+1]))[0])
            errQi_arr_sorted.append(np.mean(errQi_arr[ind[n]]))

        errQi_sorted_scaled = errQi_arr_sorted/np.nanmax(errQi_arr_sorted)*np.nanmax(DeltaQi_hist)

        plt.figure()
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.subplots_adjust(bottom=0.1, top=0.96, right=0.97, left=0.09)
#        width = (max(DeltaQi_x) - min(DeltaQi_x))/20*1e3
        width = (max(DeltaQi_x*1e3) - min(DeltaQi_x*1e3))/20
        plt.bar(DeltaQi_x*1e3, DeltaQi_hist, width = width)
        plt.bar(DeltaQi_x*1e3+width/2, errQi_sorted_scaled, width = width/5)
        plt.xlabel(r'$\Delta$ (Qi fit) [meV]', fontsize=26)
        plt.ylabel('Number of resonances', fontsize=26)
        plt.xticks(color='k', size=24)
        plt.yticks(color='k', size=24)
#        plt.xlim(0.13, 0.27)
        plt.legend([r'$\Delta$','Fit error [a.u.]'], fontsize = 20)
        plt.grid()
        
        
        
        
        
    def generate_pdf(self, var):
        device_dir = self.device_dir
        
        if glob.glob(device_dir + 'Figures/TiN_reso/Reports/Fit_reso*/'):
            split_fit_dir = glob.glob(device_dir + 'Figures/TiN_reso/Reports/Fit_reso*/')[0]
        else:
            root = tk.Tk()
            root.withdraw()
            split_fit_dir = filedialog.askdirectory(title = "Select the directory of split fitted data (ex. split_fit_data)") # Open a dialog window to select the file
            
        if glob.glob(device_dir + 'Data/TiN_reso/Fit_reso_params*/'):
            fit_params_dir = glob.glob(device_dir + 'Data/TiN_reso/Fit_reso_params*/')[0]
        else:
            root = tk.Tk()
            root.withdraw()
            fit_params_dir = filedialog.askdirectory(title = "Select the Fit params directory (ex. Fit_params_Be170227bl)") # Open a dialog window to select the file
    
        pdf_dir = device_dir + '/pdf_reports/'
        
        if not os.path.exists(pdf_dir):
            print(pdf_dir + ' directory has been created') 
            os.makedirs(pdf_dir)
            
        
        N_dir = len(os.listdir(split_fit_dir))
        for i in range(N_dir):
            curr_dir = np.sort(os.listdir(split_fit_dir))[i]
            N_files = len(os.listdir(split_fit_dir + curr_dir))
            fit_param_file = np.sort(os.listdir(fit_params_dir))[i]
            fit_params = np.loadtxt(fit_params_dir + fit_param_file)

            str_name = device_dir.split('/')[-1]
            str_pwr = fit_param_file.split('_')[1] # power in dBm
            str_temp = fit_param_file.split('_')[4][:-4] # Temp in K
            str_temp = str(int(float(str_temp)*1000))
            
            filename_report = pdf_dir + "fit_report_" + str_name + '_' + str_temp + 'mK_' + str_pwr + 'dBm'
            Report_pdf = PdfPages(filename_report + '.pdf')
            
            if len(fit_params) == N_files:
                for k in range(N_files):
                    curr_file = np.sort(os.listdir(split_fit_dir + curr_dir))[k]
                    data = np.loadtxt(split_fit_dir + '/' + curr_dir + '/' + curr_file)
                    freq, Iraw, Qraw, Ifit, Qfit = data[:,0], data[:,1], data[:,2], data[:,3], data[:,4]
                    Mag = np.sqrt(Iraw**2 + Qraw**2)
                    fitMag = np.sqrt(Ifit**2 + Qfit**2)
                    resoNum, fres0, Qi, Qc = fit_params[k,:]
                    
                    fitMag_inter = interp1d(freq, fitMag, kind='cubic')
                    Ifit_inter = interp1d(freq, Ifit, kind='cubic')
                    Qfit_inter = interp1d(freq, Qfit, kind='cubic')
                    
                    freq_inter = np.linspace(freq.min(), freq.max(), 5000)

                    plt.figure(figsize=(8.5, 11))
                    plt.subplots_adjust(bottom=0.08, top=0.8, right=0.95, left=0.11)
                    plt.subplots_adjust(wspace=0.42, hspace=0.42)
                    
                    plt.subplot(2,2,1)
                    ax = plt.gca()
                    ax.get_xaxis().get_major_formatter().set_useOffset(False)
                    plt.plot(freq*1e-6, 20*np.log10(np.abs(Mag)),'.', label='Data')
                    plt.plot(freq_inter*1e-6, 20*np.log10(np.abs(fitMag_inter(freq_inter))), label='Fit', color='red')
                    plt.plot(fres0*1e-6, 20*np.log10(np.abs(fitMag_inter(fres0))), '*', markersize=10, color='orange', label='$f_{r}$')
                    plt.xlabel("Frequency [MHz]", fontsize = 14)
                    plt.xticks(np.linspace(min(freq*1e-6), max(freq*1e-6), 3))
                    plt.xticks(color='k', size=12)
                    plt.yticks(color='k', size=12)
                    plt.ylabel("$|S_{21}|^2$ [dB]", fontsize = 14)
                    plt.grid()
                    plt.legend(bbox_to_anchor=(1.4, -0.08), fontsize = 12)
            
                    plt.subplot(2,2,2)
                    plt.gca().set_aspect('equal', adjustable='box')
                    plt.plot(Iraw, Qraw,'.', label='Data')
                    plt.plot(Ifit_inter(freq_inter), Qfit_inter(freq_inter),  label='Fit', color='red')
                    plt.plot(Ifit_inter(fres0), Qfit_inter(fres0), '*', markersize=10, color='orange',  label='Fr')
                    plt.xlabel("$S_{21}$ real", fontsize = 14)
                    plt.xticks([min(Iraw),max(Iraw)], fontsize = 12)
                    plt.ylabel("$S_{21}$ imaginary", fontsize = 14)
                    plt.yticks([min(Qraw),max(Qraw)], fontsize = 12)
                    
                    plt.subplot(2,2,3)
                    plt.plot(freq*1e-6, Iraw,'.', label='Data')
                    plt.plot(freq_inter*1e-6, Ifit_inter(freq_inter),  label='Fit', color='red')
                    plt.plot(fres0*1e-6, Ifit_inter(fres0), '*', markersize=10, color='orange',  label='Fr')
                    plt.xlabel("Frequency [MHz]", fontsize = 14)
                    plt.xticks(np.linspace(min(freq*1e-6), max(freq*1e-6), 3))
                    plt.ylabel("$S_{21}$ real", fontsize = 14)
                    plt.xticks(color='k', size=12)
                    plt.yticks(color='k', size=12)
                    plt.grid()
                    
                    plt.subplot(2,2,4)
                    plt.plot(freq*1e-6, Qraw,'.', label='Data')
                    plt.plot(freq_inter*1e-6, Qfit_inter(freq_inter),  label='Fit', color='red')
                    plt.plot(fres0*1e-6, Qfit_inter(fres0), '*', markersize=10, color='orange',  label='Fr')
                    plt.xlabel("Frequency [MHz]", fontsize = 14)
                    plt.xticks(np.linspace(min(freq*1e-6), max(freq*1e-6), 3))
                    plt.ylabel("$S_{21}$ imag", fontsize = 14)
                    plt.xticks(color='k', size=12)
                    plt.yticks(color='k', size=12)
                    plt.grid()
                    
                    str_params = "VNA Power = " + str_pwr + " dBm \n" + "Input Power = %d" %(int(str_pwr)-atten) + " dBm \n" + "Temp = " + str_temp + " mK"
                    fitwords = "Res Num : " + "%d" %resoNum + " \n" + "$f_{res}$ = " + "%.6f" %(fres0*1e-6) + " MHz \n" + "$Q_{c}$ = " + "%.2E" %Qc + "\n" + "$Q_{i}$ = " + "%.2E" %Qi #+ "\n" + "$a$ = " + "%.5f" %a[i].real + " + " + "%.5f" %a[i].imag + "j\n" + "$\phi_{0}$ = " + "%.5f" %phi[i] + " \n" + r"$\tau$ = " + "%.5f" %tau[i] + " \n"

                    plt.figtext(0.4, 0.96, str_name, fontsize = 14)
                    plt.figtext(0.1, 0.83, fitwords, fontsize = 14)
                    plt.figtext(0.6, 0.83, str_params, fontsize = 14)

                    Report_pdf.savefig()
                    plt.close()

                Report_pdf.close()
#                str_pdf_reduce = 'qpdf --linearize ' +  filename_report + '.pdf ' + filename_report + '_LowSize.pdf'
#                os.system(str_pdf_reduce)
            else: print('pdf not generated. N_files different from the number fit params!')


callback = analyze_data()
s_move_win.on_changed(callback.move_win)
s_win_width.on_changed(callback.change_win_width)
b_export.on_clicked(callback.export_cleaned_params)
b_plot_Qi_Qc.on_clicked(callback.plot_Qi_Qc)
b_fit_Qi_f0.on_clicked(callback.fit_Qi_f0)
b_pdf.on_clicked(callback.generate_pdf)

fig.canvas.mpl_connect('key_press_event', callback.press)
fig.canvas.mpl_connect('pick_event', callback.select_data)
plt.show()   







#start = time.time()
##get_nqp(T, Delta*0.98)
#2 * N0 * np.sqrt(2 * np.pi * kb * T * Delta) * np.exp(-Delta/(kb * T))
#end = time.time()
#
#
#print(end - start)
#def MB_calc_f(T, alpha, Cte, Tc, f0):
##    alpha, Cte, Tc, f0 = params
#
#    kb = 8.6173423e-5 # eV/K
#    N0 = 3.9e10 # eV ^-1 um^-3 for TiN (A.E. Lowitz, E.M. Barrentine, S.R. Golwala, P.T. Timbie, LTD 2013)
#    h = 4.13567e-15 # =2 pi hbar in eV s
#    
#    hbar= h/(2 * np.pi)
#    Delta0 = Cte * kb * Tc
#    nqp = 2 * N0 * np.sqrt(2 * np.pi * kb * T * Delta0) * np.exp(-Delta0/(kb * T))
##    nqp = get_nqp(T, Delta0, G)
#    zeta = (hbar * np.pi * f_result)/(kb * T)
##    print(f_result[0])
#    k2 = 1 / (N0 * Delta0) * (1+ np.sqrt(2 * Delta0/(np.pi * kb * T)) * np.exp(-zeta) * sp.iv(0, zeta))
#    df_f0 = -alpha/2 * k2 * nqp
#    return (df_f0) * f0 + f0
#
#popt_arr = []
#for k in range(4):#len(f0_data[0])):
#    f_result = np.array([a[k] for a in f0_data])
#    Tlist = np.array([a[k] for a in T_data])*1e-3
#    f0list = f_result
#    for i in range(10):
#        popt, pcov = sc.optimize.curve_fit(MB_calc_f, Tlist, f0list, bounds=([0.1, 0.1, 0.6, max(f0list)*0.999], [10, 6, 1.5, max(f0list)*1.02]))
#        f_result = MB_calc_f(Tlist, *popt)
#    popt_arr.append(popt)
#    #    
#    ax2.plot( MB_calc_f(Tlist, *popt), Tlist*1e3, '+')
##    plt.figure()
##    plt.plot(Tlist, f0list, 'o')
##    plt.plot(Tlist, MB_calc_f(Tlist, *popt), '+')
#
#
#
#

#    
#    
#    
#
#lb = np.array([0.05, 1e-7, f0list[0]*1])
#ub = np.array([5, 1e-2, f0list[0]*1.03])
#
#plt.figure()
#figManager = plt.get_current_fig_manager()
#figManager.window.showMaximized()
#plt.plot(Tlist*1e3, f0list*1e-6, 'o', markersize = 8)
#str_leg = ['Data points']
#
#for mm in range(5):
#    xopt, fopt = pso(minimize_for_pso, lb, ub, ieqcons=[], f_ieqcons=None, args=(Tlist, f0list), kwargs={},
#        swarmsize=2000, omega=0.5, phip=0.5, phig=0.5, maxiter=200, minstep=1e-8,
#        minfunc=1e-8, debug=False)
#    
#    alpha, Delta, f0 = xopt
#    T, f = Tlist, f0list
#    plt.plot(Tlist*1e3, MB_calc_f(T, f, alpha, Delta, f0)*1e-6, 'x', markersize = 8)
#    str_leg.append(r"$\alpha$ = %.3f, $f_0$ = %.4f MHz, $\Delta$ = %.3e, err = %.2f" %(alpha, f0*1e-6, Delta, fopt*1e-6))
##plt.figure()
##plt.plot(Tlist, f0list, 'o')
#plt.legend(str_leg, fontsize = 14, loc=3)
#plt.xlabel('Temperature [mK]', fontsize=18)
#plt.ylabel('Frequency [MHz]', fontsize=18)
#plt.xticks(color='k', size=16)
#plt.yticks(color='k', size=16)
#plt.grid()
#plt.tight_layout()



#Tnew = np.linspace(Tlist[0],Tlist[-1],100)
#dfvsf0_fit_smooth = spline(Tlist,dfvsf0_fit,Tnew)
#plt.plot(Tnew,dfvsf0_fit_smooth)
#plt.show()






    
#Tinter = np.linspace(min(Tlist), max(Tlist), 50)
#
#def MB_calc_f2(f):
#    alpha, Cte, Tc, f0 = xopt
#    kb = 8.6173423e-5 # eV/K
#    N0 = 3.9e10 # eV ^-1 um^-3 for TiN (A.E. Lowitz, E.M. Barrentine, S.R. Golwala, P.T. Timbie, LTD 2013)
#    h = 4.13567e-15 # =2 pi hbar in eV s
#
#    hbar= h/(2 * np.pi)
#    Delta0 = Cte * kb * Tc
#    nqp = 2 * N0 * np.sqrt(2 * np.pi * kb * T * Delta0) * np.exp(-Delta0/(kb * T))
#    zeta = (hbar * np.pi * f)/(kb * T)
#    k2 = 1 / (N0 * Delta0) * (1+ np.sqrt(2 * Delta0/(np.pi * kb * T)) * np.exp(-zeta) * sp.iv(0, zeta))
#    df_f0 = -alpha/2 * k2 * nqp
#    return np.sum(abs((df_f0) * f0 + f0 - f))
#
#lb = [f0list[0]*0.98]
#ub = [f0list[-1]*1.02]
#finter = np.zeros(len(Tinter))
#
#for i in range(len(Tinter)):
#    T = Tinter[i]
#    xopt2, fopt2 = pso(MB_calc_f2, lb, ub, ieqcons=[], f_ieqcons=None, args=(), kwargs={},
#        swarmsize=100, omega=0.5, phip=0.5, phig=0.5, maxiter=400, minstep=1e-8,
#        minfunc=1e-8, debug=False)
#    finter[i] = xopt2
    
    











#
#
#import lmfit as lf
#
#f0_params = lf.Parameters()
#
##Resonant frequency at zero temperature and zero power
#f0_guess = f0_data[0][0]
#
#f0_params.add('f0',
#              value = f0_guess,
#              min = f0_guess*0.95,
#              max = f0_guess*1.05)
#
##The loss roughly equivalent to tan delta
#f0_params.add('Fd',
#              value = 1e-6,
#              min = 1e-8)
#
##The kinetic inductance fraction
#f0_params.add('alpha',
#              value = 0.005,
#              min = 0,
#              max = 1)
#
##The BCS energy gap at zero temperature
#f0_params.add('delta0',
#              value = 5e-4,
#              min = 1e-5,
#              max = 1e-3,)
#
##Qi needs all of the above parameters, plus a few more
#qi_params = f0_params.copy()
#
##Q at zero power and zero temperature
#qi_params.add('q0',
#              value = 4e4,
#              min = 1e3,
#              max = 1e6)
#
##Critical power in W (modulo some calibration)
#qi_params.add('Pc',
#              value = 4,
#              min = 0,
#              max = 10000)
#
##Set the max temperature to fit to
#max_fit_temp = 800
#
#
#coco = scr.fitsSweep.f0_tlsAndMBT(qi_params, np.array([a[0] for a in T_data])*1e3, -52, [a[0] for a in f0_data])
#
#
#plt.figure()
#plt.plot(np.array([a[0] for a in T_data])*1e3, [a[0] for a in f0_data], 'o')
#plt.plot(np.array([a[0] for a in T_data])*1e3, coco)
#
#resSweeps['RES-1'].do_lmfit(['qi'],
#                            [scr.fitsSweep.qi_tlsAndMBT], #The model
#                            [qi_params], #The paramters
#                            min_pwr=-70, #S21 fits below -70 were bad
#                            max_temp=max_fit_temp)
#
#resSweeps['RES-1'].do_lmfit(['f0'],
#                            [scr.fitsSweep.f0_tlsAndMBT], #The model
#                            [f0_params], #The paramters
#                            min_pwr=-70, #S21 fits below -70 were bad
#                            max_temp=max_fit_temp)
#
#scr.do_lmfit(['f0'],
#                            [scr.fitsSweep.f0_tlsAndMBT], #The model
#                            [f0_params], #The paramters
#                            min_pwr=-70, #S21 fits below -70 were bad
#                            max_temp=max_fit_temp)
#
#
#resNames = ['RES-1']
#dataPath  = 'test/'
#resLists = {}
#for resName in resNames:
#    resLists[resName] = scr.makeResList(scr.process_file,
#                                        dataPath,
#                                        resName,
#                                        skiprows=1, delimiter='\t')
#    
#for resName in resNames:
#    for res in resLists[resName]:
#        if res.pwr < -60:
#            #Apply a filter to the data before
#            #guessing parameters for low-power measuremnts
#            res.load_params(scr.cmplxIQ_params,
#                            use_filter=True)
#        else:
#            res.load_params(scr.cmplxIQ_params,
#                            use_filter=False)
#
#        res.do_lmfit(scr.cmplxIQ_fit)
#        
#import pickle
#fName = 'saved_data.pickle'
#fPath = os.path.join('./', fName)
#
#with open(fPath, 'wb') as f:
#    pickle.dump(resLists, f, 2)
#
#print('last saved file was: '+fName)
#
#from importlib import reload
#scr.plot_tools = reload(scr.plot_tools)
#
#fig1a = scr.plot_tools.plotResListData(resLists['RES-1'],
#                            plot_types=['IQ', 'LogMag', 'uPhase'],
#                            detrend_phase = True,
#                            plot_fits = [True, False, False],
#                            color_by='temps',
#                            num_cols = 3,
#                            fig_size=3,
#                            powers = [-52],
#                            #the fit defaults to a thick dashed line. Small plots are nicer with a thinner line
#                            fit_kwargs={'linestyle':'--', 'color':'k', 'linewidth':1})
#
##Uncomment to save the figure
##fig1a.savefig('fig1a.pdf')
#
#
#resSweeps = {}
#for resName, resList in resLists.items():
#    resSweeps[resName] = scr.ResonatorSweep(resList, index='block')
#
##Look at the uncertainties on the best-fit frequencie
##for the first few files of 'RES-1'
#resSweeps['RES-1']['f0_sigma'].head()
#
#fig1c = scr.plotResSweepParamsVsTemp(resSweeps['RES-1'],
#                                    fig_size = 3,
#                                    plot_keys = ['f0', 'qi'],
#                                    plot_labels = ['$f_0$ (GHz)',
#                                                   '$Q_\mathrm{i}$'],
#                                    unit_multipliers = [1e-9, 1],
#                                    num_cols = 1,
#                                    powers = [-52],
#                                    force_square=True)
##Uncomment to save the figure
##fig1c.savefig('fig1c.pdf')
#
#
#f0_params = lf.Parameters()
#
##Resonant frequency at zero temperature and zero power
#f0_guess = resSweeps['RES-1']['f0'].iloc[0, 0]
#f0_params.add('f0',
#              value = f0_guess,
#              min = f0_guess*0.95,
#              max = f0_guess*1.05)
#
##The loss roughly equivalent to tan delta
#f0_params.add('Fd',
#              value = 1e-6,
#              min = 1e-8)
#
##The kinetic inductance fraction
#f0_params.add('alpha',
#              value = 0.005,
#              min = 0,
#              max = 1)
#
##The BCS energy gap at zero temperature
#f0_params.add('delta0',
#              value = 5e-4,
#              min = 1e-5,
#              max = 1e-3,)
#
##Qi needs all of the above parameters, plus a few more
#qi_params = f0_params.copy()
#
##Q at zero power and zero temperature
#qi_params.add('q0',
#              value = 4e5,
#              min = 1e4,
#              max = 1e6)
#
##Critical power in W (modulo some calibration)
#qi_params.add('Pc',
#              value = 4,
#              min = 0,
#              max = 10000)
#
##Set the max temperature to fit to
#max_fit_temp = 800
#
#
#resSweeps['RES-1'].do_lmfit(['qi'],
#                            [scr.fitsSweep.qi_tlsAndMBT], #The model
#                            [qi_params], #The paramters
#                            min_pwr=-70, #S21 fits below -70 were bad
#                            max_temp=max_fit_temp)
#
#resSweeps['RES-1'].do_lmfit(['f0'],
#                            [scr.fitsSweep.f0_tlsAndMBT], #The model
#                            [f0_params], #The paramters
#                            min_pwr=-70, #S21 fits below -70 were bad
#                            max_temp=max_fit_temp)
#
##Uncomment to look at the results of the fit
##lf.report_fit(resSweeps['RES-1'].lmfit_results['qi'])
#
#fig2a = scr.plotResSweep3D(resSweeps['RES-1'],
#                           plot_keys=['f0'],
#                           max_temp=775,
#                           unit_multipliers=[1e-9],
#                           plot_labels = ['$f_0$ (GHz)'],
#                           min_pwr=-70,
#                           fig_size=5,
#                           plot_lmfits=True)


