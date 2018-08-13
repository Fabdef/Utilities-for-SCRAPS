#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 15:01:04 2018

@author: Fabien Defrance
"""

import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from matplotlib.widgets import Slider, Button
import os
import sys
sys.path.append('/home/ubadmin/Documents/Caltech/local_python_library/')
#sys.path.append('C:/Users/DELL E7270/Documents/Documents/Travail/Caltech/local_python_library/')
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
devname = device_dir.split('/')[-2] # Device Name

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
#        
#    if not hasattr(res_list[0], 'devname'): 
#        for k in range(len(res_list)): res_list[k].devname = devname
        
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
plt.subplots_adjust(bottom=0.3, top=0.96, right=0.96, left=0.08)

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

ax_color = 'lightgoldenrodyellow' # Color of the sliders and buttons
b_color2 = 'lightgreen' # Color of some buttons

ax_move_win = plt.axes([0.14, 0.14, 0.4, 0.03], facecolor=ax_color)
s_move_win = Slider(ax_move_win, 'Window position', 0, 1, valinit = 0)
s_move_win.label.set_fontsize(14)
s_move_win.valtext.set_fontsize(14)

ax_win_width = plt.axes([0.14, 0.08, 0.4, 0.03], facecolor=ax_color)
s_win_width = Slider(ax_win_width, 'Window size', 0, 1, valinit = win_size_ini)
s_win_width.label.set_fontsize(14)
s_win_width.valtext.set_fontsize(14)

ax_export = plt.axes([0.6, 0.16, 0.1, 0.05])
b_export = Button(ax_export, 'Export params', color=ax_color, hovercolor='0.975')
b_export.label.set_fontsize(14)

ax_plot_Qi_Qc = plt.axes([0.6, 0.085, 0.1, 0.05])
b_plot_Qi_Qc = Button(ax_plot_Qi_Qc, 'Plot Qi & Qc', color=ax_color, hovercolor='0.975')
b_plot_Qi_Qc.label.set_fontsize(14)

ax_save_Qi_Qc = plt.axes([0.625, 0.05, 0.05, 0.035])
b_save_Qi_Qc = Button(ax_save_Qi_Qc, 'Save', color=b_color2, hovercolor='green')
b_save_Qi_Qc.label.set_fontsize(14)

ax_fit_Qi_f0 = plt.axes([0.72, 0.085, 0.1, 0.05])
b_fit_Qi_f0 = Button(ax_fit_Qi_f0, 'Fit Qi & f0 (MB)', color=ax_color, hovercolor='0.975')
b_fit_Qi_f0.label.set_fontsize(14)

ax_plot_MBfit = plt.axes([0.72, 0.05, 0.045, 0.035])
b_plot_MBfit = Button(ax_plot_MBfit, 'Plot', color=b_color2, hovercolor='green')
b_plot_MBfit.label.set_fontsize(14)

ax_save_MBfit = plt.axes([0.775, 0.05, 0.045, 0.035])
b_save_MBfit = Button(ax_save_MBfit, 'Save', color=b_color2, hovercolor='green')
b_save_MBfit.label.set_fontsize(14)

ax_fitres_pdf = plt.axes([0.72, 0.16, 0.1, 0.05])
b_fitres_pdf = Button(ax_fitres_pdf, 'Res fit report', color=ax_color, hovercolor='0.975')
b_fitres_pdf.label.set_fontsize(14)

ax_fitMB_pdf = plt.axes([0.84, 0.16, 0.1, 0.05])
b_fitMB_pdf = Button(ax_fitMB_pdf, 'MB fit report', color=ax_color, hovercolor='0.975')
b_fitMB_pdf.label.set_fontsize(14)

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
        
        
    # Check that there is an equal number of data points at each temperature
    # This requirement is due to the fact that resonances are not labeled or referenced
    # so we need to assume that resonance with index n at T1 degrees corresponds to resonance index n at T2 degrees
    # The resonances are saved as pickles  
    def export_cleaned_params(self, var):
        fres_tmp = np.array(fres_data_arr)
        fres_tmp[self.indx, self.indy] = np.NaN
        ind_fitMB = []
        # Check the resonance frequencies for the MB fit (not selected on the plot and discard any potential duplicate)
        for i in range(np.size(fres_tmp,1)):
            ind = np.where(~np.isnan(fres_tmp[:,i]))[0]
            ind_fitMB.append(ind)
        # Check that there is an equal number of resonances at each temperature
        if len(np.unique([len(a) for a in ind_fitMB])) > 1:
            print('Not equal number of data for each temperature')
            print([len(a) for a in ind_fitMB])
        else:
            for i in range(len(res_data)):
                # save
                res_list_save_file = obj_save_dir + 'pickle_' + str_pwr + '_DBM_TEMP_' + '%.3f' %(temp[i]) + '.p'
                pickle.dump(res_data[i], open(res_list_save_file,'wb'))

    # Read the pickle saved at the previous function.
    # It does not seem so useful to have 2 functions to save and read, but it can become useful if we
    # don't want to have to select all the resonances again and just want to read and plot the data
    def read_clean_files(self):
        fres_data, T_data, qi_data, qc_data, res_data, MB_ok_ind_data = [], [], [], [], [], []        
        pickle_files = np.sort(os.listdir(obj_load_dir))

        for filename in pickle_files:
#            data = np.loadtxt(cleaned_reso_dir + cleaned_param_files[i], delimiter ='\t')
            res_list = pickle.load(open(obj_save_dir + filename,'rb'))
            res_data.append(res_list)
            fres = np.array([a.lmfit_result.params['f0'].value for a in res_list])
            qi = np.array([a.lmfit_result.params['qi'].value for a in res_list])
            qc = np.array([a.lmfit_result.params['qc'].value for a in res_list])
            MB_ok_ind = np.where([a.MB_fit_ok for a in res_list])[0]
#            fit_ok_ind = np.where([a.fit_ok for a in res_list])[0]
#            MB_ok_ind_data.append(np.array(MB_ok_ind))
            temp = filename.split('_')[4][:-2]
            fres_data.append(fres[MB_ok_ind])
            qi_data.append(qi[MB_ok_ind])
            qc_data.append(qc[MB_ok_ind])
            N = len(fres_data[0])
            T_data.append(np.ones(N)*float(temp))
        
        self.res_data = res_data
        self.fres_data = fres_data
        self.qi_data = qi_data
        self.qc_data = qc_data
        self.T_data = T_data
        self.MB_ok_ind_data = MB_ok_ind_data
        
        
    # Plot Qi and Qc as function of temperature and as histograms
    def plot_Qi_Qc(self, var):
        self.read_clean_files()
        qi_data = self.qi_data
        qc_data = self.qc_data
        T_data = self.T_data
        
        fig1, ax1 = plt.subplots(figsize=(width_screen/mydpi, height_screen/mydpi), dpi=mydpi)
        fig1.canvas.set_window_title('Qi VS Temp') 
        plt.subplots_adjust(bottom=0.12, top=0.96, right=0.97, left=0.12)
        med = [np.median(a) for a in qi_data]
        for i in range(len(T_data[0])):
            T = [a[i] for a in T_data]
            Qi = [a[i] for a in qi_data]
            plt.plot(np.array(T)*1e3, Qi, 'o-', linewidth=2)  
        min_Qi = max([min([min(a) for a in qi_data])*0.85, min(med)/3])
        max_Qi = min([max([max(a) for a in qi_data])*1.15, max(med)*3])
        plt.yscale('log')
        plt.xlabel('Temperature [mK]', fontsize=38)
        plt.ylabel('Qi', fontsize=38)
        plt.xticks(color='k', size=34)
        plt.yticks(color='k', size=34)
        plt.ylim(min_Qi, max_Qi)
        plt.tick_params(axis='y', which='minor', labelsize=30)
        plt.grid(True, which="both") 
        plt.show()

        fig2, ax2 = plt.subplots(figsize=(width_screen/mydpi, height_screen/mydpi), dpi=mydpi)
        fig2.canvas.set_window_title('Histogram Qi') 
        plt.subplots_adjust(bottom=0.13, top=0.95, right=0.97, left=0.11)
        qi_hist = np.concatenate(qi_data)
        bins = np.logspace(np.log10(np.min(qi_hist)), np.log10(np.max(qi_hist)), 12)
#        med = [np.median(a) for a in qi_data]
#        min_Qi = max([min([min(a) for a in qi_data])*0.95, min(med)/3])
#        max_Qi = min([max([max(a) for a in qi_data])*1.05, max(med)*3])
        plt.hist( qi_data, alpha = 0.7, bins=bins)
        plt.xscale('log')
        plt.xlabel('Qi', fontsize=38)
        plt.ylabel('Number of resonances', fontsize=38)
        plt.xticks(color='k', size=34)
        plt.yticks(color='k', size=34)
        plt.legend([str(int(a[0]*1e3)) + ' mK' for a in T_data], fontsize=26)
        plt.xlim(min_Qi, max_Qi)
#        ax2.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
        plt.tick_params(axis='x', which='minor', labelsize=30)
        tx = ax2.xaxis.get_offset_text()
        tx.set_fontsize(26)
        plt.grid(True, which="both")     
        plt.show()

        fig3, ax3 = plt.subplots(figsize=(width_screen/mydpi, height_screen/mydpi), dpi=mydpi)
        fig3.canvas.set_window_title('Qc VS Temp') 
        plt.subplots_adjust(bottom=0.12, top=0.96, right=0.97, left=0.12)
        for i in range(len(T_data[0])):
            T = [a[i] for a in T_data]
            Qc = [a[i] for a in qc_data]
            plt.plot(np.array(T)*1e3, Qc ,'o-', linewidth=2)
        med = [np.median(a) for a in qc_data]
        min_Qc = max([min([min(a) for a in qc_data])*0.85, min(med)/5])
        max_Qc = min([max([max(a) for a in qc_data])*1.15, max(med)*5])
        plt.yscale('log')
        plt.xlabel('Temperature [mK]', fontsize=38)
        plt.ylabel('Qc', fontsize=38)
        plt.xticks(color='k', size=34)
        plt.yticks(color='k', size=34)
        plt.ylim(min_Qc, max_Qc)
        plt.tick_params(axis='y', which='minor', labelsize=30)
        plt.grid(True, which="both") 
        plt.show()
        
        fig4, ax4 = plt.subplots(figsize=(width_screen/mydpi, height_screen/mydpi), dpi=mydpi)
        fig4.canvas.set_window_title('Histogram Qc') 
        plt.subplots_adjust(bottom=0.13, top=0.95, right=0.97, left=0.11)
        qc_hist = np.concatenate(qc_data)
        bins = np.logspace(np.log10(np.min(qc_hist)), np.log10(np.max(qc_hist)), 12)
#        med = [np.median(a) for a in qi_data]
#        min_Qi = max([min([min(a) for a in qi_data])*0.95, min(med)/3])
#        max_Qi = min([max([max(a) for a in qi_data])*1.05, max(med)*3])
        plt.hist( qc_data, alpha = 0.7, bins=bins)
        plt.xscale('log')
        plt.xlabel('Qc', fontsize=38)
        plt.ylabel('Number of resonances', fontsize=38)
        plt.xticks(color='k', size=34)
        plt.yticks(color='k', size=34)
        plt.legend([str(int(a[0]*1e3)) + ' mK' for a in T_data], fontsize=26)
        plt.xlim(min_Qc, max_Qc)
#        ax2.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
        plt.tick_params(axis='x', which='minor', labelsize=30)
        tx = ax2.xaxis.get_offset_text()
        tx.set_fontsize(26)
        plt.grid(True, which="both")     
        plt.show()
        
        self.fig1 = fig1
        self.fig2 = fig2
        self.fig3 = fig3
        self.fig4 = fig4


    # Save the Qi and Qc plots
    def save_Qi_Qc(self,var):
        fig1 = self.fig1 
        fig2 = self.fig2 
        fig3 = self.fig3 
        fig4 = self.fig4 
        
        Q_plots_dir = plot_dir + 'Qi_Qc_fres/'
        if not os.path.exists(Q_plots_dir):
            print(Q_plots_dir + ' directory has been created') 
            os.makedirs(Q_plots_dir)
            
        filename = list([Q_plots_dir + devname + "_Qi_VS_temp"])
        filename.append(Q_plots_dir + devname + "_Qi_hist")
        filename.append(Q_plots_dir + devname + "_Qc_VS_temp")
        filename.append(Q_plots_dir + devname + "_Qc_hist")
        fig_num = [fig1, fig2, fig3, fig4]
        for t in range(len(filename)):
            fig_num[t].savefig(filename[t] + '.svg', dpi=fig.dpi, bbox_inches = 'tight')
            fig_num[t].savefig(filename[t] + '.png', dpi=fig.dpi, bbox_inches = 'tight')
            fig_num[t].savefig(filename[t] + '.pdf', dpi=fig.dpi, bbox_inches = 'tight')
            plt.close(fig_num[t])     
                
            
    def MB_calc_fbis(self,T, fres, alpha, Delta, f0):
        nqp = 2 * N0 * np.sqrt(2 * np.pi * kb * T * Delta) * np.exp(-Delta/(kb * T))
#            nqp = get_nqp(T, Delta)
        zeta = (hbar * np.pi * fres)/(kb * T)
        k2 = 1 / (2 * N0 * Delta) * (1+ np.sqrt(2 * Delta/(np.pi * kb * T)) * np.exp(-zeta) * sp.iv(0, zeta))
        df_f0 = -alpha/2 * k2 * nqp
        return (df_f0) * f0 + f0
    
    def MB_calc_Qibis(self,T, fres, Qi, alpha, Delta, Qi0):
        nqp = 2 * N0 * np.sqrt(2 * np.pi * kb * T * Delta) * np.exp(-Delta/(kb * T))
#            nqp = get_nqp(T, Delta)
        zeta = (hbar * np.pi * fres)/(kb * T)
        k1 = 1 / (np.pi * N0 * Delta) * np.sqrt(2 * Delta/(np.pi * kb * T)) * np.sinh(zeta) * sp.kn(0, zeta)
        d_1VSQi = alpha * k1 * nqp
        return 1/((d_1VSQi) + 1/Qi0)
        
        
    # Mattis Bardeen fit for Qi and fres
    # For this fit we assumed a fixed value of alpha (0.9) because it gives wrong results when trying to fit it as well
    # fits are correct with only Delta0, and f0 or f0 and Qi0 and to fit. Adding alpha to the variables
    # gives too much freedom to the eqation and the solution doesn't converge
    def fit_Qi_f0(self,var):
        self.read_clean_files()
        fres_data = self.fres_data
        qi_data = self.qi_data
#        qc_data = self.qc_data
        T_data = self.T_data
#        MB_ok_ind_data = self.MB_ok_ind_data
   
    # PSO algorithm is used for the fit
    # An approximate value of nqp is used. The more accurate value (integral solution should be implemented some day)
#        def MB_calc_fbis(T, fres, alpha, Delta, f0):
#            nqp = 2 * N0 * np.sqrt(2 * np.pi * kb * T * Delta) * np.exp(-Delta/(kb * T))
##            nqp = get_nqp(T, Delta)
#            zeta = (hbar * np.pi * fres)/(kb * T)
#            k2 = 1 / (2 * N0 * Delta) * (1+ np.sqrt(2 * Delta/(np.pi * kb * T)) * np.exp(-zeta) * sp.iv(0, zeta))
#            df_f0 = -alpha/2 * k2 * nqp
#            return (df_f0) * f0 + f0
#        
#        def MB_calc_Qibis(T, fres, Qi, alpha, Delta, Qi0):
#            nqp = 2 * N0 * np.sqrt(2 * np.pi * kb * T * Delta) * np.exp(-Delta/(kb * T))
##            nqp = get_nqp(T, Delta)
#            zeta = (hbar * np.pi * fres)/(kb * T)
#            k1 = 1 / (np.pi * N0 * Delta) * np.sqrt(2 * Delta/(np.pi * kb * T)) * np.sinh(zeta) * sp.kn(0, zeta)
#            d_1VSQi = alpha * k1 * nqp
#            return 1/((d_1VSQi) + 1/Qi0)
#        
        def minimize_f_for_psobis(params, *args):
            Delta, f0 = params
            alpha, T, fres = args
            return np.sum(abs(self.MB_calc_fbis(T, fres, alpha, Delta, f0) - fres)**2)
        
        def minimize_Qi_for_psobis(params, *args):
            Delta, Qi0 = params
            alpha, T, fres, Qi = args
            return np.sum(abs(self.MB_calc_Qibis(T, fres, Qi, alpha, Delta, Qi0) - Qi)**2)

        alphaFres_arr, DeltaFres_arr, errFres_arr, f0_arr, freslist_arr, Tlist_arr = [], [], [], [], [], []
        alphaQi_arr, DeltaQi_arr, errQi_arr, Qi0_arr, Qilist_arr = [], [], [], [], []

        for l in range(len(fres_data[0])):
            freslist = np.array([a[l] for a in fres_data])
            Tlist = np.array([a[l] for a in T_data])
            Qilist = np.array([a[l] for a in qi_data])
                      
            lbFres = np.array([1e-7, freslist[0]*1])
            ubFres = np.array([1e-2, freslist[0]*1.03])
            lbQi = np.array([1e-7, Qilist[0]*0.5])
            ubQi = np.array([1e-2, Qilist[0]*2])
            
            # Here is the PSO fit, the main values that can be changed are swarmsize and maxiter. Increasing swarmsize seems more useful to 
            # improve the accuracy than maxiter.
            xoptFres, foptFres = pso(minimize_f_for_psobis, lbFres, ubFres, ieqcons=[], f_ieqcons=None, args=(0.9, Tlist, freslist), kwargs={},
                swarmsize=2000, omega=0.5, phip=0.5, phig=0.5, maxiter=200, minstep=1e-8,
                minfunc=1e-8, debug=False)
            
            xoptQi, foptQi = pso(minimize_Qi_for_psobis, lbQi, ubQi, ieqcons=[], f_ieqcons=None, args=(0.9, Tlist, freslist, Qilist), kwargs={},
                swarmsize=2000, omega=0.5, phip=0.5, phig=0.5, maxiter=200, minstep=1e-8,
                minfunc=1e-8, debug=False)
            
#            alphaFres, DeltaFres, f0 = xoptFres
#            alphaQi, DeltaQi, Qi0 = xoptQi
            # Alpha is set to 0.9, the design value
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

        # The results are saved 
        filenameFres = MB_fit_dir + devname + "_MB_fres-fit_params.txt"
        fmt = ['%d','%.3f','%.3e','%.2f'] + ['%d' for a in np.ones(len(freslist_arr[0]))] + ['%.3f' for a in np.ones(len(Tlist_arr[0]))] 
        header = 'f0 [Hz]    Alpha    Delta [eV]     error [AU]           fres [Hz] (for each temp)            Temp [K]'
        np.savetxt(filenameFres, np.c_[f0_arr, alphaFres_arr, DeltaFres_arr, errFres_arr, freslist_arr, Tlist_arr], delimiter='\t', fmt=fmt, header=header)

        filenameQi = MB_fit_dir + devname + "_MB_Qi-fit_params.txt"
        fmt = ['%d','%.3f','%.3e','%.2f'] + ['%d' for a in np.ones(len(freslist_arr[0]))] + ['%.3f' for a in np.ones(len(Tlist_arr[0]))]  + ['%.3e' for a in np.ones(len(Qilist_arr[0]))] 
        header = 'Qi0         Alpha    Delta [eV]     error [AU]           fres [Hz] (for each temp)            Temp [K]          Qi'
        np.savetxt(filenameQi, np.c_[Qi0_arr, alphaQi_arr, DeltaQi_arr, errQi_arr, freslist_arr, Tlist_arr, Qilist_arr], delimiter='\t', fmt=fmt, header=header)
        self.Qilist_arr = Qilist_arr


    def plot_MBfit(self,var):
        filenameFres = MB_fit_dir + devname + "_MB_fres-fit_params.txt"
        filenameQi = MB_fit_dir + devname + "_MB_Qi-fit_params.txt"
        # After loading the saved files, the results are plotted
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
        Qilist_arr = [a[2*Nfres+4:] for a in MB_Qi_fit_params]
        
        fig10, ax10 = plt.subplots(figsize=(width_screen/mydpi, height_screen/mydpi), dpi=mydpi)
        fig10.canvas.set_window_title('df/f0 VS Temp') 
        plt.subplots_adjust(bottom=0.12, top=0.96, right=0.97, left=0.1)
        for i in range(len(freslist_arr)):
            dfvsf0_fit = (self.MB_calc_fbis(Tlist_arr[i], freslist_arr[i], alphaFres_arr[i], DeltaFres_arr[i], f0_arr[i])-f0_arr[i])/f0_arr[i]
            dfvsf0 = (freslist_arr[i] - f0_arr[i]) / f0_arr[i]
            Tnew = np.linspace(Tlist_arr[i][0],Tlist_arr[i][-1],100)
            dfvsf0_fit_smooth = interp1d(Tlist_arr[i], dfvsf0_fit, kind='cubic')
            color = plt.cm.tab20(i%20)
            plt.plot(Tnew*1e3,dfvsf0_fit_smooth(Tnew)*1e3,  color=color, linewidth = 2)
            plt.plot(Tlist_arr[i]*1e3, dfvsf0*1e3, 'o', markersize = 10, linewidth = 2, color=color)
            plt.plot(Tlist_arr[i]*1e3, dfvsf0_fit*1e3, 'x', markersize = 10, linewidth = 2, color=color)
        #    str_leg.append(r"$\alpha$ = %.3f, $f_0$ = %.4f MHz, $\Delta$ = %.3e, err = %.2f" %(alpha, f0*1e-6, Delta, fopt*1e-6))
        #plt.legend(str_leg, fontsize = 14, loc=3)
        plt.xlabel('Temperature [mK]', fontsize=38)
        plt.ylabel(r'$\delta f_{res}(T) / f_{res}(0) \quad \times 10^3$ ', fontsize=38)
        plt.xticks(color='k', size=34)
        plt.yticks(color='k', size=34)
        plt.grid()

        fig20, ax20 = plt.subplots(figsize=(width_screen/mydpi, height_screen/mydpi), dpi=mydpi)
        fig20.canvas.set_window_title('d(1/Qi) VS Temp') 
        plt.subplots_adjust(bottom=0.12, top=0.96, right=0.97, left=0.1)
        for i in range(len(Qilist_arr)):
            d_1vsQi_fit = 1/self.MB_calc_Qibis(Tlist_arr[i], freslist_arr[i], Qilist_arr[i], alphaQi_arr[i], DeltaQi_arr[i], Qi0_arr[i]) - 1/Qi0_arr[i]
            d_1vsQi = 1/Qilist_arr[i] - 1/Qi0_arr[i]
            Tnew = np.linspace(Tlist_arr[i][0],Tlist_arr[i][-1],100)
            d_1vsQi_fit_smooth = interp1d(Tlist_arr[i],d_1vsQi_fit, kind='cubic')
            color = plt.cm.tab20(i%20)
            plt.plot(Tnew*1e3,d_1vsQi_fit_smooth(Tnew)*1e5,  color=color, linewidth = 2)
            plt.plot(Tlist_arr[i]*1e3, d_1vsQi*1e5, 'o', markersize = 10, linewidth = 2, color=color)
            plt.plot(Tlist_arr[i]*1e3, d_1vsQi_fit*1e5, 'x', markersize = 10, linewidth = 2,  color=color)
        #    str_leg.append(r"$\alpha$ = %.3f, $f_0$ = %.4f MHz, $\Delta$ = %.3e, err = %.2f" %(alpha, f0*1e-6, Delta, fopt*1e-6))
        #plt.legend(str_leg, fontsize = 14, loc=3)
        plt.xlabel('Temperature [mK]', fontsize=38)
        plt.ylabel(r'$\delta \left[1/Qi(T)\right] \quad \times 10^5$ ', fontsize=38)
        plt.xticks(color='k', size=34)
        plt.yticks(color='k', size=34)
        plt.grid()
        
        alpha_hist, alpha_bins = np.histogram(alphaFres_arr, bins = 15)
        alpha_x = (alpha_bins[1:] + alpha_bins[0:-1])/2
        ind, err_arr_sorted = [], []
        for n in range(len(alpha_bins)-1):
            ind.append(np.where((alphaFres_arr>=alpha_bins[n]) & (alphaFres_arr<=alpha_bins[n+1]))[0])
            err_arr_sorted.append(np.mean(errFres_arr[ind[n]]))
        err_sorted_scaled = err_arr_sorted/max(err_arr_sorted)*max(alpha_hist)
        fig11, ax11 = plt.subplots(figsize=(width_screen/mydpi, height_screen/mydpi), dpi=mydpi)
        plt.subplots_adjust(bottom=0.12, top=0.96, right=0.97, left=0.1)
        width = (max(alpha_x) - min(alpha_x))/20
        plt.bar(alpha_x, alpha_hist, width = width)
        plt.bar(alpha_x+width/2, err_sorted_scaled, width = width/5)
        plt.xlabel(r'Kinetic Inductance Fraction $\alpha$  ($f_{res}$ fit)', fontsize=38)
        plt.ylabel('Number of resonances', fontsize=38)
        plt.xticks(color='k', size=34)
        plt.yticks(color='k', size=34)
        plt.legend([r'$\alpha$','Fit error [a.u.]'], fontsize = 28)
        plt.grid()
        
        Delta_hist, Delta_bins = np.histogram(DeltaFres_arr, bins = 15)
        Delta_x = (Delta_bins[1:] + Delta_bins[0:-1])/2
        ind, err_arr_sorted = [], []
        for n in range(len(Delta_bins)-1):
            ind.append(np.where((DeltaFres_arr>=Delta_bins[n]) & (DeltaFres_arr<=Delta_bins[n+1]))[0])
            err_arr_sorted.append(np.mean(errFres_arr[ind[n]]))
        err_sorted_scaled = err_arr_sorted/max(err_arr_sorted)*max(Delta_hist)
        fig12, ax12 = plt.subplots(figsize=(width_screen/mydpi, height_screen/mydpi), dpi=mydpi)
        plt.subplots_adjust(bottom=0.12, top=0.96, right=0.97, left=0.1)
        width = (max(Delta_x) - min(Delta_x))/20*1e3
        plt.bar(Delta_x*1e3, Delta_hist, width = width)
        plt.bar(Delta_x*1e3+width/2, err_sorted_scaled, width = width/5)
        plt.xlabel(r'Gap Energy $\Delta$ [meV]  ($f_{res}$ fit)', fontsize=38)
        plt.ylabel('Number of resonances', fontsize=38)
        plt.xticks(color='k', size=34)
        plt.yticks(color='k', size=34)
        plt.legend([r'$\Delta$','Fit error [a.u.]'], fontsize = 28)
        plt.grid()
        
        alphaQi_hist, alphaQi_bins = np.histogram(alphaQi_arr, bins = 15)
        alphaQi_x = (alphaQi_bins[1:] + alphaQi_bins[0:-1])/2
        ind, errQi_arr_sorted = [], []
        for n in range(len(alphaQi_bins)-1):
            ind.append(np.where((alphaQi_arr>=alphaQi_bins[n]) & (alphaQi_arr<=alphaQi_bins[n+1]))[0])
            errQi_arr_sorted.append(np.mean(errQi_arr[ind[n]]))
        errQi_sorted_scaled = errQi_arr_sorted/max(errQi_arr_sorted)*max(alphaQi_hist)
        fig21, ax21 = plt.subplots(figsize=(width_screen/mydpi, height_screen/mydpi), dpi=mydpi)
        plt.subplots_adjust(bottom=0.12, top=0.96, right=0.97, left=0.1)
        width = (max(alphaQi_x) - min(alphaQi_x))/20
        plt.bar(alphaQi_x, alphaQi_hist, width = width)
        plt.bar(alphaQi_x+width/2, errQi_sorted_scaled, width = width/5)
        plt.xlabel(r'Kinetic Inductance Fraction $\alpha$  ($Q_i$ fit)', fontsize=38)
        plt.ylabel('Number of resonances', fontsize=38)
        plt.xticks(color='k', size=34)
        plt.yticks(color='k', size=34)
        plt.legend([r'$\alpha$','Fit error [a.u.]'], fontsize = 28)
        plt.grid()

        max_deltaQi = 0.26e-3
        min_deltaQi = 0.20e-3
        bins = np.linspace(min_deltaQi, max_deltaQi, 15)
        DeltaQi_hist, DeltaQi_bins = np.histogram(DeltaQi_arr, bins = bins)
        DeltaQi_x = (DeltaQi_bins[1:] + DeltaQi_bins[0:-1])/2      
        ind, errQi_arr_sorted = [], []
        for n in range(len(DeltaQi_bins)-1):
            ind.append(np.where((DeltaQi_arr>=DeltaQi_bins[n]) & (DeltaQi_arr<=DeltaQi_bins[n+1]))[0])
            errQi_arr_sorted.append(np.mean(errQi_arr[ind[n]]))
        errQi_sorted_scaled = errQi_arr_sorted/np.nanmax(errQi_arr_sorted)*np.nanmax(DeltaQi_hist)
        fig22, ax22 = plt.subplots(figsize=(width_screen/mydpi, height_screen/mydpi), dpi=mydpi)
        plt.subplots_adjust(bottom=0.12, top=0.96, right=0.97, left=0.1)
        width = (max(DeltaQi_x*1e3) - min(DeltaQi_x*1e3))/20
        plt.bar(DeltaQi_x*1e3, DeltaQi_hist, width = width)
        plt.bar(DeltaQi_x*1e3+width/2, errQi_sorted_scaled, width = width/5)
        plt.xlabel(r'Gap Energy $\Delta$ [meV]  ($Q_i$ fit)', fontsize=38)
        plt.ylabel('Number of resonances', fontsize=38)
        plt.xticks(color='k', size=34)
        plt.yticks(color='k', size=34)
        plt.legend([r'$\Delta$','Fit error [a.u.]'], fontsize = 28)
        plt.grid()
               
        self.fig10 = fig10 
        self.fig11 = fig11 
        self.fig12 = fig12
        self.fig20 = fig20
        self.fig21 = fig21
        self.fig22 = fig22


    # Save Mattis Bardeen fit plots
    def save_MBfit_plots(self,var):
        fig10 = self.fig10 
        fig11 = self.fig11 
        fig12 = self.fig12
        fig20 = self.fig20
        fig21 = self.fig21
        fig22 = self.fig22
        
        MB_fres_plots_dir = plot_dir + 'MB_Fit_fres/'
        if not os.path.exists(MB_fres_plots_dir):
            print(MB_fres_plots_dir + ' directory has been created') 
            os.makedirs(MB_fres_plots_dir)
            
        MB_Qi_plots_dir = plot_dir + 'MB_Fit_Qi/'
        if not os.path.exists(MB_Qi_plots_dir):
            print(MB_Qi_plots_dir + ' directory has been created') 
            os.makedirs(MB_Qi_plots_dir)

        filename_fres = list([MB_fres_plots_dir + devname + "_dfvsf0_fit"])
        filename_fres.append(MB_fres_plots_dir + devname + "_hist_alpha_fresfit")
        filename_fres.append(MB_fres_plots_dir + devname + "_hist_delta_fresfit")
        filename_Qi = list([MB_Qi_plots_dir + devname + "_d_1vsQi_fit"])
        filename_Qi.append(MB_Qi_plots_dir + devname + "_hist_alpha_Qifit")
        filename_Qi.append(MB_Qi_plots_dir + devname + "_hist_delta_Qifit")        
        fig_num_fres = [fig10, fig11, fig12]
        fig_num_Qi = [fig20, fig21, fig22]
        for t in range(len(filename_fres)):
            fig_num_fres[t].savefig(filename_fres[t] + '.svg', dpi=fig.dpi, bbox_inches = 'tight')
            fig_num_fres[t].savefig(filename_fres[t] + '.png', dpi=plt.gcf().dpi, bbox_inches = 'tight')
            fig_num_fres[t].savefig(filename_fres[t] + '.pdf', dpi=fig.dpi, bbox_inches = 'tight')
            plt.close(fig_num_fres[t])
        for t in range(len(filename_Qi)):
            fig_num_Qi[t].savefig(filename_Qi[t] + '.svg', dpi=fig.dpi, bbox_inches = 'tight')
            fig_num_Qi[t].savefig(filename_Qi[t] + '.png', dpi=plt.gcf().dpi, bbox_inches = 'tight')
            fig_num_Qi[t].savefig(filename_Qi[t] + '.pdf', dpi=fig.dpi, bbox_inches = 'tight')
            plt.close(fig_num_Qi[t])
        
    
    def pdf_fit_reso(self, var):
        self.read_clean_files()
        res_data = self.res_data
        if glob.glob(device_dir + 'Figures/TiN_reso/Reports/Fit_reso*/'):
            pdf_dir = glob.glob(device_dir + 'Figures/TiN_reso/Reports/Fit_reso*/')[0]
        else:
            root = tk.Tk()
            root.withdraw()
            pdf_dir = filedialog.askdirectory(title = "Select the Fit params directory (ex. Fit_params_Be170227bl)") # Open a dialog window to select the file
    
        N = len(res_data)
        for i in range(N):
            N_files = len(res_data[i])
            str_name = device_dir.split('/')[-2]
            pwr = res_data[i][0].pwr # power in dBm
            str_pwr = str(int(pwr))
            temp = res_data[i][0].temp # Temp in K
            str_temp = str(int(temp*1000))
            
            filename_report = pdf_dir + "fit_report_" + str_name + '_' + str_temp + 'mK_' + str_pwr + 'dBm'
            Report_pdf = PdfPages(filename_report + '.pdf')
        
            for k in range(N_files):
                res = res_data[i][k]
                freq= res.freq
                Iraw = res.I
                Qraw = res.Q
                Ifit = res.resultI
                Qfit = res.resultQ
                
                Mag = np.sqrt(Iraw**2 + Qraw**2)
                fitMag = np.sqrt(Ifit**2 + Qfit**2)
                
                resoNum = res.index
                fres0 = res.lmfit_result.params['f0'].value
                Qi = res.lmfit_result.params['qi'].value
                Qc = res.lmfit_result.params['qc'].value
                fit_ok = res.fit_ok
#                MB_fit_ok = res.MB_fit_ok
                
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
                
                str_params = "VNA Power = " + str_pwr + " dBm \n" + "Input Power = %d-%d = %d" %(pwr, atten, (pwr-atten)) + " dBm \n" + "Temp = " + str_temp + " mK"
                fitwords = "Res Num : " + "%d" %resoNum + " \n" + "$f_{res}$ = " + "%.6f" %(fres0*1e-6) + " MHz \n" + "$Q_{c}$ = " + "%.2E" %Qc + "\n" + "$Q_{i}$ = " + "%.2E" %Qi #+ "\n" + "$a$ = " + "%.5f" %a[i].real + " + " + "%.5f" %a[i].imag + "j\n" + "$\phi_{0}$ = " + "%.5f" %phi[i] + " \n" + r"$\tau$ = " + "%.5f" %tau[i] + " \n"
                
                plt.figtext(0.35, 0.96, str_name + ' (TiN)', fontsize = 16)
                plt.figtext(0.1, 0.83, fitwords, fontsize = 14)
                plt.figtext(0.5, 0.83, str_params, fontsize = 14)
                
                if fit_ok: 
                    plt.figtext(0.42, 0.935, 'Fit OK', fontsize = 14, color = 'green', weight = 'bold')
                else: 
                    plt.figtext(0.29, 0.935, 'Bad fit, resonance discarted', fontsize = 14, color = 'red', weight = 'bold')
                
                Report_pdf.savefig()
                plt.close()
            Report_pdf.close()
#                str_pdf_reduce = 'qpdf --linearize ' +  filename_report + '.pdf ' + filename_report + '_LowSize.pdf'
#                os.system(str_pdf_reduce)


    # Read filenameFres and filenameQi to extract the params of the MB fit
    def extract_fit_MB_params(self):    
        filenameFres = MB_fit_dir + devname + "_MB_fres-fit_params.txt"
        filenameQi = MB_fit_dir + devname + "_MB_Qi-fit_params.txt"
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
        Qilist_arr = [a[2*Nfres+4:] for a in MB_Qi_fit_params]
        dfvsf0_fit_arr, dfvsf0_arr, Tnew_arr, dfvsf0_fit_smooth_arr = [], [], [], []
        
        for i in range(len(freslist_arr)):
            dfvsf0_fit = (self.MB_calc_fbis(Tlist_arr[i], freslist_arr[i], alphaFres_arr[i], DeltaFres_arr[i], f0_arr[i])-f0_arr[i])/f0_arr[i]
            dfvsf0 = (freslist_arr[i] - f0_arr[i]) / f0_arr[i]
            Tnew = np.linspace(Tlist_arr[i][0],Tlist_arr[i][-1],100)
            dfvsf0_fit_smooth = interp1d(Tlist_arr[i], dfvsf0_fit, kind='cubic')
            dfvsf0_fit_arr.append(dfvsf0_fit)
            dfvsf0_arr.append(dfvsf0)  
            Tnew_arr.append(Tnew)
            dfvsf0_fit_smooth_arr.append(dfvsf0_fit_smooth)
        
        d_1vsQi_fit_arr, d_1vsQi_arr, d_1vsQi_fit_smooth_arr = [], [], []
        for i in range(len(Qilist_arr)):
            d_1vsQi_fit = 1/self.MB_calc_Qibis(Tlist_arr[i], freslist_arr[i], Qilist_arr[i], alphaQi_arr[i], DeltaQi_arr[i], Qi0_arr[i]) - 1/Qi0_arr[i]
            d_1vsQi = 1/Qilist_arr[i] - 1/Qi0_arr[i]
            Tnew = np.linspace(Tlist_arr[i][0],Tlist_arr[i][-1],100)
            d_1vsQi_fit_smooth = interp1d(Tlist_arr[i],d_1vsQi_fit, kind='cubic')
            d_1vsQi_fit_arr.append(d_1vsQi_fit)
            d_1vsQi_arr.append(d_1vsQi)  
            d_1vsQi_fit_smooth_arr.append(d_1vsQi_fit_smooth)
        
        self.dfvsf0_fit_arr = dfvsf0_fit_arr
        self.dfvsf0_arr = dfvsf0_arr
        self.Tnew_arr = Tnew_arr
        self.dfvsf0_fit_smooth_arr = dfvsf0_fit_smooth_arr
        self.d_1vsQi_fit_arr = d_1vsQi_fit_arr
        self.d_1vsQi_arr = d_1vsQi_arr
        self.d_1vsQi_fit_smooth_arr = d_1vsQi_fit_smooth_arr
        self.Tlist_arr = Tlist_arr
        self.freslist_arr = freslist_arr
        self.Qilist_arr = Qilist_arr
        self.Qi0_arr = Qi0_arr
        self.f0_arr = f0_arr
        self.alphaFres_arr = alphaFres_arr
        self.DeltaFres_arr = DeltaFres_arr
        self.alphaQi_arr = alphaQi_arr
        self.DeltaQi_arr = DeltaQi_arr
        self.errFres_arr = errFres_arr
        self.errQi_arr = errQi_arr
        
        
    # Create a report of the MB fit, with the plots df/f0 vs. T, 1/Qi-1/Qi0 vs. T and the result params of the MB fit
    def pdf_fit_MB(self, var):
        if glob.glob(device_dir + 'Figures/TiN_reso/Reports/Fit_MB*/'):
            pdf_dir = glob.glob(device_dir + 'Figures/TiN_reso/Reports/Fit_MB*/')[0]
        else:
            root = tk.Tk()
            root.withdraw()
            pdf_dir = filedialog.askdirectory(title = "Select the Fit params directory (ex. Fit_params_Be170227bl)") # Open a dialog window to select the file
            
        self.extract_fit_MB_params()
        dfvsf0_fit_arr = self.dfvsf0_fit_arr 
        dfvsf0_arr = self.dfvsf0_arr
        Tnew_arr = self.Tnew_arr
        dfvsf0_fit_smooth_arr = self.dfvsf0_fit_smooth_arr
        d_1vsQi_fit_arr = self.d_1vsQi_fit_arr
        d_1vsQi_arr = self.d_1vsQi_arr
        d_1vsQi_fit_smooth_arr = self.d_1vsQi_fit_smooth_arr
        Tlist_arr = self.Tlist_arr
        Qilist_arr = self.Qilist_arr
        Qi0_arr = self.Qi0_arr
        freslist_arr = self.freslist_arr
        f0_arr = self.f0_arr
        alphaFres_arr = self.alphaFres_arr
        DeltaFres_arr = self.DeltaFres_arr
        alphaQi_arr = self.alphaQi_arr
        DeltaQi_arr = self.DeltaQi_arr

        str_name = device_dir.split('/')[-2]
#        pwr = res_data[i][0].pwr # power in dBm
#        str_pwr = str(int(pwr))
        
        filename_report = pdf_dir + "MB_fit_report_" + str_name #+ '_' + str_pwr + 'dBm'
        Report_pdf = PdfPages(filename_report + '.pdf')
               
        for i in range(len(alphaFres_arr)):
            
            fig1, ax1 = plt.subplots(figsize=(8.5, 11))
            plt.subplots_adjust(bottom=0.08, top=0.8, right=0.95, left=0.11)
            plt.subplots_adjust(wspace=0.42, hspace=0.42)
        
            plt.subplot(2,2,1)
            plt.plot(Tnew_arr[i]*1e3,dfvsf0_fit_smooth_arr[i](Tnew_arr[i])*1e3,  color='red')
            plt.plot(Tlist_arr[i]*1e3, dfvsf0_arr[i]*1e3, 'o', markersize = 6,  color='blue')
            plt.plot(Tlist_arr[i]*1e3, dfvsf0_fit_arr[i]*1e3, 'x', markersize = 8,  color='red')
            plt.xlabel('Temperature [mK]', fontsize=12)
            plt.ylabel(r'[f(T) - f(0)] / f(0)    $\times 10^3$ ', fontsize=12)
            plt.xticks(color='k', size=10)
            plt.yticks(color='k', size=10)
            plt.grid()
    
            plt.subplot(2,2,2)
            plt.plot(Tnew_arr[i]*1e3,d_1vsQi_fit_smooth_arr[i](Tnew_arr[i])*1e6,  color='red')
            plt.plot(Tlist_arr[i]*1e3, d_1vsQi_arr[i]*1e6, 'o', markersize = 6,  color='blue')
            plt.plot(Tlist_arr[i]*1e3, d_1vsQi_fit_arr[i]*1e6, 'x', markersize = 8,  color='red')
            plt.xlabel('Temperature [mK]', fontsize=12)
            plt.ylabel(r'[1/Qi(T) - 1/Qi(0)]   $\times 10^6$ ', fontsize=12)
            plt.xticks(color='k', size=10)
            plt.yticks(color='k', size=10)
            plt.grid()
            
            plt.subplot(2,2,3)
            plt.plot(Tlist_arr[i]*1e3, freslist_arr[i]*1e-6, 'o', markersize = 6,  color='blue', label = 'f(T)')
    #        freslist_interp = interp1d(Tlist_arr[i], freslist_arr[i], kind='cubic')
    #        plt.plot(Tnew_arr[i]*1e3, freslist_interp(Tnew_arr[i])*1e-6,  color='blue')
            plt.plot(Tlist_arr[i]*1e3, np.ones(len(Tlist_arr[i]))*f0_arr[i]*1e-6, color='black', label = 'f(0) = %.4f MHz' %(f0_arr[i]*1e-6))
            plt.xlabel('Temperature [mK]', fontsize=12)
            plt.ylabel('Frequency [MHz]', fontsize=12)
            plt.xticks(color='k', size=10)
            plt.yticks(color='k', size=10)
            plt.legend(fontsize = 10)
            plt.grid()
    
            plt.subplot(2,2,4)
            plt.plot(Tlist_arr[i]*1e3, Qilist_arr[i], 'o', markersize = 6, color='blue', label = 'Qi(T)')
    #        plt.plot(Tlist_arr[i]*1e3, Qclist_arr[i], 'o', markersize = 6, color='red', label = 'Qc')
            plt.plot(Tlist_arr[i]*1e3, np.ones(len(Tlist_arr[i]))*Qi0_arr[i], color='black', label = 'Qi(0) = %.3e ' %(Qi0_arr[i]))
            plt.xlabel('Temperature [mK]', fontsize=12)
            plt.ylabel('Qi ', fontsize=12)
            plt.xticks(color='k', size=10)
            plt.yticks(color='k', size=10)
            plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
            plt.legend(fontsize = 10)
            plt.grid()   
            
            str_MB_fit_fres = r"$\alpha$ = %.3f" %(alphaFres_arr[i]) + "\n" + r"$\Delta$ = %.3f" %(DeltaFres_arr[i]*1e3) + " meV \n" + "f(T=0K) = %.4f" %(f0_arr[i]*1e-6) + " MHz"  
            str_MB_fit_Qi = r"$\alpha$ = %.3f" %(alphaQi_arr[i]) + "\n" + r"$\Delta$ = %.3f" %(DeltaQi_arr[i]*1e3) + " meV \n" + "Qi(T=0K) = %.3e" %(Qi0_arr[i])  
    
            plt.figtext(0.35, 0.96, str_name + ' (TiN)', fontsize = 16)
            plt.figtext(0.11, 0.82, str_MB_fit_fres, fontsize = 12)
            plt.figtext(0.6, 0.82, str_MB_fit_Qi, fontsize = 12)

            Report_pdf.savefig()
            plt.close()
        Report_pdf.close()


callback = analyze_data()
s_move_win.on_changed(callback.move_win)
s_win_width.on_changed(callback.change_win_width)
b_export.on_clicked(callback.export_cleaned_params)
b_plot_Qi_Qc.on_clicked(callback.plot_Qi_Qc)
b_save_Qi_Qc.on_clicked(callback.save_Qi_Qc)
b_fit_Qi_f0.on_clicked(callback.fit_Qi_f0)
b_plot_MBfit.on_clicked(callback.plot_MBfit)
b_save_MBfit.on_clicked(callback.save_MBfit_plots)
b_fitres_pdf.on_clicked(callback.pdf_fit_reso)
b_fitMB_pdf.on_clicked(callback.pdf_fit_MB)

fig.canvas.mpl_connect('key_press_event', callback.press)
fig.canvas.mpl_connect('pick_event', callback.select_data)
plt.show()   





