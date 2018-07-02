#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 15:00:28 2018

@author: Fabien Defrance
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import peakutils
import tkinter as tk
from tkinter import filedialog
from matplotlib.widgets import Slider, Button
import os
import sys
#sys.path.append('/home/ubadmin/Documents/Caltech/local_python_library/')
sys.path.append('C:/Users/DELL E7270/Documents/Documents/Travail/Caltech/local_python_library/')
import scraps as scr
from functions_for_KIDs_data import create_folder_tree
import pickle

#
###################### Variables ####################################
#
polyOrder = 5 # Order of the polynomial to fit the baseline (increase if the fit fails)
fWindow = 2e5 # Frequency window used to select the different peaks (Hz)
thresinidB = -1 # Initial threshold for peaks detection (dB)
minthresdB = -10 # Minimum threshold for peaks detection (dB)
min_dist_ini = 10 # Initial minimum distance between the peaks, for peak detection (kHz)
# LP filter parameters, usually only FC needs to be adjusted
FC_ini = 0.05  # cutoff frequency (arbitrary)
N = 101        # number of filter taps
a = 1          # filter denominator
select_file = 1
#today = datetime.datetime.now().strftime("%Y%m%d")
delay = round((N-1)/2) # The LP filter shifts the frequency so we have to correct this shift 
offset = delay*5 # Remove some points at the beginning of the filtered spectrum because the filter distords the signal at the beginning of the spectrum
# Get the size of the screen, useful to plot full screen plots
root = tk.Tk()
root.withdraw()
width_screen = root.winfo_screenwidth()
height_screen = root.winfo_screenheight()
mydpi = 100 # Resolution for displayed plots
device_dir = 'Test-device/'


# Create the figure
fig = plt.figure()
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.subplots_adjust(bottom=0.04, top=0.96, right=0.96, left=0.08)
# Upper plot with original data (blue) and filtered data (red)
ax1 = fig.add_subplot(211)
plt.subplots_adjust(bottom=0.3)
plot0, = ax1.plot([], [], lw=2, color='blue')
plot1, = ax1.plot([], [], lw=2, color='red')
plt.ylabel('Power (dB)', fontsize=14)
plt.xticks(color='k', size=14)
plt.yticks(color='k', size=14)
        
# Lower plot with baseline corrected filtered data (green), detected resonances (black dots) and resonances windows limits (red vertical lines)
# The dashed grey line corresponds to the threshold for peaks detection                  
ax2 = fig.add_subplot(212)
plot2, = ax2.plot([], [], lw=2, color='green')
plot3, = ax2.plot([], [],'o', color='black')
plot4, = ax2.plot([], [],'--', color='grey')
plot5, = ax2.plot([], [],'|', color='red', markersize = 20)
plt.ylabel('Power (dB)', fontsize=14)
plt.xlabel('Frequency (MHz)', fontsize=14)
plt.xticks(color='k', size=14)
plt.yticks(color='k', size=14)
        
#
###################### Widgets ####################################
#
# Definition of the widgets' size and position (3 sliders and 6 buttons (for now))
ax_color = 'lightgoldenrodyellow' # Color of the sliders
ax_freq = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=ax_color)
ax_thres = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=ax_color)
ax_min_dist = plt.axes([0.25, 0.2, 0.65, 0.03], facecolor=ax_color)
ax_baseline = plt.axes([0.2, 0.025, 0.1, 0.04])
ax_isolate_peaks = plt.axes([0.35, 0.025, 0.1, 0.04])
ax_save_peaks = plt.axes([0.5, 0.025, 0.1, 0.04])
ax_fit_res = plt.axes([0.65, 0.025, 0.1, 0.04])
ax_export_params = plt.axes([0.8, 0.025, 0.1, 0.04])
ax_load_file = plt.axes([0.05, 0.025, 0.1, 0.04])

# Definition of the widgets' values and name (3 sliders and 6 buttons (for now))
s_freq = Slider(ax_freq, 'LP filter', 0.010, 0.100, valinit = FC_ini)
s_thresdB = Slider(ax_thres, 'Threshold (dB)', minthresdB, 0.0, valinit = thresinidB)
s_min_dist = Slider(ax_min_dist, 'Min peak distance (kHz)', 1, 100, valinit = min_dist_ini)
b_baseline = Button(ax_baseline, 'Baseline', color=ax_color, hovercolor='0.975')
b_isolate_peaks = Button(ax_isolate_peaks, 'Isolate Peaks', color=ax_color, hovercolor='0.975')
b_save_peaks = Button(ax_save_peaks, 'Save Peaks', color=ax_color, hovercolor='0.975')
b_fit_res = Button(ax_fit_res, 'Fit Res', color=ax_color, hovercolor='0.975')
b_export_params = Button(ax_export_params, 'Export Fit Params', color=ax_color, hovercolor='0.975')
b_load_file = Button(ax_load_file, 'Load New File', color=ax_color, hovercolor='0.975')

# Definition of the widgets' fontsize (3 sliders and 6 buttons (for now))
s_freq.label.set_fontsize(14)
s_freq.valtext.set_fontsize(14)
s_min_dist.label.set_fontsize(14)
s_min_dist.valtext.set_fontsize(14)
s_thresdB.label.set_fontsize(14)
s_thresdB.valtext.set_fontsize(14)
b_baseline.label.set_fontsize(14)
b_isolate_peaks.label.set_fontsize(14)
b_save_peaks.label.set_fontsize(14)
b_fit_res.label.set_fontsize(14)
b_export_params.label.set_fontsize(14)
b_load_file.label.set_fontsize(14)

#
###################### Assign functions to Widgets ####################################
#
class filtering(object):
    
#    def __init__(self):
        
    
    def load_new_file(self,var):
        # Select the file you want to load and plot
        root = tk.Tk()
        root.withdraw()
        filePath = filedialog.askopenfilename() # Open a dialog window to select the file
        fileData = np.loadtxt(filePath)
        freqData = fileData[:,0]
        IData = fileData[:,1]
        QData = fileData[:,2]
        MagData = np.sqrt(IData**2+QData**2)
        
        fileName = filePath.split('/')[-1]
        str_pwr = fileName.split('_')[1] # power in dBm
        str_temp = fileName.split('_')[4][:-4] # Temp in K
        str_name = fileName.split('_')[0] # Name
        
        # Main directory of the device
#        device_dir = os.path.abspath(filePath+'/../../../../..') + '/'
        create_folder_tree(device_dir) 
            
        plot0.set_data(freqData/1e6, 20*np.log10(MagData))
        ax1.relim()
        ax1.autoscale_view()
        fig.canvas.draw_idle()
        
        # Define different directories used to save the data, once filtered, split and fitted
        TiN_dir = device_dir + 'Data/TiN_reso/'
        raw_data_dir = TiN_dir + 'Raw_data/'
        split_data_dir = TiN_dir + 'Split_data/'
        Fit_res_dir = TiN_dir + 'Fit_reso_params/'
        obj_save_dir = TiN_dir + 'Fit_reso_objects/'
        temp_dir = split_data_dir + str_name + '_' + str_pwr + 'dBm_' + str(int(float(str_temp)*1000))  + 'mK/'
            
        if not os.path.exists(raw_data_dir):
            print(raw_data_dir + ' directory has been created') 
            os.makedirs(raw_data_dir)
            
        if not os.path.exists(split_data_dir):
            print(split_data_dir + ' directory has been created') 
            os.makedirs(split_data_dir)
            
        if not os.path.exists(Fit_res_dir):
            print(Fit_res_dir + ' directory has been created') 
            os.makedirs(Fit_res_dir)
            
        if not os.path.exists(obj_save_dir):
            print(obj_save_dir + ' directory has been created') 
            os.makedirs(obj_save_dir)
                       
        if not os.path.exists(temp_dir):
            print(temp_dir + ' directory has been created') 
            os.makedirs(temp_dir)
        
        self.IData = IData
        self.QData = QData
        self.freqData = freqData
        self.MagData = MagData
#        self.peaks_files_dir = peaks_files_dir
#        self.fit_params_dir = fit_params_dir
        self.str_pwr = str_pwr
        self.str_temp = str_temp
        self.str_name = str_name
#        self.peaks_files_fit_dir = peaks_files_fit_dir
        self.TiN_dir = TiN_dir
        self.raw_data_dir = raw_data_dir
        self.split_data_dir = split_data_dir
        self.Fit_res_dir = Fit_res_dir
        self.obj_save_dir = obj_save_dir
        self.temp_dir = temp_dir

        
    
    def update_filt(self, val): # Low pass filter the data and plot in red on the plot
        QData = self.QData
        IData = self.IData  
        freqData = self.freqData   
        N_data = len(freqData)                          
        FC = s_freq.val
        b = signal.firwin(N, cutoff=FC, window='hamming')    # filter numerator
        yQ = signal.lfilter(b, a, QData)                         # filtered output
        yI = signal.lfilter(b, a, IData)                         # filtered output
        yQ = yQ[delay+offset:N_data-1]
        yI = yI[delay+offset:N_data-1]
        freqSmooth = freqData[0+offset:N_data-1-delay] 
        Mag = np.sqrt(yQ**2+yI**2)
        plot1.set_data(freqSmooth/1e6, 20*np.log10(Mag))
        ax1.relim()
        ax1.autoscale_view() 
        fig.canvas.draw_idle()
        fig.canvas.draw_idle()
        self.Mag = Mag
        self.yI = yI
        self.yQ = yQ
        self.freqSmooth = freqSmooth

    
    def update_baseline(self, val): # Remove the baseline from the data, plot the data without the baseline
        Mag = self.Mag
        freqSmooth = self.freqSmooth
        base = peakutils.baseline(1/Mag, polyOrder)
        plot2.set_data(freqSmooth/1e6, 20*np.log10(Mag*base))
        ax2.relim()
        ax2.autoscale_view() 
        fig.canvas.draw_idle()
        self.base = base
    
    
    def find_peaks(self, val): # Find the peaks with are below the threshold
        thresdB = s_thresdB.val
        min_dist = s_min_dist.val
        base = self.base
        Mag = self.Mag
        freqSmooth = self.freqSmooth
#        thres = (1 - 10**(thresdB/20)) / (1 - min(Mag*base))
        thres = (1 - 10**(thresdB/20)) / (1 - 10**(minthresdB/20)) # sets the minimum threshold to -10dB
#        thres = (thresdB-1) / (-10) # sets the minimum threshold to -10dB
#        thresdB = 20*np.log10(1 - thres*(1 - min(Mag*base)))
        freq_pitch = (freqSmooth[1] - freqSmooth[0])*1e-3
        indexes = peakutils.indexes(-(Mag*base), thres=thres, min_dist=min_dist/freq_pitch)          
        plot3.set_data(freqSmooth[indexes]/1e6, 20*np.log10(Mag[indexes]*base[indexes]))
        plot4.set_data(freqSmooth[[0,-1]]/1e6, [thresdB, thresdB])
        N_res = len(indexes)
        props = dict(boxstyle='round', facecolor='wheat', alpha=1)
        ax2.text(0.05, 0.1, 'Number of reso: ' + str(N_res), transform=ax2.transAxes, fontsize=14, verticalalignment='bottom', bbox=props)
        ax2.relim()
        ax2.autoscale_view()
        fig.canvas.draw_idle()
        self.indexes = indexes
        
        
    def isolate_peaks(self, fwindow): # Isolate the peaks. The width of the window around the peaks is either fwindow, or the highest point between two peaks
        indexes = self.indexes
        freqSmooth = self.freqSmooth
        Mag = self.Mag
        base = self.base
        # find the low and high boundaries of the windows around the peaks
        split_freq_max = freqSmooth[indexes] + fWindow/2
        split_freq_min = freqSmooth[indexes] - fWindow/2
        indWin = np.zeros((len(indexes)*2), dtype=int)
        
        for k in range(len(indexes)):
            # Find the indexes of the boundaries
            indWin[2*k] = int(np.argmin(abs(split_freq_min[k]-freqSmooth)))
            indWin[2*k+1] = int(np.argmin(abs(split_freq_max[k]-freqSmooth)))
            if k == 0 & indWin[2*k]<0: # If the boundary is inferior to the first data point, the data point is the lowest boundary
                indWin[2*k] = 0
            elif (k>0) & (indWin[2*k] < indWin[2*k-1]): # If the low boundary of one resonance is higher than the high boundary of the previous resonance, 
                #Both boundaries are merged and correspond to the highest magnitude point between the resonances 
                indWin[2*k] = np.argmax(Mag[indexes[k-1]:indexes[k]]) + indexes[k-1]
                indWin[2*k-1] = indWin[2*k]
                win_LowLim = max([indWin[2*k-2], indexes[k-1] - 3*(indWin[2*k-1]-indexes[k-1])])
                indWin[2*k-2] = np.argmax(Mag[win_LowLim:indexes[k-1]]) + win_LowLim
                win_HighLim = min([indWin[2*k+1], indexes[k] + 3*(indexes[k]-indWin[2*k])])
                indWin[2*k+1] = np.argmax(Mag[indexes[k]:win_HighLim]) + indexes[k]
        plot5.set_data(freqSmooth[indWin]/1e6, 20*np.log10(Mag[indWin] * base[indWin]))
        fig.canvas.draw_idle()
        self.indWin = indWin


    def save_peaks(self, var): # Save the split peaks data as individual files 
        temp_dir = self.temp_dir
        indWin = self.indWin
        freqData = self.freqData
        IData = self.IData
        QData = self.QData
#        str_name = self.str_name
#        peaks_files_dir = self.peaks_files_dir
#        split_data_dir = self.split_data_dir
        str_pwr = self.str_pwr
        str_temp = self.str_temp
        indWin2 = indWin + int(offset)
        N_files = int(len(indWin2)/2)

        for i in range(N_files):
            res_filename = temp_dir + 'RES-' + '%04d' %(i) + '_' + str_pwr + '_DBM_TEMP_' + str_temp + '.txt'
            freq = freqData[indWin2[2*i]:indWin2[2*i+1]+1]
            I = IData[indWin2[2*i]:indWin2[2*i+1]+1]
            Q = QData[indWin2[2*i]:indWin2[2*i+1]+1]
            np.savetxt(res_filename, np.c_[freq, I, Q], delimiter='\t', fmt='%.5f', header=' Freq (Hz)       I        Q')
            
            
    def fit_res(self, var): # Look at the resonances fitted with SCRAPS and choose the ones you want to keep or discard
        temp_dir = self.temp_dir
        N_files = len(os.listdir(temp_dir)) 
        L_plots = int(np.ceil(np.sqrt(N_files)))
        qi = np.zeros(N_files)
        f0 = np.zeros(N_files)
        qc = np.zeros(N_files)
        res_list = []
        
        fig1, ax3 = plt.subplots(L_plots,L_plots,figsize=(width_screen/mydpi, height_screen/mydpi), dpi=mydpi)
        for i in range(N_files):
            res_filename = temp_dir + os.listdir(temp_dir)[i]
            fileDataDict = scr.process_file(res_filename)
            # Fit the resonances
            res = scr.makeResFromData(fileDataDict)
            res.load_params(scr.cmplxIQ_params)
            res.do_lmfit(scr.cmplxIQ_fit)
            res.fit_ok = 1 #add a variable to the object to tell if the resonance is correctly fitted
            res.index = i #Define the index of the resonance
            res_list.append(res)
            f0[i] = res.lmfit_result.params['f0'].value
            qi[i] = res.lmfit_result.params['qi'].value
            qc[i] = res.lmfit_result.params['qc'].value
            # Define the position of the plots and plot one plot for each resonance
            xpos = int(np.floor(i/L_plots) )
            ypos = int(i%L_plots)
            ax3[xpos,ypos].plot(res.freq, res.mag)
            ax3[xpos,ypos].plot(res.freq, res.resultMag)
            ax3[xpos,ypos].set_title('Res%d: %.2fMHz' %(i, f0[i]*1e-6))
            ax3[xpos,ypos].set_xticks([])
            ax3[xpos,ypos].set_yticks([])
            
        for i in range(N_files, L_plots**2):
            xpos = int(np.floor(i/L_plots) )
            ypos = int(i%L_plots)
            ax3[xpos, ypos].remove()
                       
        plt.subplots_adjust(bottom=0.02, top=0.96, right=0.98, left=0.02)
        plt.subplots_adjust(wspace=0.1, hspace=0.35)
        plt.show()
        
        def discard_reso(event): #When clicked on a resonance, the background turns red and the resonance is discarted
            size = fig1.get_size_inches()*fig.dpi
            subplots_lims = [np.linspace(0.02*size[0], 0.98*size[0], 2*L_plots+1)[1::2], np.linspace(0.02*size[1], 0.96*size[1], 2*L_plots+1)[1::2]]
            xpos = np.argmin(abs(event.x - subplots_lims[0]))
            ypos = np.argmin(abs(event.y - subplots_lims[1]))
            plotNum = (L_plots-ypos-1)*L_plots + xpos
            if res_list[plotNum].fit_ok == 1:
                ax3[L_plots-ypos-1, xpos].set_facecolor('red')
                fig1.canvas.draw()
                res_list[plotNum].fit_ok = 0
            else:
                ax3[L_plots-ypos-1, xpos].set_facecolor('white')
                fig1.canvas.draw()
                res_list[plotNum].fit_ok = 1
        fig1.canvas.mpl_connect('button_press_event', discard_reso)

        self.res_list = res_list
        
        
    def export_fit_params(self,var): #Export the fitted params of all the resonances, both as text files and pickles
        res_list = self.res_list
        Fit_res_dir = self.Fit_res_dir
        obj_save_dir = self.obj_save_dir
        str_pwr = self.str_pwr
        str_temp = self.str_temp

        fit_ok = [a.fit_ok for a in res_list]
        ind = [a.index for a in res_list]
        fres = [a.lmfit_result.params['f0'].value for a in res_list]
        qi = [a.lmfit_result.params['qi'].value for a in res_list]
        qc = [a.lmfit_result.params['qc'].value for a in res_list]

        Output_params_file = Fit_res_dir + 'fit-params_' + str_pwr + '_DBM_TEMP_' + str_temp + '.txt'
        np.savetxt(Output_params_file, np.c_[ind, fit_ok, fres, qi, qc], delimiter='\t', fmt='%.0f', header='Index  Fit_ok   F_res(Hz)     Qi     Qc')
        res_list_save_file = obj_save_dir + 'pickle_' + str_pwr + '_DBM_TEMP_' + str_temp + '.p'
        pickle.dump(res_list, open(res_list_save_file,'wb'))
        

# Define what function needs to be called by each widget
callback = filtering()
s_freq.on_changed(callback.update_filt)
b_baseline.on_clicked(callback.update_baseline)
s_thresdB.on_changed(callback.find_peaks)
s_min_dist.on_changed(callback.find_peaks)
b_isolate_peaks.on_clicked(callback.isolate_peaks)
b_save_peaks.on_clicked(callback.save_peaks)
b_fit_res.on_clicked(callback.fit_res)
b_export_params.on_clicked(callback.export_fit_params)
b_load_file.on_clicked(callback.load_new_file)