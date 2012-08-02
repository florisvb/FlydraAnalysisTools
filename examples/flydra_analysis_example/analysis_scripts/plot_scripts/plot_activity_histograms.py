#!/usr/bin/env python
import sys
sys.path.append('../')
import flydra_analysis_tools as fat
from flydra_analysis_tools import floris_plot_lib as fpl
fad = fat.flydra_analysis_dataset
dac = fat.dataset_analysis_core
fap = fat.flydra_analysis_plot
tac = fat.trajectory_analysis_core

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import analysis_configuration


################# get trajectory keys #################

def get_keys(dataset):
    keys = dataset.trajecs.keys()
    return keys

################# plotting functions #######################

def plot_activity_histogram(dataset, save_figure_path=''):
    keys = get_keys(dataset)
    print 'number of keys: ', len(keys)
    
    local_time = np.zeros(len(keys))
    for i, key in enumerate(keys):
        trajec = dataset.trajecs[key]
        local_time[i] = trajec.timestamp_local_float
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    nbins = 24 # note: if show_smoothed=True with default butter filter, nbins needs to be > ~15 
    bins = np.linspace(0,24,nbins)
    
    fpl.histogram(ax, [local_time], bins=bins, bin_width_ratio=0.8, colors=['green'], edgecolor='none', bar_alpha=1, curve_fill_alpha=0.4, curve_line_alpha=0, curve_butter_filter=[3,0.3], return_vals=False, show_smoothed=True, normed=True, normed_occurences=False, bootstrap_std=False, exponential_histogram=False)
    
    xticks = np.linspace(bins[0], bins[-1], 5, endpoint=True)
    fpl.adjust_spines(ax, ['left', 'bottom'], xticks=xticks)
    ax.set_xlabel('Time of day, hours')
    ax.set_ylabel('Occurences, normalized')
    ax.set_title('Activity, as measured by number of trajectories')

    figname = save_figure_path + 'activity_histogram' + '.pdf'
    fig.savefig(figname, format='pdf')

    
    
def main(culled_dataset, save_figure_path=''):
    print
    print 'Plotting activity histogram'
    plot_activity_histogram(culled_dataset, save_figure_path=save_figure_path)

if __name__ == '__main__':
    config = analysis_configuration.Config()
    culled_dataset = fad.load('../' + config.culled_datasets_path + config.culled_dataset_name)
    
    main(culled_dataset, save_figure_path='../../figures/activity_histograms/')
    


