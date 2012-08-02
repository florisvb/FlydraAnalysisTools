#!/usr/bin/env python
import sys
import flydra_analysis_tools as fat
from flydra_analysis_tools import floris_plot_lib as fpl
fad = fat.flydra_analysis_dataset
dac = fat.dataset_analysis_core
fap = fat.flydra_analysis_plot
tac = fat.trajectory_analysis_core

import numpy as np
import matplotlib.pyplot as plt

import analysis_configuration


################# get trajectory keys #################

def get_keys(dataset):
    keys = dataset.trajecs.keys()
    return keys

################# plotting functions #######################

def plot_heatmap(dataset, axis='xy', save_figure_path=''):
    keys = get_keys(dataset)
    print 'plotting heatmap, axis: ', axis
    print 'number of keys: ', len(keys)
        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    fap.heatmap(ax, dataset, axis=axis, keys=keys)
    
    artists = []    
    if artists is not None:
        for artist in artists:
            ax.add_artist(artist)
    
    fap.heatmap(ax, dataset, axis=axis, keys=keys)
    figname = save_figure_path + 'heatmap_' + axis + '.pdf'
    fig.savefig(figname, format='pdf')
    
def main(culled_dataset, save_figure_path=''):
    print
    print 'Plotting heatmaps'
    plot_heatmap(culled_dataset, 'xy', save_figure_path=save_figure_path)
    plot_heatmap(culled_dataset, 'yz', save_figure_path=save_figure_path)

if __name__ == '__main__':
    config = analysis_configuration.Config()
    culled_dataset = fad.load('../' + config.culled_dataset_name)
    
    main(culled_dataset, save_figure_path='../figures/heatmaps/')
    


