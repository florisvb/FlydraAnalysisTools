#!/usr/bin/env python
from flydra_analysis_tools import flydra_analysis_dataset as fad
from flydra_analysis_tools import flydra_analysis_plot as fap
from flydra_analysis_tools import floris_plot_lib as fpl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

from flydra_analysis_tools import dataset_configuration


def plot_all_heatmap(dataset, axis, savefig=None):
    keys = dataset.trajecs.keys()
    print 'number of flies: ', len(keys)
        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    fap.heatmap(ax, dataset, keys=keys, axis=axis, logcolorscale=False)
    
    if axis == 'xy':
        post = patches.Circle( (0.00130331, -0.35593937), radius=0.01, facecolor='black', edgecolor='none', alpha=1, linewidth=0)
    elif axis == 'yz':
        ax.set_ylim(0,0.33)
        post = patches.Rectangle( (-0.35593937-.01,0), width=0.02, facecolor='black', height=.16, edgecolor='none', alpha=1, linewidth=0)
    artists = [post]
    
    if artists is not None:
        for artist in artists:
            ax.add_artist(artist)
    
    fap.heatmap(ax, dataset, keys=keys, axis=axis, logcolorscale=False)
    
    if savefig:
        fig.savefig(savefig, format='pdf')
    
    
############### plotting script ############################

if __name__ == '__main__':
    
    # read config file
    # do plot functions
    filename = '../analysis.config'
    config = dataset_configuration.Config(filename)
    
    plot_heatmap(config.dataset, axis='xy', savefig='../figures/heatmaps/heatmap_xy.pdf'
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        


