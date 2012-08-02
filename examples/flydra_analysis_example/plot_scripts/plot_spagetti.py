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

def plot_spagetti(dataset, axis='xy', save_figure_path=''):
    keys = get_keys(dataset)
    print 'plotting spagetti, axis: ', axis
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
    
    
    
    
    
def example_cartesian_spagetti(dataset, axis='xy', xlim=(-.15, .15), ylim=(-.25, .25), zlim=(-.15, -.15), keys=None, keys_to_highlight=[], show_saccades=False, colormap='jet', color_attribute=None, artists=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    if axis=='xy': # xy plane
        ax.set_ylim(ylim[0], ylim[1])
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_autoscale_on(False)
        ax.set_aspect('equal')
        axes=[0,1]
        xy_spagetti(ax, dataset, keys=keys, nkeys=300, show_saccades=show_saccades, keys_to_highlight=keys_to_highlight, colormap=None, color='gray')

    if artists is not None:
        for artist in artists:
            ax.add_artist(artist)
    
    fig.savefig('example_xy_spagetti_plot.pdf', format='pdf')

def example_colored_cartesian_spagetti(dataset, axis='xy', xlim=(-0.2, .2), ylim=(-0.75, .25), zlim=(-.15, -.15), keys=None, keys_to_highlight=[], show_saccades=False, colormap='jet', color_attribute='speed', norm=(0,0.5), artists=None):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    if axis=='xy': # xy plane
        ax.set_ylim(ylim[0], ylim[1])
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_autoscale_on(True)
        ax.set_aspect('equal')
        axes=[0,1]
        cartesian_spagetti(ax, dataset, keys=keys, nkeys=300, start_key=0, axes=axes, show_saccades=show_saccades, keys_to_highlight=[], colormap=colormap, color_attribute=color_attribute, norm=norm, show_start=False)
        
    post = patches.Circle( (0, 0), radius=0.01, facecolor='black', edgecolor='none', alpha=1)
    artists = [post]
    
    if artists is not None:
        for artist in artists:
            ax.add_artist(artist)

    #prep_cartesian_spagetti_for_saving(ax)
    fpl.adjust_spines(ax, ['left', 'bottom'], xticks=[-.2, 0, .2], yticks=[-.75, -.5, -.25, 0, .25])
    ax.set_xlabel('x axis, m')
    ax.set_ylabel('y axis, m')
    ax.set_title('xy plot, color=speed from 0-0.5 m/s')

    fig.set_size_inches(8,8)

    fig.savefig('example_colored_xy_spagetti_plot.pdf', format='pdf')

    return ax
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
def main(culled_dataset, save_figure_path=''):
    print
    print 'Plotting heatmaps'
    plot_heatmap(culled_dataset, 'xy', save_figure_path=save_figure_path)
    plot_heatmap(culled_dataset, 'yz', save_figure_path=save_figure_path)

if __name__ == '__main__':
    config = analysis_configuration.Config()
    culled_dataset = fad.load('../' + config.culled_dataset_name)
    
    main(culled_dataset, save_figure_path='../figures/heatmaps/')
    


