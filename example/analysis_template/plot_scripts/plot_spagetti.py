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

################# get trajectory keys #################

def get_keys(dataset):
    keys = dataset.trajecs.keys()
    return keys

################# plotting functions #######################

def plot_colored_cartesian_spagetti(dataset, axis='xy', xlim=(-0.2, .2), ylim=(-0.75, .25), zlim=(0, 0.3), keys=None, keys_to_highlight=[], show_saccades=False, colormap='jet', color_attribute='speed', norm=(0,0.5), artists=None, save_figure_path=''):
    keys = get_keys(dataset)
    print 'plotting spagetti, axis: ', axis
    print 'number of keys: ', len(keys)
    if len(keys) < 1:
        print 'No data'
        return
        
    fig = plt.figure()
    ax = fig.add_subplot(111)

    if axis=='xy': # xy plane
        ax.set_ylim(ylim[0], ylim[1])
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_autoscale_on(True)
        ax.set_aspect('equal')
        axes=[0,1]
        fap.cartesian_spagetti(ax, dataset, keys=keys, nkeys=300, start_key=0, axes=axes, show_saccades=show_saccades, keys_to_highlight=[], colormap=colormap, color_attribute=color_attribute, norm=norm, show_start=False)
        
    if axis=='yz': # yz plane
        ax.set_ylim(zlim[0], zlim[1])
        ax.set_xlim(ylim[0], ylim[1])
        ax.set_autoscale_on(True)
        ax.set_aspect('equal')
        axes=[1,2]
        fap.cartesian_spagetti(ax, dataset, keys=keys, nkeys=300, start_key=0, axes=axes, show_saccades=show_saccades, keys_to_highlight=[], colormap=colormap, color_attribute=color_attribute, norm=norm, show_start=False)
        
    if artists is not None:
        for artist in artists:
            ax.add_artist(artist)

    #prep_cartesian_spagetti_for_saving(ax)
    xticks = np.linspace(xlim[0], xlim[1], 3, endpoint=True).tolist()
    yticks = np.linspace(ylim[0], ylim[1], 5, endpoint=True).tolist()
    zticks = np.linspace(zlim[0], zlim[1], 3, endpoint=True).tolist()
    
    if axis=='xy':
        fpl.adjust_spines(ax, ['left', 'bottom'], xticks=xticks, yticks=yticks)
        ax.set_xlabel('x axis, m')
        ax.set_ylabel('y axis, m')
        ax.set_title('xy plot, color=speed from 0-0.5 m/s')

    if axis=='yz':
        fpl.adjust_spines(ax, ['left', 'bottom'], xticks=yticks, yticks=zticks)
        ax.set_xlabel('y axis, m')
        ax.set_ylabel('z axis, m')
        ax.set_title('yz plot, color=speed from 0-0.5 m/s')

    fig.set_size_inches(8,8)
    figname = save_figure_path + 'spagetti_' + axis + '.pdf'
    fig.savefig(figname, format='pdf')

    return ax
    
    
    
def main(culled_dataset, save_figure_path=''):
    print
    print 'Plotting spagetti'
    plot_colored_cartesian_spagetti(culled_dataset, axis='xy', save_figure_path=save_figure_path)
    plot_colored_cartesian_spagetti(culled_dataset, axis='yz', save_figure_path=save_figure_path)

if __name__ == '__main__':
    config = analysis_configuration.Config()
    culled_dataset = fad.load('../' + config.culled_datasets_path + config.culled_dataset_name)
    
    main(culled_dataset, save_figure_path='../../figures/spagetti/')
    


