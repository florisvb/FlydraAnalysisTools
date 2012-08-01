#!/usr/bin/env python
import sys
sys.path.append('/home/caveman/src/flydra_analysis_tools')
sys.path.append('/home/caveman/src/odor_control_analysis')
sys.path.append('/home/caveman/src/floris_functions')
import flydra_analysis_dataset as fad
import dataset_analysis_core as dac
import trajectory_analysis_core as tac
import flydra_analysis_plot as fap
import floris_plot_lib as fpl
import odor_control_analysis as oca
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

raw_data_filename = 'DATA20120724_205559.h5'
culled_dataset_filename = 'culled_datasets/culled_dataset_' + raw_data_filename[0:-2] + 'pickle'
culled_dataset = fad.load(culled_dataset_filename)

#culled_dataset = fad.load('culled_datasets/culled_dataset_DATA20120525_181838.kalmanized.pickle')

################# get trajectories that are X long and start downwind #################


def get_keys(dataset):
    
    keys = dataset.trajecs.keys()
                
    return keys

################# plotting functions #######################

def plot_odor_heatmap(dataset):
    dataset_in_odor = fad.make_dataset_with_attribute_filter(dataset, 'in_odor', True)
    
    # get trajectories that are at least min_length long and start downstream, at y<-0.3
    keys = get_keys(dataset_in_odor)
    print 'odor: ', len(keys)
        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    axis = 'xy'
    fap.heatmap(ax, dataset, keys=keys, axis=axis, logcolorscale=False)
    
    
    # post top center: array([ 0.00130331, -0.35593937,  0.16951523])
    
    if axis == 'xy':
        post = patches.Circle( (0.00130331, -0.35593937), radius=0.01, facecolor='black', edgecolor='none', alpha=1, linewidth=0)
    elif axis == 'yz':
        ax.set_ylim(0,0.33)
        post = patches.Rectangle( (-0.35593937-.01,0), width=0.02, facecolor='black', height=.16, edgecolor='none', alpha=1, linewidth=0)
    artists = [post]
    
    if artists is not None:
        for artist in artists:
            ax.add_artist(artist)
    
    fap.heatmap(ax, dataset_in_odor, keys=keys, axis=axis, logcolorscale=False)
    fig.savefig('figures/heatmaps/odor_heatmap.pdf', format='pdf')
    
def plot_all_heatmap(dataset):
    keys = dataset.trajecs.keys()
    print 'all: ', len(keys)
        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    axis = 'xy'
    fap.heatmap(ax, dataset, keys=keys, axis=axis, logcolorscale=False)
    
    
    # post top center: array([ 0.00130331, -0.35593937,  0.16951523])
    
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
    fig.savefig('figures/heatmaps/all_heatmap.pdf', format='pdf')
    
def plot_no_odor_control_heatmap(dataset):
    dataset_no_odor = fad.make_dataset_with_attribute_filter(dataset, 'in_odor', False)
    
    # get trajectories that are at least min_length long and start downstream, at y<-0.3
    
    keys = []
    for key, trajec in dataset_no_odor.trajecs.items():
        hr = int(trajec.timestamp_local[9:11])
        minute = int(trajec.timestamp_local[11:13])
        localtime = hr + minute/60.
        if (localtime < 4 and localtime > 0) or (localtime > 16 and localtime <  24):
            keys.append(key)
    print 'no odor control: ', len(keys)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    axis = 'xy'
    fap.heatmap(ax, dataset, keys=keys, axis=axis, logcolorscale=False)
    
    
    # post top center: array([ 0.00130331, -0.35593937,  0.16951523])
    
    if axis == 'xy':
        post = patches.Circle( (0.00130331, -0.35593937), radius=0.01, facecolor='black', edgecolor='none', alpha=1, linewidth=0)
    elif axis == 'yz':
        ax.set_ylim(0,0.33)
        post = patches.Rectangle( (-0.35593937-.01,0), width=0.02, facecolor='black', height=.16, edgecolor='none', alpha=1, linewidth=0)
    artists = [post]
    
    if artists is not None:
        for artist in artists:
            ax.add_artist(artist)
    
    fap.heatmap(ax, dataset_no_odor, keys=keys, axis=axis, logcolorscale=False)
    fig.savefig('figures/heatmaps/no_odor_control_heatmap.pdf', format='pdf')
    
def plot_no_odor_heatmap(dataset):
    dataset_no_odor = fad.make_dataset_with_attribute_filter(dataset, 'in_odor', False)
    
    # get trajectories that are at least min_length long and start downstream, at y<-0.3
    
    keys = []
    for key, trajec in dataset_no_odor.trajecs.items():
        hr = int(trajec.timestamp_local[9:11])
        minute = int(trajec.timestamp_local[11:13])
        localtime = hr + minute/60.
        if localtime > 5 and localtime < 15:
            keys.append(key)
    print 'no odor: ', len(keys)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    axis = 'xy'
    fap.heatmap(ax, dataset, keys=keys, axis=axis, logcolorscale=False)
    
    
    # post top center: array([ 0.00130331, -0.35593937,  0.16951523])
    
    if axis == 'xy':
        post = patches.Circle( (0.00130331, -0.35593937), radius=0.01, facecolor='black', edgecolor='none', alpha=1, linewidth=0)
    elif axis == 'yz':
        ax.set_ylim(0,0.33)
        post = patches.Rectangle( (-0.35593937-.01,0), width=0.02, facecolor='black', height=.16, edgecolor='none', alpha=1, linewidth=0)
    artists = [post]
    
    if artists is not None:
        for artist in artists:
            ax.add_artist(artist)
    
    fap.heatmap(ax, dataset_no_odor, keys=keys, axis=axis, logcolorscale=False)
    fig.savefig('figures/heatmaps/no_odor_heatmap.pdf', format='pdf')
    
############### plotting script ############################

plot_all_heatmap(culled_dataset)
plot_odor_heatmap(culled_dataset)
plot_no_odor_heatmap(culled_dataset)
plot_no_odor_control_heatmap(culled_dataset)


