import sys, os
import fly_plot_lib
fly_plot_lib.set_params.pdf()
import fly_plot_lib.plot as fpl
import fly_plot_lib.flymath as flymath
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

import flydra_analysis_tools as fat
fad = fat.flydra_analysis_dataset
dac = fat.dataset_analysis_core
fap = fat.flydra_analysis_plot
tac = fat.trajectory_analysis_core

import xyz_axis_grid
    
FIGURE_PATH = 'figures/diagnostic_plots'
    
def print_basic_statistics(dataset, keys=None):
    if keys is None:
        keys = dataset.trajecs.keys()
    
    n_flies = 0
    mean_speeds = []
    frame_length = []
    
    x = []
    y = []
    z = []
    for key in keys:
        trajec = dataset.trajecs[key]
        n_flies += 1
        mean_speeds.append( np.mean(trajec.speed) )
        frame_length.append( len(trajec.speed) )
        
        x.append([np.min(trajec.positions[:,0]), np.max(trajec.positions[:,0])])
        y.append([np.min(trajec.positions[:,1]), np.max(trajec.positions[:,1])])
        z.append([np.min(trajec.positions[:,2]), np.max(trajec.positions[:,2])])
    
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    
    mean_speeds = np.array(mean_speeds)
    frame_length = np.array(frame_length)
        
    print 'num flies: ', n_flies
    print 'mean speed: ', np.mean(mean_speeds)
    print
    print 'mean frame length: ', np.mean(frame_length)
    print 'max  frame length: ', np.max(frame_length)
    print 'min  frame length: ', np.min(frame_length)
    print
    print np.min(x[:,0]), np.max(x[:,1])
    print np.min(y[:,0]), np.max(y[:,1])
    print np.min(z[:,0]), np.max(z[:,1])
    
############################################################
    
def plot_births_and_deaths(config, dataset, keys=None):
    if keys is None:
        keys = dataset.trajecs.keys()
        
    def make_plot(ax, births, deaths, keys, axis='yz'):    
            
        if axis == 'yz':
            x_births = births[:, [1]]
            y_births = births[:, [2]]
            x_deaths = deaths[:, [1]]
            y_deaths = deaths[:, [2]]
        
        if axis == 'xy':
            x_births = births[:, [0]]
            y_births = births[:, [1]]
            x_deaths = deaths[:, [0]]
            y_deaths = deaths[:, [1]]
        
        if axis == 'xz':
            x_births = births[:, [0]]
            y_births = births[:, [2]]
            x_deaths = deaths[:, [0]]
            y_deaths = deaths[:, [2]]
        
        fpl.scatter(ax, x_births.flatten(), y_births.flatten(), color='green', radius=0.005, alpha=1, use_ellipses=False)
        fpl.scatter(ax, x_deaths.flatten(), y_deaths.flatten(), color='red', radius=0.005, alpha=1, use_ellipses=False)
    
    def collect_births_and_deaths(keys):
        births = []
        deaths = []
        for key in keys:
            trajec = dataset.trajecs[key]
            births.append(trajec.positions[0])
            deaths.append(trajec.positions[-1])
        return np.array(births), np.array(deaths)
    
    births, deaths = collect_births_and_deaths(keys)
    
    fig = plt.figure(figsize=(8, 4))
    axes = xyz_axis_grid.get_axes(config, fig, figure_padding=[0.15, 0.15, 0.15, 0.15], subplot_padding=0.01)
    fig = axes['xy'].figure
    
    make_plot(axes['xy'], births, deaths, keys, axis='xy')
    make_plot(axes['xz'], births, deaths, keys, axis='xz')
    make_plot(axes['yz'], births, deaths, keys, axis='yz')
    
    xyz_axis_grid.set_spines(config, axes, spines=True)
    
    name = 'births_and_deaths.pdf'
    path = os.path.join(config.path, FIGURE_PATH)
    filename = os.path.join(path,name) 
    fig.savefig(filename, format='pdf')
    

############################################################


def plot_heatmaps(config, dataset, keys=None, zslice='all', xz_axis_z_velocity_range=[-100,100]):
    if keys is None:
        keys = dataset.trajecs.keys()
        
    def make_plot(ax, keys, axis='yz'):    
        n_frames = 0
        for key in keys:
            trajec = dataset.trajecs[key]
            n_frames += trajec.length
        
        binres = 0.003
        binsx = np.linspace(config.xlim[0], config.xlim[1], int( (config.xlim[1] - config.xlim[0]) /binres))
        binsy = np.linspace(config.ylim[0], config.ylim[1], int( (config.ylim[1] - config.ylim[0]) /binres))
        binsz = np.linspace(config.zlim[0], config.zlim[1], int( (config.zlim[1] - config.zlim[0]) /binres))
        
        if axis=='xy':
            depth = (config.zlim[1] - config.zlim[0])
        elif axis=='yz':
            depth = (config.xlim[1] - config.xlim[0])
        elif axis=='xz':
            depth = (config.ylim[1] - config.ylim[0])
        colornorm = [0,0.0003*n_frames*depth]
        colormap = 'hot'
        print n_frames
        
        depth_range = [-2, 2]
                
        # get rid of errant walking flies
        if axis=='xz':
            velocity_range = xz_axis_z_velocity_range
        else:
            velocity_range = [-100,100]
            
        img = fap.heatmap(ax, dataset, axis=axis, keys=keys, xticks=config.ticks['x'], yticks=config.ticks['y'], zticks=config.ticks['z'], rticks=config.ticks['r'], colornorm=colornorm, normalize_for_speed=False, bins=[binsx,binsy,binsz], depth_range=depth_range, colormap=colormap, return_img=False, velocity_range=velocity_range)
        
    
    fig = plt.figure(figsize=(8, 5))
    axes = xyz_axis_grid.get_axes(config, fig, figure_padding=[0.15, 0.15, 0.15, 0.15], subplot_padding=0.01)
    fig = axes['xy'].figure
    
    make_plot(axes['xy'], keys, axis='xy')
    make_plot(axes['xz'], keys, axis='xz')
    make_plot(axes['yz'], keys, axis='yz')
    
    xyz_axis_grid.set_spines(config, axes, spines=True)
    
    name = 'diagnostic_heatmaps.pdf'
    path = os.path.join(config.path, FIGURE_PATH)
    filename = os.path.join(path,name) 
    fig.savefig(filename, format='pdf')
    
    
    
    


