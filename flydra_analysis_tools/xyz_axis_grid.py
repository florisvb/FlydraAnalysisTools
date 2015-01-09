from matplotlib import gridspec
import fly_plot_lib.plot as fpl
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np

def get_axes(config, fig=None, figure_padding=[0.125, 0.125, 0.05, 0.125], subplot_padding=0.05):

    if fig is None:
        fig = plt.figure(figsize=(4.5, 2.025))
    
    x = (config.xlim[1] - config.xlim[0])
    y = (config.ylim[1] - config.ylim[0])
    z = (config.zlim[1] - config.zlim[0])

    aspect_ratio = (y+z+subplot_padding)/(x+y+subplot_padding)

    gs1 = gridspec.GridSpec(2, 2, width_ratios=[x,y])
    gs1.update(left=figure_padding[0]*aspect_ratio, right=1-figure_padding[1]*aspect_ratio, wspace=subplot_padding, hspace=subplot_padding, top=1-figure_padding[2]+subplot_padding, bottom=figure_padding[3]-subplot_padding)
    
    ax_xy = plt.subplot(gs1[1, 0])
    ax_xz = plt.subplot(gs1[0, 0])
    ax_yz = plt.subplot(gs1[0, 1])

    ax_xy.set_aspect('equal')
    ax_xz.set_aspect('equal')
    ax_yz.set_aspect('equal')
    
    axes = [ax_xy, ax_xz, ax_yz]
    axes = {'xy': axes[0], 'xz': axes[1], 'yz': axes[2]}
    return axes
    
def set_spines(config, axes, spines=True):
    
    axes['yz'].set_xlim(config.ylim[0],config.ylim[1])
    axes['yz'].set_ylim(config.zlim[0],config.zlim[1])
    
    axes['xy'].set_xlim(config.xlim[0],config.xlim[1])
    axes['xy'].set_ylim(config.ylim[0],config.ylim[1])
    
    axes['xz'].set_xlim(config.xlim[0],config.xlim[1])
    axes['xz'].set_ylim(config.zlim[0],config.zlim[1])
        
    if spines:
        yticks = config.ticks['y']
        xticks = config.ticks['x']
        zticks = config.ticks['z']
    
        fpl.adjust_spines(axes['xz'], ['left'], yticks=yticks)
        fpl.adjust_spines(axes['xy'], ['left', 'bottom'], xticks=xticks, yticks=zticks)
        fpl.adjust_spines(axes['yz'], ['right', 'bottom'], xticks=yticks, yticks=zticks)
            
    else:
        fpl.adjust_spines(axes['xy'], [])
        fpl.adjust_spines(axes['xz'], [])
        fpl.adjust_spines(axes['yz'], [])
    
def draw_post(config, axes, postcolor='black'):
    ax_xy, ax_xz, ax_yz = axes
    
    circle = patches.Circle(config.post_center[0:2], config.post_radius, color=postcolor, edgecolor='none')
    ax_xy.add_artist(circle)
    
    height = config.post_center[2] - config.ticks['z'][0]
    
    rectangle_xz = patches.Rectangle([-1*config.post_radius + config.post_center[0], config.ticks['z'][0]], config.post_radius*2, height, color=postcolor, edgecolor='none')
    ax_xz.add_artist(rectangle_xz)
    
    rectangle_yz = patches.Rectangle([-1*config.post_radius + config.post_center[1], config.ticks['z'][0]], config.post_radius*2, height, color=postcolor, edgecolor='none')
    ax_yz.add_artist(rectangle_yz)
    
def set_spines_and_labels(axes):
    ax_xy, ax_xz, ax_yz = axes
    
    yticks = [-.15, 0, .15]
    xticks = [-.2, 0, 1]
    zticks = [-.15, 0, .15]
    
    ax_xy.set_xlim(xticks[0], xticks[-1])
    ax_xy.set_ylim(yticks[0], yticks[-1])
    
    ax_xz.set_xlim(xticks[0], xticks[-1])
    ax_xz.set_ylim(zticks[0], zticks[-1])
    
    ax_yz.set_xlim(yticks[0], yticks[-1])
    ax_yz.set_ylim(zticks[0], zticks[-1])
    
    
    fpl.adjust_spines(ax_xy, ['left'], xticks=xticks, yticks=yticks)
    fpl.adjust_spines(ax_xz, ['left', 'bottom'], xticks=xticks, yticks=zticks)
    fpl.adjust_spines(ax_yz, ['right', 'bottom'], xticks=yticks, yticks=zticks)
    
    ax_xy.set_xlabel('')
    ax_xy.set_ylabel('y axis')
    
    ax_xz.set_ylabel('z axis')
    ax_xz.set_xlabel('x axis, upwind negative')
    
    ax_yz.set_xlabel('y axis')
    ax_yz.yaxis.set_label_position('right')
    ax_yz.set_ylabel('z axis')
    
    ax_xy.set_aspect('equal')
    ax_xz.set_aspect('equal')
    ax_yz.set_aspect('equal')
    

def plot_trajectory(axes, trajec, frames=None):
    ax_xy, ax_xz, ax_yz = axes
    
    if frames is None:
        frames = np.arange(0,trajec.length)
    
    ax_xy.plot(trajec.positions[frames,0], trajec.positions[frames,1], 'black')
    ax_xz.plot(trajec.positions[frames,0], trajec.positions[frames,2], 'black')
    ax_yz.plot(trajec.positions[frames,1], trajec.positions[frames,2], 'black')
    
    for frame, behavior in trajec.post_behavior_dict.items():
        if behavior == 'landing':
            color = 'green'
        elif behavior == 'takeoff':
            color = 'red'
        else:
            print behavior
        ax_xy.plot(trajec.positions[frame,0], trajec.positions[frame,1], 'o', color=color)
        ax_xz.plot(trajec.positions[frame,0], trajec.positions[frame,2], 'o', color=color)
        ax_yz.plot(trajec.positions[frame,1], trajec.positions[frame,2], 'o', color=color)
    
    
    
    
    
    
