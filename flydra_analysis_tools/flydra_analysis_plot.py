import sys
import fly_plot_lib
fly_plot_lib.set_params.pdf()
import fly_plot_lib.plot as fpl
import fly_plot_lib.flymath as flymath
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

import trajectory_analysis_core as tac


def plot_trajectory_lengths(dataset):
    lengths = []
    for key, trajec in dataset.trajecs.items():
        lengths.append(trajec.length)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)

    fpl.histogram(ax, [np.array(lengths)], bins=50, show_smoothed=False)

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

def cartesian_spagetti(ax, dataset, axes=[0,1], keys=None, nkeys=300, start_key=0, show_saccades=False, keys_to_highlight=[], colormap='jet', color='gray', color_attribute=None, norm=None, show_start=True):
    if keys is None:
        keys = dataset.trajecs.keys()
    else:
        if type(keys) is not list:
            keys = [keys]
            
    if nkeys < len(keys):
        last_key = np.min([len(keys), start_key+nkeys])
        keys = keys[start_key:last_key]
        keys.extend(keys_to_highlight)
            
    for key in keys:
        trajec = dataset.trajecs[key]    
        
        frames = np.arange(trajec.frame_range_roi[0], trajec.frame_range_roi[-1])
        
        if key in keys_to_highlight:
            alpha = 1
            linewidth = 1
            color = 'black'
            zorder = 500
            ax.plot(trajec.positions[frames,axes[0]], trajec.positions[frames,axes[1]], '-', color='black', alpha=1, linewidth=linewidth, zorder=zorder)
        else:
            alpha = 1
            linewidth = 0.5
            color = color
            zorder = 100

            if color_attribute is not None:
                c = trajec.__getattribute__(color_attribute)                
                fpl.colorline(ax,trajec.positions[frames,axes[0]], trajec.positions[frames,axes[1]], c[frames], colormap=colormap, linewidth=linewidth, alpha=alpha, zorder=zorder, norm=norm)
            else:
                ax.plot(trajec.positions[frames,axes[0]], trajec.positions[frames,axes[1]], '-', color=color, alpha=1, linewidth=linewidth, zorder=zorder)
            
        if show_saccades:
            if len(trajec.sac_ranges) > 0:
                for sac_range in trajec.sac_ranges:
                    angle_subtended = trajec.angle_subtended_by_post[sac_range[0]]
                    if sac_range[0] in frames and angle_subtended < 100.*np.pi/180.:
                    #if 1:
                        ax.plot(trajec.positions[sac_range,0], trajec.positions[sac_range,1], '-', color='red', alpha=1, linewidth=linewidth, zorder=zorder+1)
                        #sac = patches.Circle( (trajec.positions[sac_range[0],0], trajec.positions[sac_range[0],1]), radius=0.001, facecolor='red', edgecolor='none', alpha=alpha2, zorder=zorder+1)
                        #ax.add_artist(sac)
            
        if show_start:
            start = patches.Circle( (trajec.positions[frames[0],axes[0]], trajec.positions[frames[0],axes[1]]), radius=0.004, facecolor='green', edgecolor='none', linewidth=0, alpha=1, zorder=zorder+1)
            ax.add_artist(start)
    
    
def prep_cartesian_spagetti_for_saving(ax):
    fig.set_size_inches(fig_width,fig_height)
    rect = patches.Rectangle( [-.25, -.15], .5, .3, facecolor='none', edgecolor='gray', clip_on=False, linewidth=0.2)
    ax.add_artist(rect)
    
    offset = 0.00
    dxy = 0.05
    #xarrow = patches.FancyArrowPatch(posA=(-.25+offset, -.15+offset), posB=(-.25+offset+dxy, -.15+offset), arrowstyle='simple') 
    #patches.Arrow( -.25+offset, -.15+offset, dxy, 0, color='black', width=0.002)
    xarrow = patches.FancyArrowPatch((-.25+offset, -.15+offset), (-.25+offset+dxy, -.15+offset), arrowstyle="-|>", mutation_scale=10, color='gray', shrinkA=0, clip_on=False)
    ax.add_patch(xarrow)
    yarrow = patches.FancyArrowPatch((-.25+offset, -.15+offset), (-.25+offset, -.15+offset+dxy), arrowstyle="-|>", mutation_scale=10, color='gray', shrinkA=0, clip_on=False)
    ax.add_artist(yarrow)
    text_offset = -.011
    ax.text(-.25+offset+dxy+text_offset, -.15+offset+.005, 'x', verticalalignment='bottom', horizontalalignment='left', color='gray', weight='bold')
    ax.text(-.25+offset+.005, -.15+offset+dxy+text_offset, 'y', verticalalignment='bottom', horizontalalignment='left', color='gray', weight='bold')
    
    scale_bar_offset = 0.01
    ax.hlines(-0.15+scale_bar_offset, 0.25-scale_bar_offset-.1, 0.25-scale_bar_offset, linewidth=1, color='gray')
    ax.text(0.25-scale_bar_offset-.1/2., -0.15+scale_bar_offset+.002, '10cm', horizontalalignment='center', verticalalignment='bottom', color='gray')
    
    ax.set_aspect('equal')
    
    scaling = .5/.75
    margin = 0.04
    aspect_ratio = 3/5. # height/width
    
    fig_width = 7.204*scaling
    plt_width = fig_width - 2*margin*(1-aspect_ratio)
    fig_height = plt_width*aspect_ratio + 2*margin
    
    fig = ax.figure
    
    fig.set_size_inches(fig_width,fig_height)
    fig.subplots_adjust(bottom=margin, top=1-margin, right=1, left=0)
    ax.set_axis_off()
    
    fpl.adjust_spines(ax, ['left', 'bottom'])




###############

def heatmap(ax, dataset, keys=None, frame_list=None, axis='xy', logcolorscale=False, xticks=None, yticks=None, zticks=None, rticks=None, normalize_for_speed=True, colornorm=None, center=[0,0], bins=[100,100,100], depth_range=None, colormap='jet', return_img=False, weight_attribute=None, velocity_axis=2, velocity_range=[-100,100]):  
    
    if axis == 'xy':
        axis_num = [0,1,2]
    elif axis == 'xz':
        axis_num = [0,2,1]
    elif axis == 'yz':
        axis_num = [1,2,0] 

    if keys is None:
        keys = dataset.trajecs.keys()
        
    if xticks is None:
        xticks = [-1, 0, 1]
    if yticks is None:
        yticks = [-1, 0, 1]
    if zticks is None:
        zticks = [-1, 0, 1]
    if rticks is None:
        rticks = [-1, -1, 1]
        
    if axis == 'rz':
        try:
            trajec = dataset.get_trajec()
            tmp = trajec.xy_distance_to_point
        except:
            raise(ValueError, 'Need to calculate xy_distance_to_point prior to calling this function, see trajectory_analysis_core.calc_xy_distance_to_point')
            #fad.iterate_calc_function(dataset, tac.calc_xy_distance_to_point, center)        
            
    # collect data
    xpos = np.array([])
    ypos = np.array([])
    zpos = np.array([])
    if weight_attribute is not None:
        weights = np.array([])
    else:
        weights = None
    radial = np.array([])
    
    i = -1
    for key in keys:
        i += 1
        trajec = dataset.trajecs[key]
        
        if frame_list is None:
            if depth_range is None:
                frames = np.arange(0, trajec.length-1)
            else:
                frames = []
                for f in range(trajec.length):
                    if flymath.in_range(trajec.positions[f,axis_num[-1]], depth_range):
                        if flymath.in_range(trajec.velocities[f,velocity_axis], velocity_range):
                            frames.append(f)
        else:
            if depth_range is None:
                frames = frame_list[i]
            else:
                frames = []
                for f in range(trajec.length):
                    if flymath.in_range(trajec.positions[f,axis_num[-1]], depth_range):
                        if flymath.in_range(trajec.velocities[f,velocity_axis], velocity_range):
                            if f in frame_list[i]:
                                frames.append(f)
            
            
            
        
        
            
        if len(frames) == 0:
            continue
            
        if normalize_for_speed:
            xpos = np.hstack( (xpos, trajec.positions_normalized_by_speed[0:-1,0]) )
            ypos = np.hstack( (ypos, trajec.positions_normalized_by_speed[0:-1,1]) )
            zpos = np.hstack( (zpos, trajec.positions_normalized_by_speed[0:-1,2]) )
            if axis == 'rz':
                radial = np.hstack( (radial, trajec.xy_distance_to_point_normalized_by_speed[frames]) )
    
        else:
            xpos = np.hstack( (xpos, trajec.positions[frames,0]) )
            ypos = np.hstack( (ypos, trajec.positions[frames,1]) )
            zpos = np.hstack( (zpos, trajec.positions[frames,2]) )
            if weight_attribute is not None:
                weights = np.hstack( (weights, trajec.__getattribute__(weight_attribute)[frames]) )
            if axis == 'rz':
                radial = np.hstack( (radial, trajec.xy_distance_to_point[frames]) )
    
    if axis == 'xy':
        img = fpl.histogram2d(ax, xpos, ypos, bins=[bins[0], bins[1]], logcolorscale=logcolorscale, colornorm=colornorm, colormap=colormap, return_img=return_img, weights=weights)
    elif axis == 'xz':
        img = fpl.histogram2d(ax, xpos, zpos, bins=[bins[0], bins[2]], logcolorscale=logcolorscale, colornorm=colornorm, colormap=colormap, return_img=return_img, weights=weights)
    elif axis == 'yz':
        img = fpl.histogram2d(ax, ypos, zpos, bins=[bins[1], bins[2]], logcolorscale=logcolorscale, colornorm=colornorm, colormap=colormap, return_img=return_img, weights=weights)
    elif axis == 'rz':
        img = fpl.histogram2d(ax, radial, zpos, bins=[bins[0], bins[1]], logcolorscale=logcolorscale, colornorm=colornorm, colormap=colormap, return_img=return_img, weights=weights)
        
    if axis == 'xy':
        use_xticks = xticks
        use_yticks = yticks
        ax.set_xlabel('x axis')
        ax.set_ylabel('y axis')
    elif axis == 'yz':
        use_xticks = yticks
        use_yticks = zticks
        ax.set_xlabel('y axis')
        ax.set_ylabel('z axis')
    elif axis == 'xz':
        use_xticks = xticks
        use_yticks = zticks
        ax.set_xlabel('x axis')
        ax.set_ylabel('z axis')
    elif axis == 'rz':
        use_xticks = rticks
        use_yticks = zticks
        ax.set_xlabel('radial axis')
        ax.set_ylabel('z axis')
    
    if 1:
        ax.set_xlim(use_xticks[0], use_xticks[-1])
        ax.set_ylim(use_yticks[0], use_yticks[-1])
        
    fpl.adjust_spines(ax, ['left', 'bottom'], xticks=use_xticks, yticks=use_yticks)
    
    ax.set_aspect('equal')
    
    if return_img:
        return  img

def show_start_stop(dataset):
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    artists = []
    
    xpos = []
    ypos = []
    
    for key, trajec in dataset.trajecs.items():
        if 1:
            x = trajec.positions[0,0]
            y = trajec.positions[0,1]
            start = patches.Circle( (x, y), radius=0.003, facecolor='green', edgecolor='none', alpha=1, linewidth=0)
            x = trajec.positions[-1,0]
            y = trajec.positions[-1,1]
            stop = patches.Circle( (x, y), radius=0.003, facecolor='red', edgecolor='none', alpha=1, linewidth=0)
            
            #artists.append(start)
            artists.append(stop)
        if 0:
            xpos.append(trajec.positions[-1,0])
            ypos.append(trajec.positions[-1,1])
        
    if 1:
        for artist in artists:
            ax.add_artist(artist)
            
        
    #fpl.histogram2d(ax, np.array(xpos), np.array(ypos), bins=100, logcolorscale=True, xextent=[-.2,.2], yextent=[-.75,.25])
        
    ax.set_aspect('equal')
        
    fpl.adjust_spines(ax, ['left', 'bottom'], xticks=[-.2, 0, .2], yticks=[-.75, -.5, -.25, 0, .25])
    ax.set_xlabel('x axis, m')
    ax.set_ylabel('y axis, m')
    ax.set_title('xy plot, color=speed from 0-0.5 m/s')

    fig.set_size_inches(8,8)
    
    
    fig.savefig('start_stop.pdf', format='pdf')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


