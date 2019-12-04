import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from satellite_analysis import satellite_io as sio
from satellite_analysis import math_funcs as mf
from satellite_analysis import angular as ang
from satellite_analysis import kinematics as kin
from satellite_analysis import spatial as spa
from satellite_analysis import isotropic as iso
from satellite_analysis import rand_axes as ra


CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

line_style = ['-.', '--', ':']

def plot_angle_width(sat, mask_key, fraction=0.68, title=''):
    '''
    Generate a plot of the angle enclosing a fraction of satellites over the redshift
    range of the satellite halo catalogs.
    '''
    angle_width = sio.loop_hal(sat, mask_key, ang.angle_width, **{'threshold_fraction':fraction, 'angle_bins':sat.a_bins})
    iso_angle_width = sio.loop_iso(sat, mask_key, iso.iso_angle_width, **{'threshold_fraction':fraction, 'angle_bins':sat.a_bins})

    plt.figure()
    for i, hal_name in enumerate(angle_width.keys()):
        plt.plot(sat.redshift, angle_width[hal_name], label=sat.hal_label[mask_key][i], color=CB_color_cycle[i])
        plt.plot(sat.redshift, iso_angle_width[hal_name], color='k', linestyle=line_style[i], alpha=0.5, label=sat.hal_label[mask_key][i]+' isotropic')
    plt.legend()
    plt.xlabel('z', fontsize=12)
    plt.ylabel(r'Angle enclosing 68$\%$ of sats. [deg]', fontsize=12)
    plt.title(title)
    plt.show()

def plot_angle_width2(sat_list, mask_keys, fraction=0.68, title=''):
    '''
    Generate a two panel plot of the angle enclosing a fraction of satellites
    over the redshift range of the len(sat_list) number of satellite halo catalogs.
    '''
    fig, ax = plt.subplots(1, len(sat_list), sharex=True, sharey=True, figsize=(12,5))

    for i, (sat, mask_key) in enumerate(zip(sat_list, mask_keys)):
        angle_width = sio.loop_hal(sat, mask_key, ang.angle_width, **{'threshold_fraction':fraction, 'angle_bins':sat.a_bins})
        iso_angle_width = sio.loop_iso(sat, mask_key, iso.iso_angle_width, **{'threshold_fraction':fraction, 'angle_bins':sat.a_bins})

        for j, hal_name in enumerate(angle_width.keys()):
            ax[i].plot(sat.redshift, angle_width[hal_name], label=sat.hal_label[mask_key][i], color=CB_color_cycle[j])
            ax[i].plot(sat.redshift, iso_angle_width[hal_name], color='k', linestyle=line_style[j], alpha=0.5, label=sat.hal_label[mask_key][i]+' isotropic')
        ax[i].legend()
    ax[1].set_xlabel('Redshift (z)', fontsize=12)
    ax[0].set_ylabel('Angle enclosing 68$\%$ of sats. [deg]', fontsize=12)
    plt.title(title)
    plt.show()

def plot_coadd_angle(sat, mask_key, stat_type='mean', iso_regions=False):
    '''
    Generate a plot of fraction of satellites enclosed vs. enclosing angle,
    which varies from [0, 180] degrees for a single sat object.
    '''
    angle_range = sat.a_bins_plt
    angle_frac = sio.loop_hal(sat, mask_key, ang.fraction_open_angle, **{'angle_bins':sat.a_bins})
    coadd_angle_frac = mf.coadd_redshift(angle_frac, dict_key=None, coadd_axis=0)
    iso_angle_frac = sio.loop_iso(sat, mask_key, iso.iso_frac_open_angle, **{'angle_bins':sat.a_bins})
    coadd_iso_angle_frac = mf.coadd_redshift(iso_angle_frac, dict_key=None, coadd_axis=0)
    
    fig = plot_coadd_angle_comparison(angle_range, coadd_angle_frac, iso_prop_list=coadd_iso_angle_frac,
        iso_regions=iso_regions, halo_names=sat.hal_name,
        plot_title='', xlabel='Enclosing angle [deg]', ylabel='Fraction of satellites enclosed',
        stat_type=stat_type)

    return fig

def plot_coadd_angle_comp(sat_list, mask_keys, iso_regions=False):
    '''
    Generate a plot of fraction of satellites enclosed vs. enclosing angle,
    which varies from [0, 180] degrees. Compare host halos down columns, and compare
    halo catalog selection across rows.
    '''
    angle_ticks = [0, 45, 90, 135, 180]
    angle_labels = ['0', '45', '90', '135', '180']
    frac_ticks = [0, 0.5, 1]
    frac_labels = ['0', '0.5', '1']
    xlabel='Enclosing angle [deg]'
    ylabel='Fraction of satellites enclosed'

    fig, ax = plt.subplots(len(sat_list[0].hal_name), len(sat_list), sharex=True, sharey=True, figsize=(10,10))

    plt.xticks(angle_ticks, angle_labels)
    plt.yticks(frac_ticks, frac_labels)

    for i, (sat, mask_key) in enumerate(zip(sat_list, mask_keys)):
        angle_range = sat.a_bins_plt
        angle_frac = sio.loop_hal(sat, mask_key, ang.fraction_open_angle, **{'angle_bins':sat.a_bins})
        coadd_angle_frac = mf.coadd_redshift(angle_frac, dict_key=None, coadd_axis=0)
        iso_angle_frac = sio.loop_iso(sat, mask_key, iso.iso_frac_open_angle, **{'angle_bins':sat.a_bins})
        coadd_iso_angle_frac = mf.coadd_redshift(iso_angle_frac, dict_key=None, coadd_axis=0)

        for j, hal_name in enumerate(angle_frac.keys()):
            ax[j, i].fill_between(angle_range, coadd_angle_frac[hal_name]['percentile'][2], coadd_angle_frac[hal_name]['percentile'][3], alpha=0.25,
                color=CB_color_cycle[j], linewidth=0, hatch='-')
            ax[j, i].fill_between(angle_range, coadd_angle_frac[hal_name]['percentile'][0], coadd_angle_frac[hal_name]['percentile'][1], alpha=0.4,
                color=CB_color_cycle[j], linewidth=0, hatch='|')
            ax[j, i].plot(angle_range, coadd_angle_frac[hal_name]['mean'], color=CB_color_cycle[j], label=sat.hal_label[mask_key][i])
            if iso_regions is True:
                ax[j, i].fill_between(angle_range, coadd_iso_angle_frac[hal_name]['percentile'][2], coadd_iso_angle_frac[hal_name]['percentile'][3], alpha=0.25,
                    color='k', linewidth=0, hatch='-')
                ax[j, i].fill_between(angle_range, coadd_iso_angle_frac[hal_name]['percentile'][0], coadd_iso_angle_frac[hal_name]['percentile'][1], alpha=0.4,
                    color='k', linewidth=0, hatch='|')
            ax[j, i].plot(angle_range, coadd_iso_angle_frac[hal_name]['mean'], alpha=0.8, color='k', label=sat.hal_label[mask_key][i]+' isotropic')
            ax[j, i].legend()

    ax[2, 1].set_xlabel(xlabel, fontsize=12)
    ax[1, 0].set_ylabel(ylabel, fontsize=12)
    plt.show()

def plot_angle_v_r(sat_list, mask_keys, radius_fraction=0.5, isotropic=False, title=''):
    '''
    Generate one panel figure comparing enclosing angle vs. radius for either the
    true satellite distribution (isotropic=False) or the isotropic distribution
    (isotropic=True).
    '''
    point_style_arr = np.array((['^', 'v', '<'],['+', 'x', '3'],['s','p','h']))
    cm = plt.cm.get_cmap('viridis')

    plt.figure()

    for i, (sat, mask_key) in enumerate(zip(sat_list, mask_keys)):
        if isotropic is False:
            rad_angle_dict = sio.loop_hal(sat, mask_key, ang.open_angle_v_r, **{'threshold_fraction':radius_fraction, 'radius_bins':sat.r_bins, 'angle_bins':sat.a_bins})
        elif isotropic is True:
            rad_angle_dict = sio.loop_iso(sat, mask_key, iso.iso_open_angle_v_r, **{'threshold_fraction':radius_fraction, 'radius_bins':sat.r_bins, 'angle_bins':sat.a_bins})
        radii = sio.single_property_dict(rad_angle_dict, 'radii')
        angles = sio.single_property_dict(rad_angle_dict, 'angles')
    
        for j, hal_name in enumerate(rad_angle_dict.keys()):
            label_str = r'{}: $R^2$ = {:.3f}'.format(sat.hal_label[mask_key][j], np.corrcoef(radii[hal_name], angles[hal_name])[0,1]**2)
            pts = plt.scatter(radii[hal_name], angles[hal_name], c=sat.redshift, marker=point_style_arr[i, j], s=100, label=label_str, cmap=cm)

    cbar = plt.colorbar(pts)
    cbar.set_label('Redshift [z]')
    plt.legend()

    rlabel = int(radius_fraction*100)
    plt.xlabel('$R{}$ [kpc]'.format(rlabel))
    plt.ylabel('Enclosing angle [deg]')
    if isotropic == True:
        plt.title('Isotropic '+title, fontsize=12)
    else:
        plt.title(title, fontsize=12)

    plt.show()

def plot_angle_v_r_comp(sat_list, mask_keys, radius_fraction=0.5):
    '''
    Generate two panel figure comparing enclosing angle vs. radius for the true
    satellite distribution(s) in sat_list and the corresponding isotropic distribution(s).
    '''
    point_style_arr = np.array((['^', 'v', '<'],['+', 'x', '3'],['s','p','h']))
    cm = plt.cm.get_cmap('viridis')
    fig = plt.figure(figsize=(14,6))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1,1,0.05])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1, sharey=ax1)
    ax3 = fig.add_subplot(gs[2])
    ax = [ax1, ax2, ax3]

    for i, (sat, mask_key) in enumerate(zip(sat_list, mask_keys)):
        rad_angle_dict = sio.loop_hal(sat, mask_key, ang.open_angle_v_r, **{'threshold_fraction':radius_fraction, 'radius_bins':sat.r_bins, 'angle_bins':sat.a_bins})
        radii = sio.single_property_dict(rad_angle_dict, 'radii')
        angles = sio.single_property_dict(rad_angle_dict, 'angles')
        iso_rad_angle_dict = sio.loop_iso(sat, mask_key, iso.iso_open_angle_v_r, **{'threshold_fraction':radius_fraction, 'radius_bins':sat.r_bins, 'angle_bins':sat.a_bins})
        iso_radii = sio.single_property_dict(iso_rad_angle_dict, 'radii')
        iso_angles = sio.single_property_dict(iso_rad_angle_dict, 'angles')
        for j, hal_name in enumerate(rad_angle_dict.keys()):
            label_str = r'{}: $R^2$ = {:.3f}'.format(sat.hal_label[mask_key][j], np.corrcoef(radii[hal_name], angles[hal_name])[0,1]**2)
            pts = ax[0].scatter(radii[hal_name], angles[hal_name], c=sat.redshift, marker=point_style_arr[i, j], s=100, label=label_str, cmap=cm)
            iso_label_str = r'{}: $R^2$ = {:.3f}'.format(sat.hal_label[mask_key][j], np.corrcoef(iso_radii[hal_name], iso_angles[hal_name])[0,1]**2)
            iso_pts = ax[1].scatter(iso_radii[hal_name], iso_angles[hal_name], c=sat.redshift, marker=point_style_arr[i, j], s=100, label=iso_label_str, cmap=cm)
            ax[0].legend()
            ax[1].legend()

    cbar = fig.colorbar(pts, cax=ax[2], orientation='vertical')
    cbar.set_label('Redshift [z]')

    rlabel = int(radius_fraction*100)
    ax[0].set_xlabel('$R{}$ [kpc]'.format(rlabel))
    ax[1].set_xlabel('$R{}$ [kpc]'.format(rlabel))
    ax[0].set_ylabel('Enclosing angle [deg]')
    ax[1].set_title('Isotropic $N_{iter}=$'+str(sat_list[0].n_iter), fontsize=12)
    ax[0].set_title(' ', fontsize=12)
    plt.show()

def plot_coadd_angle_comparison(
    prop_bins,
    prop_list,
    iso_prop_list=None,
    iso_regions=False,
    halo_names='',
    plot_title='',
    xlabel='', 
    ylabel='',
    stat_type='mean'):
    '''
    Generate a plot of fraction of satellites enclosed vs. enclosing angle,
    which varies from [0, 180] degrees for a single sat object. Single row of
    subplots and number of columns matches number of host halos in prop_list.
    prop_list = [0=means or 1=medians or 2=percentiles][0=m12i or 1=m12f or 2=m12m]
    '''
    angle_ticks = [0, 30, 60, 90, 120, 150, 180]
    angle_labels = ['0', '30', '60', '90', '120', '150', '180']
    prop_length = len(prop_list.keys())
    fig, axes = plt.subplots(1, prop_length, sharex=True, sharey=True, figsize=(7*prop_length, 6))

    # make axes iterable if it is not already
    try:
        iter(axes)
    except TypeError:
        axes = np.array([axes])

    for i, (ax, hal_name) in enumerate(zip(axes, prop_list.keys())):
        color_i = CB_color_cycle[i]
        ax.fill_between(prop_bins, prop_list[hal_name]['percentile'][2], prop_list[hal_name]['percentile'][3], alpha=0.25,
            color=color_i, linewidth=0)
        ax.fill_between(prop_bins, prop_list[hal_name]['percentile'][0], prop_list[hal_name]['percentile'][1], alpha=0.4,
            color=color_i, linewidth=0)
        ax.plot(prop_bins, prop_list[hal_name][stat_type], color=color_i, label=halo_names[i])

        if iso_prop_list is not None:
            if iso_regions is True:
                ax.fill_between(prop_bins, iso_prop_list[hal_name]['percentile'][2], iso_prop_list[hal_name]['percentile'][3], alpha=0.25,
                    color='k', linewidth=0)
                ax.fill_between(prop_bins, iso_prop_list[hal_name]['percentile'][0], iso_prop_list[hal_name]['percentile'][1], alpha=0.4,
                    color='k', linewidth=0)
            ax.plot(prop_bins, iso_prop_list[hal_name][stat_type], color='k', label='isotropic')

        ax.set_xticks(angle_ticks)
        ax.set_xticklabels(angle_labels, fontsize=18)
        ax.set_xlabel(xlabel, fontsize=22)
        ax.legend(fontsize=18, loc='center right')

    axes[0].set_ylabel(ylabel, fontsize=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.show()

    return fig

def plot_enc_angle_vs_r_ratio(sat_list, mask_keys, angle_frac=0.68, rfrac1=0.9, rfrac2=0.1, ax_key ='moi'):
    '''
    Generate a two panel comparison plot of the angle enclosing angle_frac fraction
    of the satellites for a given measure of concentration, r1/r2, where r1 and
    r2 are input as fractions and calculated as radii enclosing their
    respective fractions of the satellite distribution.
    '''
    ps_arr = np.array((['^', 's', 'p'],['+', 'x', '3'],['s','p','h']))
    cm = plt.cm.get_cmap('viridis')
    fig = plt.figure(figsize=(14,6))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1,1,0.05])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1, sharey=ax1)
    plt.setp(ax2.get_yticklabels(), visible=False)
    ax3 = fig.add_subplot(gs[2])
    ax = [ax1, ax2, ax3]

    for i, (sat, mask_key) in enumerate(zip(sat_list, mask_keys)):
        r_ratio = sio.loop_hal(sat, mask_key, spa.rfrac_vs_rfrac, **{'frac1':rfrac1, 'frac2':rfrac2, 'r_bins':sat.r_bins})
        # experimental section
        #rms_dict = sio.loop_hal(sat, mask_key, spa.rms_vs_r_frac, **{'r_frac':angle_frac,'radius_bins':sat.r_bins})
        #r_ratio = sio.single_property_dict(rms_dict, 'rmsx')
        #r_ratio = sio.loop_hal(sat, mask_key, spa.r_fraction, **{'frac':0.5, 'radius_bins':sat.r_bins})
        
        if ax_key == 'moi':
            enc_angle = sio.loop_hal(sat, mask_key, ang.angle_width, **{'threshold_fraction':angle_frac, 'angle_bins':sat.a_bins})
            iso_enc_angle = sio.loop_iso(sat, mask_key, iso.iso_angle_width, **{'threshold_fraction':angle_frac, 'angle_bins':sat.a_bins})
        elif ax_key == 'rand':
            enc_angle = sio.loop_hal(sat, mask_key, ra.rand_angle_width, **{'n_iter':sat.n_iter, 'fraction':angle_frac, 'angle_range':sat.a_bins})
            enc_angle = sio.single_property_dict(enc_angle, 'angle')
            iso_enc_angle = sio.loop_iso(sat, mask_key, iso.iso_rand_angle_width, **{'threshold_fraction':angle_frac, 'angle_range':sat.a_bins, 'n_iter':sat.n_iter})
            iso_enc_angle = sio.single_property_dict(iso_enc_angle, 'angle')        
        for j, hal_name in enumerate(enc_angle.keys()):
            label_str = r'{}: $R^2$ = {:.3f}'.format(sat.hal_label[mask_key][j], np.corrcoef(r_ratio[hal_name], enc_angle[hal_name])[0,1]**2)
            pts = ax[0].scatter(r_ratio[hal_name], enc_angle[hal_name], c=sat.redshift, marker=ps_arr[i, j], s=100, label=label_str, cmap=cm)
            ax[0].legend()

            iso_label_str = r'{}: $R^2$ = {:.3f}'.format(sat.hal_label[mask_key][j], np.corrcoef(r_ratio[hal_name], iso_enc_angle[hal_name])[0,1]**2)
            pts = ax[1].scatter(r_ratio[hal_name], iso_enc_angle[hal_name], c=sat.redshift, marker=ps_arr[i, j], s=100, label=iso_label_str, cmap=cm)
            ax[1].legend()

    ax[2].tick_params(axis=u'both', which=u'both',length=0)
    cbar = fig.colorbar(pts, cax=ax[2], orientation='vertical')
    cbar.set_label('Redshift [z]')

    rfrac1_str = int(rfrac1*100)
    rfrac2_str = int(rfrac2*100)
    angfrac_str = int(angle_frac*100)
    ax[0].set_xlabel('$R{}$/$R{}$'.format(rfrac1_str, rfrac2_str))
    ax[1].set_xlabel('$R{}$/$R{}$'.format(rfrac1_str, rfrac2_str))
    ax[0].set_ylabel('Angle enclosing {}$\%$ of sats [deg]'.format(angfrac_str))
    ax[1].set_title('Isotropic $N_{iter}=$'+str(sat_list[0].n_iter), fontsize=12)
    ax[0].set_title(' ', fontsize=12)
    plt.show()

    return fig

def plot_enc_angle_vs_r_ratio2(sat_list, mask_keys, angle_frac=0.68, rfrac1=0.9, rfrac2=0.1, ax_key ='moi'):
    '''
    Generate a two panel comparison plot of the angle enclosing angle_frac fraction
    of the satellites for a given measure of concentration, r1/r2, where r1 and
    r2 are input as fractions and calculated as radii enclosing their
    respective fractions of the satellite distribution.
    '''
    ps_arr = np.array(['^', 's', 'p', 'h', '+', 'x', '3', 's','p','h'])
    colors = np.array([CB_color_cycle, ['darkgray','darkgray','darkgray']])
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(12,6))

    for i, (sat, mask_key) in enumerate(zip(sat_list, mask_keys)):
        r_ratio = sio.loop_hal(sat, mask_key, spa.rfrac_vs_rfrac, **{'frac1':rfrac1, 'frac2':rfrac2, 'r_bins':sat.r_bins})
        # experimental section
        #rms_dict = sio.loop_hal(sat, mask_key, spa.rms_vs_r_frac, **{'r_frac':angle_frac,'radius_bins':sat.r_bins})
        #r_ratio = sio.single_property_dict(rms_dict, 'rmsx')
        #r_ratio = sio.loop_hal(sat, mask_key, spa.r_fraction, **{'frac':0.5, 'radius_bins':sat.r_bins})
        
        if ax_key == 'moi':
            enc_angle = sio.loop_hal(sat, mask_key, ang.angle_width, **{'threshold_fraction':angle_frac, 'angle_bins':sat.a_bins})
            iso_enc_angle = sio.loop_iso(sat, mask_key, iso.iso_angle_width, **{'threshold_fraction':angle_frac, 'angle_bins':sat.a_bins})
        elif ax_key == 'rand':
            enc_angle = sio.loop_hal(sat, mask_key, ra.rand_angle_width, **{'n_iter':sat.n_iter, 'fraction':angle_frac, 'angle_range':sat.a_bins})
            enc_angle = sio.single_property_dict(enc_angle, 'angle')
            #iso_enc_angle = sio.loop_iso(sat, mask_key, iso.iso_rand_angle_width, **{'threshold_fraction':angle_frac, 'angle_range':sat.a_bins, 'n_iter':250})
            #iso_enc_angle = sio.single_property_dict(iso_enc_angle, 'angle')        
        for j, hal_name in enumerate(enc_angle.keys()):
            label_str = r'{}: $r$ = {:.3f}'.format(sat.hal_label[mask_key][j], np.corrcoef(r_ratio[hal_name], enc_angle[hal_name])[0,1])
            ax[0].plot(r_ratio[hal_name], enc_angle[hal_name], color=colors[i][j], alpha=0.8, marker=ps_arr[j], markeredgecolor='k', markeredgewidth=1, markersize=10, linestyle='None', label=label_str)
            ax[0].legend(fontsize=16)
            print(hal_name, np.mean(enc_angle[hal_name]))

            #iso_label_str = r'{}: $r$ = {:.3f}'.format(sat.hal_label[mask_key][j], np.corrcoef(r_ratio[hal_name], iso_enc_angle[hal_name])[0,1])
            #ax[1].plot(r_ratio[hal_name], iso_enc_angle[hal_name], color=colors[i][j], alpha=0.8, marker=ps_arr[j], markeredgecolor='k', markeredgewidth=1, markersize=10, linestyle='None', label=iso_label_str)
            #ax[1].legend(fontsize=16)

    rfrac1_str = int(rfrac1*100)
    rfrac2_str = int(rfrac2*100)
    angfrac_str = int(angle_frac*100)
    ax[0].set_xlabel('$R{}$/$R{}$'.format(rfrac1_str, rfrac2_str), fontsize=18)
    ax[1].set_xlabel('$R{}$/$R{}$'.format(rfrac1_str, rfrac2_str), fontsize=18)
    ax[0].set_ylabel('Enclosing angle of plane [deg]', fontsize=18)
    #ax[1].set_title('Isotropic $N_{iter}=$'+str(sat_list[0].n_iter), fontsize=14)
    #ax[0].set_title(' ', fontsize=12)

    plt.show()

    return fig

def plot_rand_enc_angle_vs_r_ratio(sat_list, mask_keys, angle_frac=0.68, rfrac1=0.9, rfrac2=0.1):
    '''
    Generate a two panel comparison plot of the angle enclosing angle_frac fraction
    of the satellites for a given measure of concentration, r1/r2, where r1 and
    r1 are input as fractions (and interpreted/calculated as radii enclosing their
    respective fractions of the satellite distribution.
    '''
    ps_arr = np.array((['^', 'v', '<'],['+', 'x', '3'],['s','p','h'],['.','.','.'],['.','.','.']))
    cm = plt.cm.get_cmap('viridis')
    fig = plt.figure(figsize=(14,6))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1,1,0.05])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    plt.setp(ax2.get_yticklabels(), visible=False)
    ax3 = fig.add_subplot(gs[2])
    ax = [ax1, ax2, ax3]
    #ax = [ax1, ax2]

    for i, (sat, mask_key) in enumerate(zip(sat_list, mask_keys)):
        r_ratio = sio.loop_hal(sat, mask_key, spa.rfrac_vs_rfrac, **{'frac1':rfrac1, 'frac2':rfrac2, 'r_bins':sat.r_bins})
        angle_width = sio.loop_hal(sat, mask_key, ra.rand_angle_width, **{'n_iter':sat.n_iter, 'threshold_fraction':angle_frac, 'angle_range':sat.a_bins})
        angle = sio.single_property_dict(angle_width, 'angle')
        iso_angle_width = sio.loop_iso(sat, mask_key, iso.iso_rand_angle_width, **{'n_iter':sat.n_iter, 'threshold_fraction':angle_frac, 'angle_range':sat.a_bins})
        iso_angle = sio.single_property_dict(iso_angle_width, 'angle')
    
        for j, hal_name in enumerate(angle_width.keys()):
            label_str = r'{}: $R^2$ = {:.3f}'.format(sat.hal_label[mask_key][i], np.corrcoef(r_ratio[hal_name], angle[hal_name])[0,1]**2)
            pts = ax[0].scatter(r_ratio[hal_name], angle[hal_name], c=sat.redshift, marker=ps_arr[i, j], s=100, label=label_str, cmap=cm)
            ax[0].legend()

            iso_label_str = r'{}: $R^2$ = {:.3f}'.format(sat.hal_label[mask_key][i], np.corrcoef(r_ratio[hal_name], iso_angle[hal_name])[0,1]**2)
            pts = ax[1].scatter(r_ratio[hal_name], iso_angle[hal_name], c=sat.redshift, marker=ps_arr[i, j], s=100, label=iso_label_str, cmap=cm)
            ax[1].legend()

    #ax[1].tick_params(axis=u'both', which=u'both',length=0)
    cbar = fig.colorbar(pts, cax=ax[2], orientation='vertical')
    cbar.set_label('Redshift [z]')

    rfrac1_str = int(rfrac1*100)
    rfrac2_str = int(rfrac2*100)
    angfrac_str = int(angle_frac*100)
    ax[0].set_xlabel('$R{}$/$R{}$'.format(rfrac1_str, rfrac2_str))
    ax[0].set_ylabel('Angle enclosing {}$\%$ of sats [deg]'.format(angfrac_str))
    ax[0].set_title(' ', fontsize=12)
    plt.show()

    return fig

def plot_rand_vs_moi_angle(sat, mask_key, fraction=0.68):
    '''
    PLot 68% angle widths for angles as measured by the MOI axes and also the
    min rand axes.
    '''
    angle_width = sio.loop_hal(sat, mask_key, ang.angle_width, **{'threshold_fraction':fraction, 'angle_bins':sat.a_bins})
    rand_angle_width = sio.loop_hal(sat, mask_key, ra.rand_angle_width, **{'n_iter':sat.n_iter, 'threshold_fraction':fraction, 'angle_range':sat.a_bins})
    rand_angle = sio.single_property_dict(rand_angle_width, 'angle')
    plt.figure()

    for i, hal_name in enumerate(angle_width.keys()):
        plt.plot(sat.redshift, angle_width[hal_name], color=CB_color_cycle[i], label=sat.hal_label[mask_key][i])
        plt.plot(sat.redshift, rand_angle[hal_name], '--', color=CB_color_cycle[i], label=sat.hal_label[mask_key][i]+': rand axes')

    plt.legend()
    plt.xlabel('Redshift [z]')
    plt.ylabel('68$\%$ enclosing angle')
    plt.show()

def plot_rand_axes_dotted(sat, mask_key):
    '''
    Plot the (absolute value of the) dot product of the min rand z axis enclosing
    a given fraction of satellites with the min rand z axis that encloses 100%
    of satellites (first plot) and then with the MOI minor axis (second plot).
    '''
    fracs = np.arange(0.1, 1.1, .1)

    angle_dict = sio.loop_hal(sat, mask_key, ang.open_angle, **{'return_vec':True})
    moi_axes = sio.single_property_dict(angle_dict, 'axis')

    rand_angle_dict = sio.loop_hal(sat, mask_key, ra.rand_frac_open_angle, **{'frac_bins':fracs, 'angle_range':sat.a_bins, 'n_iter':sat.n_iter})
    rand_min_axes = sio.single_property_dict(rand_angle_dict, 'axes')

    for i, hal_name in enumerate(rand_min_axes.keys()):
        fig = plt.figure(figsize=(8,8))

        for j, snap in enumerate(rand_min_axes[hal_name]):
            axes = rand_min_axes[hal_name][j]
            norm_max_angle = axes[-1][2]
            dots = [abs(np.dot(ax[2], norm_max_angle)) for ax in axes]
            plt.plot(fracs, dots, label=sat.hal_label[mask_key][i]+' z='+str(sat.redshift[j]))

        plt.xlabel('Fraction of sats enclosed')
        plt.ylabel('$\| {\hat{r}_{min}} \cdot {\hat{r}_{min1}} \|$')
        plt.legend()
        plt.show()
    
    for i, hal_name in enumerate(rand_min_axes.keys()):
        fig = plt.figure(figsize=(8,8))
        for j, snap in enumerate(rand_min_axes[hal_name]):
            dots = [abs(np.dot(ax[2], moi_axes[hal_name][j][2])) for ax in axes]
            plt.plot(fracs, dots, label=sat.hal_label[mask_key][i]+' z='+str(sat.redshift[j]))
        plt.xlabel('Fraction of sats enclosed')
        plt.ylabel('$\| {\hat{r}_{min}} \cdot {\hat{r}_{MOI}} \|$')
        plt.legend()
        plt.show()

def plot_rand_min_coadd_angle(sat, mask_key, stat_type='median'):
    '''
    Generate a plot of fraction of satellites enclosed vs. enclosing angle,
    which varies from [0, 180] degrees, wrt random axes for a single sat object.
    '''
    angle_range = sat.a_bins_plt

    angle_frac = sio.loop_hal(sat, mask_key, ang.fraction_open_angle, **{'angle_bins':sat.a_bins})
    coadd_angle_frac = mf.coadd_redshift(angle_frac, dict_key=None, coadd_axis=0)

    iso_angle_frac = sio.loop_iso(sat, mask_key, iso.iso_frac_open_angle, **{'angle_bins':sat.a_bins})
    coadd_iso_angle_frac = mf.coadd_redshift(iso_angle_frac, dict_key=None, coadd_axis=0)

    rand_angle_frac = sio.loop_hal(sat, mask_key, ra.rand_frac_open_angle, **{'frac_bins':np.arange(0,1.1,0.1), 'angle_range':sat.a_bins, 'n_iter':sat.n_iter})
    coadd_rand_angle_frac = mf.coadd_redshift(rand_angle_frac, dict_key='angle', coadd_axis=0)


    plot_coadd_angle_comparison(coadd_rand_angle_frac['m12f'][stat_type], np.arange(0,1.1,0.1), iso_prop_list=coadd_iso_angle_frac, halo_names=sat.hal_name,
        plot_title='Minimum enclosing angles for random axes', xlabel='Enclosing angle [deg]', ylabel='Fraction of satellites enclosed',
        stat_type=stat_type)
    # colored lines = random min axes, black lines = isotropic distribution with MOI axes
    #plot_coadd_angle_comparison(angle_range, coadd_rand_angle_frac, iso_prop_list=coadd_iso_angle_frac, halo_names=sat.hal_name,
    #    plot_title='Minimum enclosing angles for random axes', xlabel='Enclosing angle [deg]', ylabel='Fraction of satellites enclosed',
    #    stat_type=stat_type)

    # colored lines = MOI axes, black lines = random min axes
    #plot_coadd_angle_comparison(angle_range, coadd_angle_frac, iso_prop_list=coadd_rand_angle_frac, halo_names=sat.hal_name,
    #    plot_title='Minimum enclosing angles for random axes', xlabel='Enclosing angle [deg]', ylabel='Fraction of satellites enclosed',
    #    stat_type=stat_type)
