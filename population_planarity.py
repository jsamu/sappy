import numpy as np
import os
import itertools
import pandas as pd
from matplotlib import cm
from matplotlib import pyplot as plt
from scipy.stats import spearmanr
from collections import defaultdict
import satellite_analysis as sa


########################
### helper functions ###
########################

def color_cycle(cycle_length=14, cmap_name='plasma', low=0, high=1):
    cm_subsection = np.linspace(low, high, cycle_length)
    cmap=plt.get_cmap(cmap_name)
    colors = [cmap(x) for x in cm_subsection]

    return colors
    
def plot_coadd(
    x_values, host_medians, host_percentiles, own_figs=False, color=False):
    if color:
        colors = color_cycle(len(host_medians))
    else:
        colors = ['k' for k in range(len(host_medians))]
    for i,(host_median,host_percentile) in enumerate(zip(host_medians, host_percentiles)):
        if own_figs is True:
            plt.figure(figsize=(7,6))
        _err = np.array(host_median-host_percentile[0], host_percentile[1]-host_median)
        plt.errorbar(x_values[i], host_median, _err, capsize=0, color=colors[i], alpha=0.6)
        plt.plot(np.array(x_values[i]), np.array(host_median), '.', color=colors[i])

def host_probability_stats(grouped_table, y_type, return_dict=False):
    if return_dict is True:
        median_probs = defaultdict(list)
        percentile_probs = defaultdict(list)

        for i,(host_key,host_group) in enumerate(grouped_table):
            median_probs[host_key].append(host_group[y_type].median())
            percentile_probs[host_key].append(np.nanpercentile(host_group[y_type], [16, 84, 2.5, 97.5]))
        return median_probs, percentile_probs
    else:
        median_probs = []
        percentile_probs = []

        for i,(host_key,host_group) in enumerate(grouped_table):
            median_probs.append(host_group[y_type].median())
            percentile_probs.append(np.nanpercentile(host_group[y_type], [16, 84, 2.5, 97.5]))

        return np.array(median_probs), np.array(percentile_probs)

######################
### time evolution ###
######################

def plot_value_vs_time(
    grouped_table, y_type, x_type='redshift', cmap_name='plasma'):
    colors = color_cycle(len(grouped_table), cmap_name=cmap_name)
    for i,(host_key,host_group) in enumerate(grouped_table):
        plt.figure(figsize=(7,6))
        plt.plot(host_group[x_type], host_group[y_type], color=colors[i], label=host_key)
        plt.legend(loc=1)
        plt.xlabel(x_type)
        plt.ylabel(y_type)

def plot_probability_vs_time(
    grouped_table, y_type, x_type='redshift', cmap_name='plasma'):
    colors = color_cycle(len(grouped_table), cmap_name=cmap_name)

    for i,(host_key,host_group) in enumerate(grouped_table):
        plt.figure(figsize=(7,6))
        plt.plot(host_group[x_type], host_group[y_type], color=colors[i], label=host_key)
        plt.legend(loc=1)
        plt.xlabel('redshift [z]')
        plt.ylabel('prob of greater isotropic planarity')
        #plt.xlim((0,0.1))

def histogram_planar_intervals2(
    grouped_table_list, y_type_list, threshold_value_list, x_type='time', 
    exclude_single_snapshot=False, t_bin_width=0.25, 
    color_list=['C0', 'C1', 'C2', 'k'], 
    histtype_list = ['bar', 'bar', 'bar', 'step'], legend_title=None):

    fig, ax = plt.subplots(1,1,figsize=(6,5))
    fig.set_tight_layout(False)
    fig.subplots_adjust(left=0.14, right=0.98, top=0.98, bottom=0.16)
    t_hist_list = []
    t_bin_list = []
    t_hist_dict = {}
    t_bin_dict = {}
    for grouped_table, y_type, threshold_value in zip(
        grouped_table_list, y_type_list, threshold_value_list):
        
        print(y_type)

        # finding t_corr intervals, lengths of time that planarity falls
        # below a set threshold
        t_corr = {}
        for i,(host_key,host_group) in enumerate(grouped_table):      
            below_thresh_indices = np.where(host_group[y_type] <= threshold_value)[0]

            times_below_threshold = np.full(len(host_group[y_type]), -1.0)
            for j in below_thresh_indices:
                times_below_threshold[j] = host_group[x_type].values[j]

            deltat = []
            t_below = []

            for n,ti in enumerate(times_below_threshold):
                # if the time isn't a nan, append it to a new array
                if ti >= 0.0:
                    t_below.append(ti)
                    # if you're at the end of the list, get time interval now
                    if (n == len(times_below_threshold) - 1) & (len(t_below) > 1):
                        deltat.append(t_below[0] - t_below[-1])
                    elif (n == len(times_below_threshold) - 1) & (len(t_below) == 1):
                        deltat.append(0.0)
                else:
                    # if you've only hit -1's thus far, keep going
                    if np.sum(times_below_threshold[:n+1] < 0) == n+1:
                        pass
                    # if you hit a nan after a series of numbers, take the time 
                    # interval between the previous ti and the earliest consecutive 
                    # ti that is not a nan and append it to deltat
                    elif (times_below_threshold[n-1] > 0) & (times_below_threshold[n-2] > 0):
                        deltat.append(t_below[0] - t_below[-1])
                        # then reset t_below to an empty list to find new time intervals
                        t_below = []
                    elif (times_below_threshold[n-1] > 0) & (times_below_threshold[n-2] < 0):
                        deltat.append(0.0)
                        # then reset t_below to an empty list to find new time intervals
                        t_below = []
                    else:
                        pass

            # catch hosts that are always below threshold probability
            if (len(deltat) == 0) & (len(t_below) == np.sum(~np.isnan(times_below_threshold))):
                deltat.append(t_below[0] - t_below[-1])
            
            deltat = np.array(deltat)
            
            # excludes single-snapshot planarity
            if exclude_single_snapshot is True:
                t_corr[host_key] = deltat[deltat > 0]
            else:
                t_corr[host_key] = deltat


        all_t_corr = []
        hosts_with_planarity = []
        for host in t_corr.keys():
            all_t_corr = np.concatenate((all_t_corr, t_corr[host]))
            if len(t_corr[host]) > 0:
                #print(host, ': intervals of planarity in Gyr =', t_corr[host])
                print(host, ': max interval of planarity in Gyr =', np.max(t_corr[host]))
                hosts_with_planarity.append(host)
            #print(host, 'significant intervals of planarity in Gyr =', t_corr[host][t_corr[host] >= 1])

        print('total number of hosts with planarity:', len(hosts_with_planarity), hosts_with_planarity)
        print('total number of time intervals with planarity:', len(all_t_corr))    
        try:
            print('min time interval =', np.min(all_t_corr), 'max time interval =', np.max(all_t_corr))
            t_bins = np.arange(0,np.max(all_t_corr)+t_bin_width,t_bin_width)
            t_hist, tb = np.histogram(all_t_corr, bins=t_bins)
            t_hist_list.append(t_hist)
            t_bin_list.append(tb)
            t_hist_dict[y_type] = t_hist
            t_bin_dict[y_type] = tb
            print('instances of >1 gyr:', np.sum(all_t_corr > 1))
        except:
            #pass
            t_hist_list.append([np.nan, np.nan])
            t_bin_list.append([np.nan, np.nan])

        print('\n')

    prop_labels = {'rms.min':'RMS height', 'axis.ratio':'Axis ratio', 
                'opening.angle':'Opening angle', 'orbital.pole.dispersion':'Orbital dispersion',
                'coherent.frac':'Coherent velocity fraction'}

    t_hist_max = np.nanmax(list(itertools.chain.from_iterable(t_hist_list)))
    for t_h,t_b, c, y_key in zip(t_hist_list, t_bin_list, color_list, y_type_list):
        t_h_ = t_h/t_hist_max
        if 'orb' in y_key:
            plt.bar(t_b[:-1], t_h_, align='edge', color="none", alpha=1, 
                    linewidth=3, label=prop_labels[y_key], width=t_bin_width,
                    edgecolor='k')
        elif 'coherent' in y_key:
            plt.bar(t_b[:-1], t_h_, align='edge', color="none", alpha=1, 
                    linewidth=3, label=prop_labels[y_key], width=t_bin_width,
                    edgecolor='k')
        else:
            plt.bar(t_b[:-1], t_h_, align='edge', color=c, alpha=0.55, 
                    linewidth=3, label=prop_labels[y_key], width=t_bin_width)
        
    plt.legend(loc='upper right', title_fontsize=18, title=legend_title)
    plt.xlabel(r'$\Delta t_{plane}$ [Gyr]', fontsize=20)
    #plt.ylabel('Frequency', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    #ax.set_xticks([0,1,2,3,4,5])
    #ax.set_xticks([k for k in range(int(np.max(all_t_corr)))])
    #ax.set_xticklabels([str(k) for k in range(int(np.max(all_t_corr)))])
    plt.show()

    return fig

def histogram_planar_intervals(
    grouped_table, y_type, x_type='time', exclude_single_snapshot=True,
    threshold_probability=0.5, t_bin_width=0.25):
    # finding t_corr intervals, lengths of time that planar probability falls
    # below some set threshold
    t_corr = {}
    for i,(host_key,host_group) in enumerate(grouped_table):      
        below_thresh_indices = np.where(host_group[y_type] < threshold_probability)[0]
        #times_below_threshold = np.zeros(len(host_group[y_type]))
        times_below_threshold = np.full(len(host_group[y_type]), -1.0)
        for j in below_thresh_indices:
            times_below_threshold[j] = host_group[x_type].values[j]

        deltat = []
        t_below = []
        '''
        for n,ti in enumerate(times_below_threshold):
            if ti != 0:
                t_below.append(ti)
            elif ti == 0:
                # once you hit a 0, take the time interval
                if (times_below_threshold[n-1] != 0) & (n != 0):
                    deltat.append(t_below[0] - t_below[-1])
                # restarts/keeps list empty while looping over zeros
                else:
                    t_below = []
        '''

        for n,ti in enumerate(times_below_threshold):
            # if the time isn't a nan, append it to a new array
            if ti >= 0.0:
                t_below.append(ti)
                # if you're at the end of the list, get time interval now
                if (n == len(times_below_threshold) - 1) & (len(t_below) > 1):
                    deltat.append(t_below[0] - t_below[-1])
                elif (n == len(times_below_threshold) - 1) & (len(t_below) == 1):
                    deltat.append(0.0)
            else:
                # if you've only hit -1's thus far, keep going
                if np.sum(times_below_threshold[:n+1] < 0) == n+1:
                    pass
                # if you hit a nan after a series of numbers, take the time 
                # interval between the previous ti and the earliest consecutive 
                # ti that is not a nan and append it to deltat
                elif (times_below_threshold[n-1] > 0) & (times_below_threshold[n-2] > 0):
                    deltat.append(t_below[0] - t_below[-1])
                    # then reset t_below to an empty list to find new time intervals
                    t_below = []
                elif (times_below_threshold[n-1] > 0) & (times_below_threshold[n-2] < 0):
                    deltat.append(0.0)
                    # then reset t_below to an empty list to find new time intervals
                    t_below = []
                else:
                    pass
                '''
                # if you're at the end of the list, get time interval now
                elif (n == len(times_below_threshold) - 1) & (len(t_below) > 1):
                    deltat.append(t_below[0] - t_below[-1])
                elif (n == len(times_below_threshold) - 1) & (len(t_below) == 1):
                    deltat.append(0.0)
                elif (n == len(times_below_threshold) - 1) & (len(t_below) == 0):
                    pass
                '''

        # catch hosts that are always below threshold probability
        if (len(deltat) == 0) & (len(t_below) == np.sum(~np.isnan(times_below_threshold))):
            deltat.append(t_below[0] - t_below[-1])
        
        deltat = np.array(deltat)
        
        # excludes single-snapshot planarity
        if exclude_single_snapshot is True:
            t_corr[host_key] = deltat[deltat > 0]
        else:
            t_corr[host_key] = deltat


    all_t_corr = []
    for host in t_corr.keys():
        all_t_corr = np.concatenate((all_t_corr, t_corr[host]))
        print(host, ': intervals of planarity in Gyr =', t_corr[host])
        #print(host, 'significant intervals of planarity in Gyr =', t_corr[host][t_corr[host] >= 1])

    print('total number of time intervals with planarity:', len(all_t_corr))    
    print('min time interval =', np.min(all_t_corr), 'max time interval =', np.max(all_t_corr))

    t_bins = np.arange(0,np.max(all_t_corr)+t_bin_width,t_bin_width)
    fig, ax = plt.subplots(1,1,figsize=(6,5))
    plt.hist(all_t_corr, bins=t_bins)
    plt.xlabel(r'$\Delta t_{plane}$ [Gyr]', fontsize=20)
    plt.ylabel('Frequency', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xticks([k for k in range(6)])
    ax.set_xticklabels([str(k) for k in range(6)])
    plt.show()

def dot_poles_vs_time(
    group1, group2, key1='min.axis', key2='avg.orbital.pole', subsample=None):
    for ((host_key1,host_group1), (host_key2,host_group2)) in zip(group1, group2):
        assert host_key1 == host_key2
        group1_pole = np.dstack([host_group1[key1+'.x'], host_group1[key1+'.y'], host_group1[key1+'.z']])[0]
        group1_dot = np.array([np.dot(group1_pole_i, group1_pole[0])/(np.linalg.norm(group1_pole_i)*np.linalg.norm(group1_pole[0]))
                            for group1_pole_i in group1_pole])
        group1_dot_angle = np.degrees(np.arccos(np.abs(group1_dot)))
        
        
        group2_pole = np.dstack([host_group2[key2+'.x'], host_group2[key2+'.y'], host_group2[key2+'.z']])[0]
        group2_dot = np.array([np.dot(group2_pole_i, group2_pole[0])/(np.linalg.norm(group2_pole_i)*np.linalg.norm(group2_pole[0]))
                            for group2_pole_i in group2_pole])
        group2_dot_angle = np.degrees(np.arccos(np.abs(group2_dot)))
        
        
        group12_dot = np.array([np.dot(pole1, pole2)/(np.linalg.norm(pole1)*np.linalg.norm(pole2))
                                for pole1, pole2 in zip(group1_pole, group2_pole)])
        group12_dot_angle = np.degrees(np.arccos(np.abs(group12_dot)))


        redshifts_ = np.array(host_group1['redshift'])
        if subsample is not None:
            sub_sample = np.arange(0, len(redshifts_), subsample)
        else:
            sub_sample = np.arange(0, len(redshifts_), 1)
            
        #if host_key1 in ['m12f', 'm12m']:

        plt.figure(figsize=(9,7))
        plt.plot(redshifts_[sub_sample], np.full_like(redshifts_, 45)[sub_sample], 
                 color='k', alpha=0.4)
        plt.plot(redshifts_[sub_sample], group1_dot_angle[sub_sample], color='b', 
                 label=r'$\hat{n}_{1} \cdot \hat{n}_{1, z=0}$')
        plt.plot(redshifts_[sub_sample], group2_dot_angle[sub_sample], color='r',
                 label=r'$\hat{n}_{2} \cdot \hat{n}_{2, z=0}$')
        plt.plot(redshifts_[sub_sample], group12_dot_angle[sub_sample], color='g', linestyle=':',
                 label=r'$\hat{n}_{1} \cdot \hat{n}_{2}$')

        plt.legend(title=host_key1+'  1='+key1+'  2='+key2, loc=1)
        plt.xlabel('Redshift [z]')
        plt.ylabel('Angle btwn plane normals [deg]')
        plt.ylim((0,90))
        plt.show()

def plane_prob_and_dot_poles_vs_time(
    group1, group2, key1='min.axis', key2='avg.orbital.pole', 
    key3='isotropic.probability.axis.ratio',
    key4='orbital.pole.dispersion.isotropic.probability', 
    subsample=None, lmc_data=None):
    for ((host_key1,host_group1), (host_key2,host_group2)) in zip(group1, group2):
        assert host_key1 == host_key2
        group1_pole = np.dstack([host_group1[key1+'.x'], host_group1[key1+'.y'], host_group1[key1+'.z']])[0]
        group1_dot = np.array([np.dot(group1_pole_i, group1_pole[0])/(np.linalg.norm(group1_pole_i)*np.linalg.norm(group1_pole[0]))
                            for group1_pole_i in group1_pole])
        group1_dot_angle = np.degrees(np.arccos(np.abs(group1_dot)))
        
        
        group2_pole = np.dstack([host_group2[key2+'.x'], host_group2[key2+'.y'], host_group2[key2+'.z']])[0]
        group2_dot = np.array([np.dot(group2_pole_i, group2_pole[0])/(np.linalg.norm(group2_pole_i)*np.linalg.norm(group2_pole[0]))
                            for group2_pole_i in group2_pole])
        group2_dot_angle = np.degrees(np.arccos(np.abs(group2_dot)))
        
        
        group12_dot = np.array([np.dot(pole1, pole2)/(np.linalg.norm(pole1)*np.linalg.norm(pole2))
                                for pole1, pole2 in zip(group1_pole, group2_pole)])
        group12_dot_angle = np.degrees(np.arccos(np.abs(group12_dot)))


        redshifts_ = np.array(host_group1['redshift'])
        if subsample is not None:
            sub_sample = np.arange(0, len(redshifts_), subsample)
        else:
            sub_sample = np.arange(0, len(redshifts_), 1)
            
            
        #if host_key1 in ['m12f', 'm12m']:

        fig, ax1 = plt.subplots(figsize=(9,7))

        # plot LMC pericenter passages
        if lmc_data is not None:
            first_passage = lmc_data['nth passage'] == 1
            host_name = lmc_data['host'] == host_key1
            lmc_peri_passages = lmc_data['redshift'][host_name & first_passage].values
            if len(lmc_peri_passages) > 0:
                ax1.vlines(lmc_peri_passages, 0, 90, alpha='0.7', linestyles='--')

        color = 'darkblue'#'tab:blue'
        ax1.set_xlabel('Redshift [z]')
        ax1.set_ylabel('Angle btwn plane normals [deg]', color=color)

        ax1.plot(redshifts_[sub_sample], np.full_like(redshifts_, 45)[sub_sample], 
                    color='k', alpha=0.5, linestyle=':')
        ax1.plot(redshifts_[sub_sample], group1_dot_angle[sub_sample], color=color, alpha=0.8,
                    label=r'$\hat{n}_{MOI} \cdot \hat{n}_{MOI, z=0}$')
        #ax1.plot(redshifts_[sub_sample], group2_dot_angle[sub_sample], color=color, linestyle='--', alpha=0.8,
        #         label='$\hat{n}_{orb} \cdot \hat{n}_{orb, z=0}$')
        ax1.plot(redshifts_[sub_sample], group12_dot_angle[sub_sample], color=color, linestyle='-.', alpha=0.8,
                    label=r'$\hat{n}_{MOI} \cdot \hat{n}_{orb}$')

        ax1.set_ylim((0,90))
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.legend(title=host_key1, loc=2)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        color = 'maroon'
        ax2.set_ylabel('Plane probability metric', color=color)
        ax2.plot(redshifts_[sub_sample], 100*host_group1[key3].values[sub_sample], color=color, linestyle='-',
                label='axis ratio', alpha=0.8)
        ax2.plot(redshifts_[sub_sample], host_group2[key4].values[sub_sample], color=color, linestyle='--',
                label='orbital pole dispersion', alpha=0.8)
        ax2.tick_params(axis='y', labelcolor=color)

        #ax2.set_ylim((0,1))
        ax2.legend(loc='upper left', bbox_to_anchor=(0, 0.8))

        #plt.legend(title=host_key1+'  1='+key1+'  2='+key2, loc=1)
        plt.xlim((0, 0.5))
        plt.show()

####################
### group infall ###
####################

def sat_first_infall(sat_infall_dict, time_table, verbose=False):
    infall_times = {}

    for host in sat_infall_dict.keys():    
        # get first infall snapshot
        sat_infall_snaps = []
        sat_infall_times = []
        for sat_distance in np.reshape(sat_infall_dict[host][0]['host.distance.total'].T, 
        (len(sat_infall_dict[host][0]['host.distance.total'][0]),600)):
            infall_mask = sat_distance < sat_infall_dict[host][0]['host.radius']
            sat_infall_snaps.append(np.nanmin(sat_infall_dict[host][0]['snapshot'][infall_mask]))
            sat_infall_times.append(np.nanmin(time_table['time[Gyr]'][1::][infall_mask]))

        infall_times[host] = sat_infall_times

    if verbose is True:
        for host in infall_times.keys():
            print(host, np.median(infall_times[host]), np.std(infall_times[host]))
        
    return infall_times

def group_size(sim_infall_dict, time_table, snapshot_list=np.arange(0,600,1)):
    first_sat_infall = sat_first_infall(sim_infall_dict, time_table)
    
    group_size_dict = {}
    sat_group_size_dict = {}
    
    for host in sim_infall_dict.keys():
        # get relevant quantities from the simulation's satellite dictionary
        sim_infall_history = sim_infall_dict[host][0]
        group_central_track = sim_infall_history['central.index']
        central_mass_track = sim_infall_history['central.mass']
        group_size_central_track = sim_infall_history['central.group.size']
        #group_size_central_star_track = sim_infall_history['central.group.size.star']

        main_host_track = sim_infall_history['host.index']
        main_host_radius_track = sim_infall_history['host.index']

        satellite_track = sim_infall_history['tree.index']
        satellite_mass_track = sim_infall_history['mass']

        # set up secondary tracking arrays
        group_history = defaultdict(list)
        sat_group_size_history = np.full(satellite_track.shape, np.nan)
        group_size_central_history = np.zeros(satellite_track.shape)

        for snap,time in zip(snapshot_list, time_table['time[Gyr]'][1::]):
            for i, group_host in enumerate(group_central_track[snap]):
                if time < first_sat_infall[host][i]:
                    if group_host > 0:
                        # exclude snaps where the host is the main host or the satellite itself (maybe overkill?)
                        if (group_host != main_host_track[snap]):# & (group_host != satellite_track[snap][i]):
                            group_history['snapshot'].append(snap+1)
                            host_in_sats = np.sum(satellite_track[snap] == group_host)
                            num_sats_in_group = np.sum(group_central_track[snap] == group_host)
                            sat_group_size_history[snap][i] = host_in_sats + num_sats_in_group
                            group_size_central_history[snap][i] = group_size_central_track[snap][i]
                            
        group_size_dict[host] = np.nanmax(group_size_central_history, axis=0)
        sat_group_size_dict[host] = np.nanmax(sat_group_size_history, axis=0)
        
    return group_size_dict, sat_group_size_dict

def frac_in_groups(sat_infall_dict, time_table):
    tot_group_size, sat_group_size = group_size(sat_infall_dict, time_table, snapshot_list=np.arange(0,600,1))
    group_fractions = {}
    largest_group_fractions = {}
    largest_sat_group_fractions = {}
    for host, sat_host in zip(tot_group_size.keys(), sat_group_size.keys()):
        groupsizes = tot_group_size[host]
        sat_groupsizes = sat_group_size[sat_host]
        
        group_frac = np.sum(groupsizes>0)/len(groupsizes)
        group_fractions[host] = group_frac
        
        group_frac_1_sat = np.sum(groupsizes == 1)/len(groupsizes)
        group_frac_2_sat = np.sum(groupsizes == 2)/len(groupsizes)
        group_frac_3_sat = np.sum(groupsizes == 3)/len(groupsizes)
        group_frac_4plus_sat = np.sum(groupsizes > 3)/len(groupsizes)
        
        if group_frac_4plus_sat > 0:
            largest_group_fractions[host] = group_frac_4plus_sat
        elif group_frac_3_sat > 0:
            largest_group_fractions[host] = group_frac_3_sat
        elif group_frac_2_sat > 0:
            largest_group_fractions[host] = group_frac_2_sat
        elif group_frac_1_sat > 0:
            largest_group_fractions[host] = group_frac_1_sat
        else:
            largest_group_fractions[host] = 0
            

        sat_group_frac_1_sat = np.sum(sat_groupsizes == 1)/len(sat_groupsizes)
        sat_group_frac_2_sat = np.sum(sat_groupsizes == 2)/len(sat_groupsizes)
        sat_group_frac_3_sat = np.sum(sat_groupsizes == 3)/len(sat_groupsizes)
        sat_group_frac_4plus_sat = np.sum(sat_groupsizes > 3)/len(sat_groupsizes)
        
        if sat_group_frac_4plus_sat > 0:
            largest_sat_group_fractions[sat_host] = sat_group_frac_4plus_sat
        elif sat_group_frac_3_sat > 0:
            largest_sat_group_fractions[sat_host] = sat_group_frac_3_sat
        elif sat_group_frac_2_sat > 0:
            largest_sat_group_fractions[sat_host] = sat_group_frac_2_sat
        elif sat_group_frac_1_sat > 0:
            largest_sat_group_fractions[sat_host] = sat_group_frac_1_sat
        else:
            largest_sat_group_fractions[sat_host] = 0

            
    return {'largest.group.fracs':largest_group_fractions, 
            'largest.sat.group.fracs':largest_sat_group_fractions,
            'group.fracs':group_fractions}

####################################
### correlations with host, etc. ###
####################################

def host_mass_correlation(
    host_table, grouped_table, y_type, dmo_grouped_table=None, y_label=None,
    redshift_limit=0.2, mass_kind='star.mass', loc=None):
    #median_probs, percentile_probs = host_probability_stats(grouped_table, y_type)
    #if dmo_grouped_table:
    #    dmo_median_probs, dmo_percentile_probs = host_probability_stats(dmo_grouped_table, y_type)
    
    #prob vs host stellar mass
    f1 = plt.figure(figsize=(6.5,5))
    f1.set_tight_layout(False)
    f1.subplots_adjust(left=0.13, right=0.96, top=0.98, bottom=0.16)
    group_host_mass = []
    medians = []
    median_dict = {}
    err_dict = {}
    for host, (group_host, group_metrics) in zip(host_table['host'], grouped_table):
        #assert host == group_host
        redshift_mask = group_metrics['redshift'] <= redshift_limit
        median = np.nanmedian(group_metrics[y_type][redshift_mask])
        medians.append(median)
        median_dict[group_host] = (median)
        percentile_ = np.reshape(np.nanpercentile(group_metrics[y_type][redshift_mask], [16, 84]), (2,1))
        _err = np.array([median-percentile_[0], percentile_[1]-median])
        err_dict[group_host] = _err
        group_host_mass.append(np.array(host_table['star.mass'][host_table['host'] == group_host])[0])
        plt.errorbar(np.array(host_table['star.mass'][host_table['host'] == group_host])/1e10, 
            median, _err, color='k', alpha=0.6)
        cb1 = plt.scatter(np.array(host_table['star.mass'][host_table['host'] == group_host])/1e10, 
            median_dict[group_host], 
            c=host_table['m.200m'][host_table['host'] == group_host]/1e12, 
            s=144, marker='o', cmap=cm.plasma, vmin=0.8, vmax=2.2, edgecolors='k')
        """
        if group_host in ['Romeo', 'Juliet', 'Thelma', 'Louise', 'Romulus', 'Remus']:
            if group_host == 'Romeo':
                plt.plot(np.array(host_table['star.mass'][host_table['host'] == group_host])/1e10, 
                    median, 'o', color='r', markeredgecolor='k', alpha=0.8, label='paired host')
            else:
                plt.plot(np.array(host_table['star.mass'][host_table['host'] == group_host])/1e10, 
                    median, 'o', color='r', markeredgecolor='k', alpha=0.8)
        else:
            if group_host == 'm12i':
                plt.plot(np.array(host_table['star.mass'][host_table['host'] == group_host])/1e10, 
                    median, '^', color='b', markeredgecolor='k', alpha=0.8, label='isolated host')
            else:
                plt.plot(np.array(host_table['star.mass'][host_table['host'] == group_host])/1e10, 
                    median, '^', color='b', markeredgecolor='k', alpha=0.8)
        """
    cb_1 = f1.colorbar(cb1, ticks=[1.0,1.5,2.0])
    cb_1.set_label(label=r'Host M$_{\rm 200m}$ [$10^{12}$ M$_{\odot}$]', fontsize=20)
    cb_1.ax.tick_params(labelsize=20)
    plt.xlabel(r'Host M$_*$ [$10^{10}$ M$_{\odot}$]', fontsize=20)
    if y_label is None:
        plt.ylabel('prob of greater isotropic planarity', fontsize=20)
    else:
        plt.ylabel(y_label, fontsize=20)
    #plt.legend(ncol=3, loc=1)
    plt.legend(loc=loc, fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=20)

    print('stellar mass pearson correlation coefficient = ', np.corrcoef(group_host_mass, medians)[0][1])
    print('stellar mass spearman correlation coefficient = ', spearmanr(group_host_mass, medians))

    #prob vs host 200m
    f2 = plt.figure(figsize=(6,5))
    f2.set_tight_layout(False)
    f2.subplots_adjust(left=0.13, right=0.96, top=0.98, bottom=0.16)
    group_host_mass_200m = []
    for host, (group_host, group_metrics) in zip(host_table['host'], grouped_table):
        group_host_mass_200m.append(np.array(host_table['m.200m'][host_table['host'] == group_host])[0])
        plt.errorbar(np.array(host_table['m.200m'][host_table['host'] == group_host])/1e12, 
            median_dict[group_host], err_dict[group_host], color='k', alpha=0.6)
        print(np.array(host_table['m.200m'][host_table['host'] == group_host])/1e12, 
            median_dict[group_host], host_table['star.mass'][host_table['host'] == group_host]/1e10)
        cb = plt.scatter(np.array(host_table['m.200m'][host_table['host'] == group_host])/1e12, 
            median_dict[group_host], 
            c=host_table['star.mass'][host_table['host'] == group_host]/1e10, 
            s=144, marker='o', cmap=cm.plasma, vmin=1, vmax=10, edgecolors='k')
        """
        if group_host in ['Romeo', 'Juliet', 'Thelma', 'Louise', 'Romulus', 'Remus']:
            if group_host == 'Romeo':
                plt.plot(np.array(host_table['m.200m'][host_table['host'] == group_host])/1e12, 
                    median_dict[group_host], 'o', color='r', markeredgecolor='k', alpha=0.8, 
                    label='paired host')
            else:
                plt.plot(np.array(host_table['m.200m'][host_table['host'] == group_host])/1e12, 
                    median_dict[group_host], 'o', color='r', markeredgecolor='k', alpha=0.8)
        else:
            if group_host == 'm12i':
                plt.plot(np.array(host_table['m.200m'][host_table['host'] == group_host])/1e12, 
                    median_dict[group_host], '^', color='b', markeredgecolor='k', alpha=0.8, 
                    label='isolated host')
            else:
                plt.plot(np.array(host_table['m.200m'][host_table['host'] == group_host])/1e12, 
                    median_dict[group_host], '^', color='b', markeredgecolor='k', alpha=0.8)
        """
    cb_= f2.colorbar(cb)
    cb_.set_label(label=r'Host M$_*$ [$10^{10}$ M$_{\odot}$]', fontsize=20)
    cb_.ax.tick_params(labelsize=20)
    plt.xlabel(r'Host M$_{\rm 200m}$ [$10^{12}$ M$_{\odot}$]', fontsize=20)
    if y_label is None:
        plt.ylabel('prob of greater isotropic planarity', fontsize=20)
    else:
        plt.ylabel(y_label, fontsize=20)
    #plt.legend(ncol=3, loc=1)
    plt.legend(loc=loc, fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=20)

    print('halo mass pearson correlation coefficient = ', np.corrcoef(group_host_mass_200m, medians)[0][1])
    print('halo mass spearman correlation coefficient = ', spearmanr(group_host_mass_200m, medians))

    #prob vs host m*/200m
    f3 = plt.figure(figsize=(6,5))
    mass_ratios = []
    for host_sm, host_hm, (group_host, group_metrics) in zip(group_host_mass, group_host_mass_200m, grouped_table):
        mass_ratios.append(host_sm/host_hm)
        plt.errorbar(host_sm/host_hm, median_dict[group_host], err_dict[group_host], color='k', alpha=0.6)
        if group_host in ['Romeo', 'Juliet', 'Thelma', 'Louise', 'Romulus', 'Remus']:
            if group_host == 'Romeo':
                plt.plot(host_sm/host_hm, median_dict[group_host], 'o', color='r', 
                        markeredgecolor='k', alpha=0.8, label='paired host')
            else:
                plt.plot(host_sm/host_hm, median_dict[group_host], 'o', color='r', 
                        markeredgecolor='k', alpha=0.8)
        else:
            if group_host == 'm12i':
                plt.plot(host_sm/host_hm, median_dict[group_host], '^', color='b', 
                        markeredgecolor='k', alpha=0.8, label='isolated host')
            else:
                plt.plot(host_sm/host_hm, median_dict[group_host], '^', color='b', 
                        markeredgecolor='k', alpha=0.8)

    plt.xlabel(r'Host M$_*$/M$_{\rm 200m}$', fontsize=20)
    if y_label is None:
        plt.ylabel('prob of greater isotropic planarity', fontsize=20)
    else:
        plt.ylabel(y_label, fontsize=20)
    #plt.legend(ncol=3, loc=1)
    plt.legend(loc=loc, fontsize=18)
    plt.tick_params(axis='both', which='major', labelsize=20)

    print('sm/hm pearson correlation coefficient = ', np.corrcoef(mass_ratios, medians)[0][1])
    print('sm/hm spearman correlation coefficient = ', spearmanr(mass_ratios, medians))

    return f1, f2, f3

def plot_probability_vs_concentration(
    grouped_table, grouped_concentration_table, con_metric='r90/r10', 
    table_metric='isotropic.probability.min.rms', redshift_limit=0.2):
    plt.figure(figsize=(8,7))
    median_probs = []
    median_cons = []
    for (host, metrics), (con_host, con_metrics) in zip(grouped_table, grouped_concentration_table):
        assert host == con_host
        redshift_mask = metrics['redshift'] <= redshift_limit
        median_prob = np.nanmedian(metrics[table_metric][redshift_mask])
        percentile_prob = np.nanpercentile(metrics[table_metric][redshift_mask], [16, 84, 2.5, 97.5])

        median_con = np.nanmedian(con_metrics[con_metric][redshift_mask])
        percentile_con = np.nanpercentile(con_metrics[con_metric][redshift_mask], [16, 84, 2.5, 97.5])

        # correlation of probability of thinness with r90/r10
        con_err = np.array([median_con-percentile_con[0], percentile_con[1]-median_con])
        table_err = np.array([median_prob-percentile_prob[0], percentile_prob[1]-median_prob])
        plt.errorbar(median_con, median_prob, np.reshape(table_err, (2,1)), 
            np.reshape(con_err, (2,1)), fmt='None', capsize=0, alpha=0.6, color='k')

        if host in ['Romeo', 'Juliet', 'Thelma', 'Louise', 'Romulus', 'Remus']:
            if host == 'Romeo':
                plt.plot(median_con, median_prob, 'o', color='r', markeredgecolor='k', 
                         alpha=0.8, label='paired host')
            else:
                plt.plot(median_con, median_prob, 'o', color='r', markeredgecolor='k', 
                         alpha=0.8)
        else:
            if host == 'm12i':
                plt.plot(median_con, median_prob, '^', color='b', markeredgecolor='k', 
                         alpha=0.8, label='isolated host')
            else:
                plt.plot(median_con, median_prob, '^', color='b', markeredgecolor='k', 
                         alpha=0.8) 

        median_cons.append(median_con)
        median_probs.append(median_prob)

    #plt.tick_params(axis='both', which='major', labelsize=16)
    label_dict = {'axis.ratio':'Axis ratio [c/a]', 
                  'opening.angle':'Opening angle [deg]',
                  'rms.min':'RMS thickness [kpc]',
                  'orbital.pole.dispersion':'Orbital pole dispersion [deg]',
                  'r90/r10':r'$R_{90}/R_{10}$',
                  'r90/r50':r'$R_{90}/R_{50}$',
                  'r50':r'$R_{50}$'}
    plt.ylabel(label_dict[table_metric], fontsize=22)
    plt.xlabel(label_dict[con_metric], fontsize=22)
    plt.tick_params(axis='both', which='major', labelsize=18)
    #plt.ylabel(table_metric, fontsize=18)
    #plt.xlabel(con_metric, fontsize=18)
    plt.legend(loc=4, fontsize=20)
    plt.show()

    print('pearson correlation coefficient = ', np.corrcoef(median_cons, median_probs))
    print('spearman correlation coefficient = ', spearmanr(median_cons, median_probs))

def concentration_correlation_values(
    grouped_table, table_metric, grouped_concentration_table, con_metric='r90/r10', redshift_limit=0.2):
    
    median_probs = []
    median_cons = []
    for (host, metrics), (con_host, con_metrics) in zip(grouped_table, grouped_concentration_table):
        assert host == con_host
        redshift_mask = metrics['redshift'] <= redshift_limit
        median_prob = np.nanmedian(metrics[table_metric][redshift_mask])
        median_con = np.nanmedian(con_metrics[con_metric][redshift_mask])

        median_cons.append(median_con)
        median_probs.append(median_prob)
    
    return spearmanr(median_cons, median_probs)[0], spearmanr(median_cons, median_probs)[1]

def all_sim_concentration_correlation(
    all_grouped_tables, all_table_metrics, concentration_table, con_metric):

    con_correlation = defaultdict(list)
    for i, (table, metric) in enumerate(zip(all_grouped_tables, all_table_metrics)):
        a,b = concentration_correlation_values(table, metric, concentration_table, con_metric, 
                                               redshift_limit=0.2)
        c,d = concentration_correlation_values(table, metric, concentration_table, con_metric, 
                                               redshift_limit=0.5)

        con_correlation['metric'].append(metric)
        con_correlation['sp.corr.z0.2'].append(a)
        con_correlation['sp.p.z0.2'].append(b)
        con_correlation['sp.corr.z0.5'].append(c)
        con_correlation['sp.p.z0.5'].append(d)

    return pd.DataFrame(con_correlation)

def group_frac_correlation_values(
    grouped_table, table_metric, sat_infall_dict, time_table, redshift_limit, 
    frac_key):
    # get correlation values without plotting
    group_fractions = frac_in_groups(sat_infall_dict, time_table)
    group_fractions = group_fractions[frac_key]
    
    med_fracs = []
    med_probs = []
    for grouped_host, grouped_metrics in grouped_table:
        if grouped_host not in group_fractions.keys():
            continue
        redshift_mask = grouped_metrics['redshift'] <= redshift_limit
        median_prob = np.nanmedian(grouped_metrics[table_metric][redshift_mask])
        percentile_prob = np.nanpercentile(grouped_metrics[table_metric][redshift_mask], [16, 84, 2.5, 97.5])

        med_fracs.append(group_fractions[grouped_host])
        med_probs.append(median_prob)

    return spearmanr(med_fracs, med_probs)[0], spearmanr(med_fracs, med_probs)[1]

def all_sim_infall_correlation(
    all_grouped_tables, all_table_metrics, sat_infall_dict, time_table, frac_key):

    group_infall_correlation = defaultdict(list)
    for i, (table, metric) in enumerate(zip(all_grouped_tables, all_table_metrics)):
        a,b = group_frac_correlation_values(table, metric, sat_infall_dict, time_table, redshift_limit=0.2, 
                                             frac_key=frac_key)
        c,d = group_frac_correlation_values(table, metric, sat_infall_dict, time_table, redshift_limit=0.5, 
                                             frac_key=frac_key)

        group_infall_correlation['metric'].append(metric)
        group_infall_correlation['sp.corr.z0.2'].append(a)
        group_infall_correlation['sp.p.z0.2'].append(b)
        group_infall_correlation['sp.corr.z0.5'].append(c)
        group_infall_correlation['sp.p.z0.5'].append(d)

    return pd.DataFrame(group_infall_correlation)

def host_correlation_values(
    grouped_table, table_metric, host_table, host_metric='star.mass', redshift_limit=0.2):
    # get correlations from a table with single values for each host, like mass & LMC-passages
    group_host_mass = []
    medians = []

    for host, (group_host, group_metrics) in zip(host_table['host'], grouped_table):
        redshift_mask = group_metrics['redshift'] <= redshift_limit
        median = np.nanmedian(group_metrics[table_metric][redshift_mask])
        medians.append(median)
        group_host_mass.append(np.array(host_table[host_metric][host_table['host'] == group_host])[0])

    return spearmanr(group_host_mass, medians)[0], spearmanr(group_host_mass, medians)[1]

def all_sim_host_correlation(
    all_grouped_tables, all_table_metrics, host_table, host_metric):

    host_correlation = defaultdict(list)
    for i, (table, metric) in enumerate(zip(all_grouped_tables, all_table_metrics)):
        a,b = host_correlation_values(table, metric, host_table, host_metric, redshift_limit=0.2)
        c,d = host_correlation_values(table, metric, host_table, host_metric, redshift_limit=0.5)

        host_correlation['metric'].append(metric)
        host_correlation['sp.corr.z0.2'].append(a)
        host_correlation['sp.p.z0.2'].append(b)
        host_correlation['sp.corr.z0.5'].append(c)
        host_correlation['sp.p.z0.5'].append(d)

    return pd.DataFrame(host_correlation)

def plot_correlation_vs_metric_category(
    metric_labels, correlation_tbls, correlation_properties, legend_title,
    redshift_limit='0.2', color_map='magma'):
    markers_ = ['^', 'o', 's', 'd']
    fig, ax = plt.subplots(figsize=(11,7))
    for tbl, prop_label, mark in zip(correlation_tbls, correlation_properties, markers_):
        plt_sat = ax.scatter(metric_labels, tbl['sp.corr.z'+redshift_limit], 
                            label=prop_label,
                            s=300, c=tbl['sp.p.z'+redshift_limit], 
                            marker=mark, cmap=color_map, alpha=0.8,
                            edgecolors='k',linewidth=1.5, 
                            norm=plt.Normalize(0, 1))

    ax.legend(title=legend_title)
    if 'iso' in correlation_tbls[0]['metric'][0]:
        ax.set_xlabel('Isotropic plane probability metric')
    else:
        ax.set_xlabel('Plane metric')
    ax.set_ylabel('Spearman correlation coefficient')
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.tick_params(axis='x', which='major', labelsize=16)
    ax.tick_params(axis='x', which='minor', bottom=False, top=False)
    plt.xlim(-0.5, 3.5)
    fig.colorbar(plt_sat, label=r'$p$-value', ticks=np.arange(0,1.1, 0.1))
    plt.show()

def plot_group_frac_correlation(
    grouped_table, table_metric, sat_infall_dict, time_table, redshift_limit, 
    frac_key):
    fig, ax = plt.subplots(1,1,figsize=(6,5))
    fig.subplots_adjust(left=0.14, bottom=0.16, right=0.98, top=0.98)
    group_fractions = frac_in_groups(sat_infall_dict, time_table)
    group_fractions = group_fractions[frac_key]
    
    med_fracs = []
    med_probs = []
    for grouped_host, grouped_metrics in grouped_table:
        if grouped_host not in group_fractions.keys():
            continue
        redshift_mask = grouped_metrics['redshift'] <= redshift_limit
        median_prob = np.nanmedian(grouped_metrics[table_metric][redshift_mask])
        percentile_prob = np.nanpercentile(grouped_metrics[table_metric][redshift_mask], [16, 84, 2.5, 97.5])

        med_fracs.append(group_fractions[grouped_host])
        med_probs.append(median_prob)

        table_err = np.array([median_prob-percentile_prob[0], percentile_prob[1]-median_prob])
        plt.errorbar(group_fractions[grouped_host], median_prob, np.reshape(table_err, (2,1)), 
                     fmt='None', capsize=0, alpha=0.6, color='k')

        if grouped_host in ['Romeo', 'Juliet', 'Thelma', 'Louise', 'Romulus', 'Remus']:
            if grouped_host == 'Romeo':
                plt.plot(group_fractions[grouped_host], median_prob, 'o', color='r', markeredgecolor='k', 
                         alpha=0.8, label='paired host')
            else:
                plt.plot(group_fractions[grouped_host], median_prob, 'o', color='r', markeredgecolor='k', 
                         alpha=0.8)
        else:
            if grouped_host == 'm12i':
                plt.plot(group_fractions[grouped_host], median_prob, '^', color='b', markeredgecolor='k', 
                         alpha=0.8, label='isolated host')
            else:
                plt.plot(group_fractions[grouped_host], median_prob, '^', color='b', markeredgecolor='k', 
                         alpha=0.8)                

        #median_cons.append(median_con)
        #median_probs.append(median_prob)
    ax.tick_params(axis='both', which='major', labelsize=20)
    label_dict = {'axis.ratio':'Axis ratio [c/a]', 
                  'opening.angle':'Opening angle [deg]',
                  'rms.min':'RMS thickness [kpc]',
                  'orbital.pole.dispersion':'Orbital pole dispersion [deg]',
                  'largest.sat.group.fracs':'Fraction of z=0 sats. in largest groups at infall',
                  'largest.group.fracs':'Fraction of z=0 satellites',
                  'group.fracs':'Fraction of z=0 satellites'}
    plt.ylabel(label_dict[table_metric], fontsize=20)
    plt.xlabel(label_dict[frac_key], fontsize=18)
    plt.legend(loc=4, fontsize=18)
    plt.xlim((0,0.5))
    plt.show()
    
    print('pearson correlation coefficient = ', np.corrcoef(med_fracs, med_probs))
    print('spearman correlation coefficient = ', spearmanr(med_fracs, med_probs))

    #return fig

def lmc_correlation(
    host_table, grouped_table, y_type, y_label=None, redshift_limit=0.2):
    fig, ax = plt.subplots(1,1,figsize=(6,5))
    fig.subplots_adjust(left=0.15, bottom=0.16, right=0.98, top=0.98)
    group_host_mass = []
    medians = []
    median_dict = {}
    err_dict = {}
    for host, (group_host, group_metrics) in zip(host_table['host'], grouped_table):
        redshift_mask = group_metrics['redshift'] <= redshift_limit
        median = np.nanmedian(group_metrics[y_type][redshift_mask])
        medians.append(median)
        median_dict[group_host] = (median)
        percentile_ = np.reshape(np.nanpercentile(group_metrics[y_type][redshift_mask], [16, 84]), (2,1))
        _err = np.array([median-percentile_[0], percentile_[1]-median])
        err_dict[group_host] = _err
        group_host_mass.append(np.array(host_table['num.lmc.passages'][host_table['host'] == group_host])[0])
        plt.errorbar(np.array(host_table['num.lmc.passages'][host_table['host'] == group_host]), 
            median, _err, color='k', alpha=0.6)

        if group_host in ['Romeo', 'Juliet', 'Thelma', 'Louise', 'Romulus', 'Remus']:
            if group_host == 'Romeo':
                plt.plot(np.array(host_table['num.lmc.passages'][host_table['host'] == group_host]), 
                         median, 'o', color='r', markeredgecolor='k', alpha=0.8, label='paired host')
            else:
                plt.plot(np.array(host_table['num.lmc.passages'][host_table['host'] == group_host]), 
                         median, 'o', color='r', markeredgecolor='k', 
                         alpha=0.8)
        else:
            if group_host == 'm12i':
                plt.plot(np.array(host_table['num.lmc.passages'][host_table['host'] == group_host]), 
                         median, '^', color='b', markeredgecolor='k', alpha=0.8, label='isolated host')
            else:
                plt.plot(np.array(host_table['num.lmc.passages'][host_table['host'] == group_host]), 
                         median, '^', color='b', markeredgecolor='k', alpha=0.8)  

    plt.xlabel(r'Number of LMC-like passages', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=20)
    label_dict = {'axis.ratio':'Axis ratio [c/a]', 
                  'opening.angle':'Opening angle [deg]',
                  'rms.min':'RMS thickness [kpc]',
                  'orbital.pole.dispersion':'Orbital pole dispersion [deg]',
                  'largest.sat.group.fracs':'Fraction of z=0 satellites in largest group at infall'}
    if y_label is None:
        plt.ylabel(label_dict[y_type], fontsize=20)
    else:
        plt.ylabel(y_label, fontsize=20)

    plt.legend(fontsize=20, loc='center left')
    print('pearson correlation coefficient = ', np.corrcoef(group_host_mass, medians)[0][1])
    print('spearman correlation coefficient = ', spearmanr(group_host_mass, medians))

    return fig

#############################
### isotropic comparisons ###
#############################

def plot_iso_hist(
    grouped_table, iso_dist_, x_type, dmo_grouped_table=None, 
    dmo_iso_dist_=None, color_low=0, bins=None, obs=None):
    colors = sa.population_planarity.color_cycle(cycle_length=len(iso_dist_.keys()), low=color_low)
    medians, probs = sa.population_planarity.host_probability_stats(grouped_table, x_type, return_dict=True)
    if dmo_grouped_table is not None:
        dmo_medians, dmo_probs = sa.population_planarity.host_probability_stats(dmo_grouped_table, x_type, return_dict=True)

    for i,host in enumerate(iso_dist_.keys()):
        iso_dist = np.ndarray.flatten(np.array(iso_dist_[host]))
        
        plt.figure()
        plt.hist(iso_dist, bins=bins, cumulative=True, normed=True, color=colors[i])
        plt.axvline(medians[host], color='k', label=host)
        plt.axvspan(probs[host][0][0], probs[host][0][1], alpha=0.3, color='k')
        if obs is not None:
            plt.axvline(obs['MW'], color='r', linestyle='-', label='MW')
            try:
                plt.axvline(obs['M31'], color='r', linestyle='--', label='M31')
            except:
                pass
        
        if dmo_grouped_table is not None:
            dmo_iso_dist = np.ndarray.flatten(np.array(dmo_iso_dist_[host]))
            if host in dmo_medians.keys():
                plt.hist(dmo_iso_dist, cumulative=True, normed=True, color='k', histtype='step', alpha=0.5)
                plt.axvline(dmo_medians[host][0], color='k', label='dmo', alpha=0.5)
                plt.axvline(dmo_probs[host][0][0], color='k', linestyle='--', alpha=0.5)
                plt.axvline(dmo_probs[host][0][1], color='k', linestyle='--', alpha=0.5)
        
        plt.xlabel(x_type)
        plt.legend()
        plt.show()

def plot_norm_to_isotropic(
    grouped_table, y_type, y_type_isotropic, dmo_grouped_table=None, 
    sub_sample=None):
    # plot median/scatter across hosts
    # normalized to average isotropic axis ratio of each host
    iso_avg = []
    redshifts = []
    for host_key,host_group in grouped_table:
        iso_avg.append(host_group[y_type_isotropic].median())
        redshifts.append(host_group['redshift'])
    redshifts = np.array(redshifts[0])

    norm_stack = np.array([host_group[y_type].values/host_group[y_type_isotropic].mean() for host_key,host_group in grouped_table])
    iso_norm_stack = np.array([host_group[y_type_isotropic].values/host_group[y_type_isotropic].mean() for host_key,host_group in grouped_table])
    host_rms_percentile = np.nanpercentile(norm_stack, [16, 84, 2.5, 97.5], axis=0, interpolation='nearest')
    iso_host_rms_percentile = np.nanpercentile(iso_norm_stack, [16, 84, 2.5, 97.5], axis=0, interpolation='nearest')

    if dmo_grouped_table is not None:
        dmo_iso_avg = []
        dmo_redshifts = []
        for host_key,host_group in dmo_grouped_table:
            dmo_iso_avg.append(host_group[y_type_isotropic].median())
            dmo_redshifts.append(host_group['redshift'])
        dmo_redshifts = np.array(dmo_redshifts[0])

        dmo_norm_stack = np.array([host_group[y_type].values/host_group[y_type_isotropic].mean() for host_key,host_group in dmo_grouped_table])
        dmo_iso_norm_stack = np.array([host_group[y_type_isotropic].values/host_group[y_type_isotropic].mean() for host_key,host_group in dmo_grouped_table])
        dmo_host_rms_percentile = np.nanpercentile(dmo_norm_stack, [16, 84, 2.5, 97.5], axis=0, interpolation='nearest')
        dmo_iso_host_rms_percentile = np.nanpercentile(dmo_iso_norm_stack, [16, 84, 2.5, 97.5], axis=0, interpolation='nearest')

    if sub_sample is not None:
        sub_sample = np.arange(0, len(redshifts), sub_sample)
    else:
        sub_sample = np.arange(0, len(redshifts), 1)

    plt.figure(figsize=(7,6))
    plt.plot(redshifts[sub_sample], np.nanmedian(norm_stack, axis=0)[sub_sample], color='b', label='baryonic')
    plt.fill_between(redshifts[sub_sample], host_rms_percentile[0][sub_sample], host_rms_percentile[1][sub_sample], color='b', alpha=0.4)
    #plt.plot(redshifts, np.nanmedian(iso_norm_stack, axis=0), color='g', label='isotropic')
    #plt.fill_between(redshifts, iso_host_rms_percentile[0], iso_host_rms_percentile[1], color='g', alpha=0.4)
    plt.axhline(1.0, color='k', linestyle='--')
    if dmo_grouped_table is not None:
        plt.plot(dmo_redshifts[sub_sample], np.nanmedian(dmo_norm_stack, axis=0)[sub_sample], color='r', label='dmo')
        plt.fill_between(dmo_redshifts[sub_sample], dmo_host_rms_percentile[0][sub_sample], dmo_host_rms_percentile[1][sub_sample], color='r', alpha=0.2)
        #plt.plot(dmo_redshifts, np.nanmedian(dmo_iso_norm_stack, axis=0), color='y', label='dmo isotropic')
        #plt.fill_between(dmo_redshifts, dmo_iso_host_rms_percentile[0], dmo_iso_host_rms_percentile[1], color='y', alpha=0.2)
    plt.legend(loc=3)
    plt.xlabel('redshift [z]', fontsize=16)
    plt.ylabel(y_type+' relative to isotropic', fontsize=16)
    plt.show()
