import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec


CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

line_style = ['-.', '--', ':']


def plot_prop(prop_bins,
              prop_list1,
              prop_list2,
              halo_names,
              dmo_halo_names,
              label_names1,
              label_names2,
              plot_title,
              xlabel,
              ylabel,
              MW_data=None,
              M31_data=None,
              color_list=CB_color_cycle,
              xscale='linear',
              location=2):

    #for z in range(11):
    z=0
    fig = plt.figure(figsize=(8,6))
    plt.tight_layout()

    for i, (name, dmo_name) in enumerate(zip(halo_names, dmo_halo_names)):
        plt.plot(prop_bins, prop_list1[name][z], color=color_list[i], label=label_names1[i])
        if prop_list2 is not None:
            plt.plot(prop_bins, prop_list2[dmo_name][z], color=color_list[i], linestyle='--', label=label_names2[i])

    if MW_data is not None:
        plt.plot(prop_bins, MW_data, color='k', label='MW', linestyle='-')
    if M31_data is not None:
        plt.plot(prop_bins, M31_data, color='k', label='M31', linestyle='--')

    plt.xscale(xscale)	
    plt.legend(loc=location, handlelength=2, fontsize=16)
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.title(plot_title)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()

    return fig

def plot_coadded_prop(
    prop_medians, prop_percentiles, prop_bins,
    halo_names, plot_title=None, xlabel=None, ylabel=None,
    MW_data=None, M31_data=None, color_list=CB_color_cycle,
    xscale='linear', location=2):
    """
    Plots coadded data with 68% and 95% confidence limits for all host halos
    on the same subplot.
    """

    fig = plt.figure()
    plt.tight_layout()
    for i, name in enumerate(halo_names):
        plt.fill_between(prop_bins, prop_percentiles[i][2], prop_percentiles[i][3], alpha=0.3,
            color=color_list[i], linewidth=0)
        plt.fill_between(prop_bins, prop_percentiles[i][0], prop_percentiles[i][1], alpha=0.5,
            color=color_list[i], linewidth=0)
        plt.plot(prop_bins, prop_medians[i], color=color_list[i], label=name)

    if MW_data is not None:
        plt.plot(prop_bins, MW_data, color='k', label='MW', linestyle='-')
    if M31_data is not None:
        plt.plot(prop_bins, M31_data, color='k', label='M31', linestyle='--')
    if xscale == 'log':
        plt.semilogx()
        
    plt.legend(loc=location)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(plot_title)
    plt.show()

    return fig

def plot_comparison(
    prop_list,
    prop_bins,
    halo_names,
    plot_title1,
    plot_title2,
    xlabel, 
    ylabel,
    MW_data=None, 
    M31_data=None, 
    color_list=CB_color_cycle,
    xscale='linear',
    location=2):
    '''
    plots data for each host in each host_list
    prop_list = prop_list[host_list#][][host_halo_name][]
    '''

    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(12, 6))
    ax1.set_ylabel(ylabel)
    for k, ax in enumerate([ax1, ax2]):
        for i, name in enumerate(halo_names):
            ax.plot(prop_bins, prop_list[k][0][i][0], color=color_list[i], label=name)
            ax.set_xlabel(xlabel)

        if MW_data is not None:
            ax.plot(prop_bins, MW_data, color='k', label='MW', linestyle='-')
        if M31_data is not None:
            ax.plot(prop_bins, M31_data, color='k', label='M31', linestyle='--')

        ax.set_xscale(xscale)
        ax.legend(loc=location)

    ax1.set_title(plot_title1)
    ax2.set_title(plot_title2)
    plt.show()

def plot_coadd_comparison(
    x_data,
    y_data,
    isotropic_y_data=None,
    host_halo_names=None,
    plot_title='',
    xlabel=None,
    ylabel=None,
    MW_data=None, 
    M31_data=None, 
    color_list=CB_color_cycle,
    xscale='linear',
    location=None):
    '''
    Plot coadded data for 3 host halos side by side.

    prop_list = [prop1_list, prop2_list]
    prop1_list = [prop_means1, prop_medians1, prop_percentiles1]
    prop_list = [halo][means or medians or percentiles]
    '''

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(15, 5))
    for i, ax in enumerate([ax1, ax2, ax3]):
        ax.fill_between(x_data[i], y_data[2][i][2], y_data[2][i][3], alpha=0.3,
            color=color_list[i], linewidth=0)
        ax.fill_between(x_data[i], y_data[2][i][0], y_data[2][i][1], alpha=0.5,
            color=color_list[i], linewidth=0)
        ax.plot(x_data[i], y_data[0][i], color=color_list[i], label=host_halo_names[i])
        ax1.set_ylabel(ylabel, fontsize=12)
        '''
        if MW_data is not None:
            ax.plot(prop_bins, MW_data, color='k', label='MW', linestyle='-')
        if M31_data is not None:
            ax.plot(prop_bins, M31_data, color='k', label='M31', linestyle='--')
        '''

        #ax.set_xticks(angle_ticks)
        #ax.set_xticklabels(angle_labels)
        ax.set_xscale(xscale)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.legend(loc=location)

    #ax1.set_ylabel(ylabel, fontsize=12)
    fig.suptitle(plot_title, fontsize=12)
    fig.tight_layout()
    plt.show()
