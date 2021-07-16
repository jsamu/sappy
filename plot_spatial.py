import numpy as np
import math
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.optimize import curve_fit
from satellite_analysis import satellite_io as sio
from satellite_analysis import math_funcs as mf
from satellite_analysis import angular as ang
from satellite_analysis import kinematics as kin
from satellite_analysis import spatial as spa
from satellite_analysis import isotropic as iso
from satellite_analysis import plot_general as pg
from satellite_analysis import rand_axes as ra
from satellite_analysis import observation as obs
from satellite_analysis import population_planarity as popp

# change default font size
font = {'size'   : 20}
plt.rc('font', **font)

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00',
                  '#ffc107', '#6b0315', '#056765']

CB3_0 = ['#09B0C1', '#175F02', '#FF9D07']
CB3_1 = ['#0DBDC7', '#B906AF', '#E49507']

line_style = ['-', '--', '-.', ':']


def radial_distribution(
    sat, mask_key, redshift_index=None, stat=None, MW_data=False, M31_data=False,
    norm=False, n_iter=1000):
    """
    Plot each host's median profile if stat=='median', or the host's profile at
    a specific snapshot/redshift given by redshift_index.
    """
    n_sats, obs_sats = spa.cumulative_distance(sat, mask_key, norm=norm, obs=False)
    n_sats_coadd, obs_sats = spa.cumulative_distance(sat, mask_key, norm=norm, coadd=True, obs=False)
    #for redshift_index in range(len(sat.redshift)):
    fig, ax = plt.subplots(1, 1, figsize=(7,6))
    fig.set_tight_layout(False)
    fig.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95)

    #ax.set_yticks(np.arange(1,14,1), minor=True)
    #ax.set_yticks(np.arange(0,14,2), minor=False)
    ax.set_xticks([0,100,200,300], minor=False)
    ax.set_xticks(np.arange(25,300,25), minor=True)

    if MW_data:
        mw_dict, mw_samp = obs.mw_sample(
                                n_iter=1000, 
                                rbins=sat.r_bins, 
                                star_mass=sat.star_mass[0])
        ax.plot(sat.r_bins, mw_dict['median'], color='k', label='MW',
            linestyle='-', linewidth=2)
    if M31_data:
        m31_dict, m31_samp = obs.m31_sample(table_dir='./', 
                                M31_data_file='conn_m31_sats_posteriors.txt', 
                                sky_pos_file='mccon2012.vot',
                                n_iter=n_iter, rbins=sat.r_bins, 
                                star_mass=sat.star_mass[0], M31_sm_file='m31_sats.txt')
        # shade M31's line to reflect observational incompleteness
        rbins = np.arange(0, 310, 10)
        m31_rbins1_mask = rbins <= 150
        m31_rbins2_mask = rbins >= 150
        m31_rbins1 = np.arange(0, 160, 10)
        m31_rbins2 = np.arange(150, 310, 10)
        ax.plot(m31_rbins1, m31_dict['median'][m31_rbins1_mask], color='k', linestyle='--', label='M31')
        ax.plot(m31_rbins2, m31_dict['median'][m31_rbins2_mask], color='k', linestyle='--', alpha=0.4)

    name_iter = sio.hal_name_iter(sat)
    for i, name in enumerate(name_iter):
        if redshift_index is not None:
            ax.plot(sat.r_bins, n_sats[name][redshift_index],
                        color=CB_color_cycle[i],
                        linewidth=2.5, alpha=0.8, linestyle='--')
        elif stat:
            ax.plot(sat.r_bins, n_sats_coadd[name][stat],
                    color=CB_color_cycle[i], linewidth=2.5, alpha=0.7,
                    linestyle='-', label=sat.hal_name[i])
        else:
            pass
    
    plt.legend(loc=2, fontsize=18)
    plt.xlabel('Distance from host [kpc]', fontsize=22)
    plt.ylabel('N$_{sat}$(<d)', fontsize=22)
    plt.xlim(0,300)
    #plt.ylim(0,14)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.show()
    #fig.savefig('/home1/05385/tg846849/figures/radial_distribution/res7100/m12f_{}.pdf'.format(redshift_index))
    #fig.savefig('/home1/05385/tg846849/figures/radial_distribution/res7100/m12f_test.pdf')

    return fig

def radial_distribution_coadd(
    sat, mask_key, MW_data=False, M31_data=False, norm=False, figdir=None,
    stat='median', conf_int=True, n_iter=1000):
    """
    Plot the coadded profile for each host on a separate figure, using 68% and 
    95% regions.
    """
    n_sats, obs_sats = spa.cumulative_distance(sat, mask_key, norm=norm, coadd=True)
    sm_mag = math.floor(np.log10(sat.star_mass[0]))
    sm_coeff = sat.star_mass[0]/10**(sm_mag)
    sm_str = '{:.1f}_by_10_{}'.format(sm_coeff, sm_mag)
    
    name_iter = sio.hal_name_iter(sat)
    for i, name in enumerate(name_iter):
        fig = plt.figure(figsize=(7,6))
        if conf_int:
            plt.fill_between(sat.r_bins, n_sats[name]['percentile'][2], n_sats[name]['percentile'][3], alpha=0.3,
                color=CB_color_cycle[i], linewidth=0)
            plt.fill_between(sat.r_bins, n_sats[name]['percentile'][0], n_sats[name]['percentile'][1], alpha=0.5,
                color=CB_color_cycle[i], linewidth=0)
        plt.plot(sat.r_bins, n_sats[name][stat], color=CB_color_cycle[i], label=sat.hal_label[mask_key][i])

        # sample and plot median of observations
        if MW_data:
            mw_dict, mw_samp = obs.mw_sample(table_dir='./', 
                                    MW_data_file='mw_sats_distances.txt', 
                                    sky_pos_file='mccon2012.vot',
                                    n_iter=n_iter, 
                                    rbins=sat.r_bins, 
                                    star_mass=sat.star_mass[0])
            plt.plot(sat.r_bins, mw_dict['median'], color='k', label='MW', linestyle='-')
        if M31_data:
            m31_dict, m31_samp = obs.m31_sample(table_dir='./', 
                                    M31_data_file='conn_m31_sats_posteriors.txt', 
                                    sky_pos_file='mccon2012.vot',
                                    n_iter=n_iter, rbins=sat.r_bins, 
                                    star_mass=sat.star_mass[0], M31_sm_file='m31_sats.txt')
            # shade M31's line to reflect observational incompleteness
            rbins = np.arange(0, 310, 10)
            m31_rbins1_mask = rbins <= 150
            m31_rbins2_mask = rbins >= 150
            m31_rbins1 = np.arange(0, 160, 10)
            m31_rbins2 = np.arange(150, 310, 10)
            plt.plot(m31_rbins1, m31_dict['median'][m31_rbins1_mask], color='k', linestyle='--', label='M31')
            plt.plot(m31_rbins2, m31_dict['median'][m31_rbins2_mask], color='k', linestyle='--', alpha=0.4)

            
        plt.legend(loc=2, fontsize=16)
        plt.xlabel('Distance from host [kpc]', fontsize=22)
        plt.ylabel('N$_{sat}(<D)$', fontsize=22)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.show()
        if figdir:
            fig.savefig('{}median{}_{}.pdf'.format(figdir, max(sat.redshift), sm_str))

    return fig

def radial_distribution_coadd_subplots(
    sat, sat_lg, mask_key, MW_data=False, M31_data=False, norm=False,
    figdir=None, stat='median', n_iter=1000, obs_dir='./'):
    """
    Plot each of the 12 Latte and ELVIS hosts' profiles on their own panel.
    """
    m12_nsat, obs_sats = spa.cumulative_distance(sat, mask_key, norm=norm, coadd=True)
    lg_nsat, obs_sats = spa.cumulative_distance(sat_lg, mask_key, norm=norm, coadd=True)
    all_nsat = {**m12_nsat, **lg_nsat}

    fig, ax = plt.subplots(nrows=3, ncols=4, sharex=True, sharey=True, figsize=(12, 9))
    fig.set_tight_layout(False)

    # sample observations
    if MW_data:
        mw_dict, mw_samp = obs.mw_sample(table_dir=obs_dir, 
                                MW_data_file='mw_sats_distances.txt', 
                                sky_pos_file='mccon2012.vot',
                                n_iter=n_iter, 
                                rbins=sat.r_bins, 
                                star_mass=sat.star_mass[0])
    if M31_data:
        m31_dict, m31_samp = obs.m31_sample(table_dir=obs_dir, 
                                M31_data_file='conn_m31_sats_posteriors.txt', 
                                sky_pos_file='mccon2012.vot',
                                n_iter=n_iter, rbins=sat.r_bins, 
                                star_mass=sat.star_mass[0], M31_sm_file='m31_sats.txt')
        # shade M31's line to reflect observational incompleteness
        rbins = np.arange(0, 310, 10)
        m31_rbins1_mask = rbins <= 150
        m31_rbins2_mask = rbins >= 150
        m31_rbins1 = np.arange(0, 160, 10)
        m31_rbins2 = np.arange(150, 310, 10)

    # order hosts from most stellar mass to least
    name_iter = ['m12m', 'm12b', 'm12f', 'Thelma', 'Romeo', 'm12i', 'm12c',
                 'm12w', 'Juliet', 'Louise', 'm12z', 'm12r']
    for i, name in enumerate(name_iter):
        if 'm12' in name:
            plot_color = CB3_1[1]
        else:
            plot_color = CB3_1[2]
        j = math.floor(i/4)
        k = i - math.floor(i/4)*4
        ax[j, k].fill_between(sat.r_bins, all_nsat[name]['percentile'][2], all_nsat[name]['percentile'][3], alpha=0.25,
            color=plot_color, linewidth=0)
        ax[j, k].fill_between(sat.r_bins, all_nsat[name]['percentile'][0], all_nsat[name]['percentile'][1], alpha=0.25,
            color=plot_color, linewidth=0)
        ax[j, k].plot(sat.r_bins, all_nsat[name][stat], color=plot_color, linewidth=2, label=name)

        if MW_data:
            ax[j, k].plot(sat.r_bins, mw_dict['median'], color='k', linestyle='-')
        if M31_data:
            rbins = np.arange(0, 310, 10)
            m31_rbins1_mask = rbins <= 150
            m31_rbins2_mask = rbins >= 150
            m31_rbins1 = np.arange(0, 160, 10)
            m31_rbins2 = np.arange(150, 310, 10)
            ax[j, k].plot(m31_rbins1, m31_dict['median'][m31_rbins1_mask], color='k', linestyle='--')
            ax[j, k].plot(m31_rbins2, m31_dict['median'][m31_rbins2_mask], color='k', linestyle='--', alpha=0.4)

        ax[j, k].legend(loc=2, fontsize=20)
        ax[j, k].tick_params(axis='both', which='major', labelsize=20)
        ax[j, k].set_xlim((0,300))
        ax[j, k].set_ylim((0,30))
        ax[j, k].set_xticks([50, 150, 250], minor=True)
        ax[j, k].set_xticks([100, 200], minor=False)
        ax[j, k].set_yticks(np.arange(1,30,1), minor=True)
        ax[j, k].set_yticks([0, 5, 10, 15, 20, 25], minor=False)
        ax[j, k].set_yticklabels(['', '', '10', '', '20', ''], minor=False)
        ax[j, k].tick_params(axis='y', which='both', right=False)

    ax[0, 3].tick_params(axis='y', which='both', right=True)
    ax[1, 3].tick_params(axis='y', which='both', right=True)
    ax[2, 3].tick_params(axis='y', which='both', right=True)

    if MW_data:
        ax[0, 0].plot(sat.r_bins, mw_dict['median'], color='k', label='MW', linestyle='-')
    if M31_data:
        ax[0, 0].plot(m31_rbins1, m31_dict['median'][m31_rbins1_mask], color='k', linestyle='--', label='M31')

    ax[0, 0].legend(loc=2, fontsize=18)
    fig.text(0.525, 0.02, 'Distance from host [kpc]', ha='center', fontsize=22)
    fig.text(0.02, 0.53, r'N$_{\rm sat}(<\rm{d})$', va='center', rotation='vertical', fontsize=23)
    #fig.text(0.02, 0.53, '0', va='center', fontsize=20)

    # extra axis labels
    fig.text(0.08, 0.075, '0', va='center', fontsize=20)
    fig.text(0.065, 0.945, '30', va='center', fontsize=20)
    fig.text(0.93, 0.075, '300', va='center', fontsize=20)

    fig.subplots_adjust(left=0.1, bottom=0.1, wspace=0, hspace=0)
    plt.show()
    if figdir:
        sm_mag = math.floor(np.log10(sat.star_mass[0]))
        sm_coeff = sat.star_mass[0]/10**(sm_mag)
        sm_str = '{:.1f}_by_10_{}'.format(sm_coeff, sm_mag)
        fig.savefig('{}coadd_all_hosts_z{}_{}_smticks.pdf'.format(figdir, max(sat.redshift), sm_str))

    return fig

def radial_diff_and_cumu_all(
    dmo_m12, m12_sat, dmo_lg=None, lg_sat=None, mask_key=None, norm=False,
    stat='median', bins=None, legend_title=None, figdir=None):
    """
    Plot the satellite/subhalo counts of one set of simulations (sat) normalized
    to satellite counts of another set of simulations (baryon_sat). Top subplot
    is cumulative, and bottom is differential counts.
    """
    if bins is None:
        bins = dmo_m12.r_bins

    bins_cumul = np.arange(0,305,5)
    n_diff1 = spa.coadd_num_v_radius_bin_ratio(dmo_m12, m12_sat, dmo_lg, lg_sat, mask_key=mask_key, bins=bins_cumul, diff=True)
    n_cum1 = spa.coadd_num_v_radius_bin_ratio(dmo_m12, m12_sat, dmo_lg, lg_sat, mask_key=mask_key, bins=bins_cumul, diff=False)

    n_diff2 = spa.coadd_num_v_radius_bin_ratio(dmo_m12, m12_sat, dmo_lg, lg_sat, mask_key=mask_key, bins=bins, diff=True)
    n_cum2 = spa.coadd_num_v_radius_bin_ratio(dmo_m12, m12_sat, dmo_lg, lg_sat, mask_key=mask_key, bins=bins, diff=False)

    fig = plt.figure(figsize=(6, 8.5))
    fig.set_tight_layout(False)
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1, sharey=ax1)
    fig.subplots_adjust(left=0.20, bottom=0.1, hspace=0.07)

    a = np.empty(len(bins))
    a.fill(1)
    ax1.plot(bins, a, color='k', linestyle=':', alpha=0.35)
    ax2.plot(bins, a, color='k', linestyle=':', alpha=0.35)

    ax1.set_xlim(0,300)
    ax1.set_ylim(0, 1)
    ax2.set_xlim(0,300)

    ax1.fill_between(bins, n_cum1['percentile'][2],
        n_cum1['percentile'][3], color='indigo', alpha=0.25, linewidth=0)
    ax1.fill_between(bins, n_cum1['percentile'][0],
        n_cum1['percentile'][1], color='indigo', alpha=0.25, linewidth=0)
    ax1.plot(bins, n_cum1[stat], color='indigo', linestyle='-')
    ax1.set_ylabel('N$_{baryon}(<D)/$N$_{DMO}(<D)$', fontsize=22)

    ax2.fill_between(bins, n_diff2['percentile'][2],
        n_diff2['percentile'][3], color='indigo', alpha=0.25, linewidth=0)
    ax2.fill_between(bins, n_diff2['percentile'][0],
        n_diff2['percentile'][1], color='indigo', alpha=0.25, linewidth=0)
    ax2.plot(bins, n_diff2[stat], color='indigo', linestyle='-')
    ax2.set_ylabel('N$_{baryon}/$N$_{DMO}$', fontsize=22)

    ax1.tick_params(axis='x', which='major', labelbottom=False)
    ax1.tick_params(axis='both', which='major', labelsize=18)
    ax2.set_xlabel('Distance from host [kpc]', fontsize=22)
    ax2.tick_params(axis='both', which='major', labelsize=18)
    plt.show()

    if figdir:
        fig.savefig(figdir+'dmo_baryon_normed_z{}.pdf'.format(max(dmo_m12.redshift)))
        plt.close(fig)

    return fig

def radial_dist_coadd_all(
    sat, mask_key, MW_data=False, M31_data=False, norm=False, figdir=None,
    stat='median', norm_by=None, leg_title=None, n_iter=1000):
    """
    Plot all profiles coadded together for all hosts/snapshots.
    """
    if sat.sat_type == 'tree':
        plot_name = 'm12'
    elif sat.sat_type == 'tree.lg':
        plot_name = 'LG'

    n_sats, obs_sats = spa.coadd_distance_all_hosts(sat, mask_key, norm=norm, bins=None)

    fig = plt.figure(figsize=(6, 8))
    fig.set_tight_layout(False)
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    fig.subplots_adjust(left=0.21, bottom=0.11, wspace=0.05, hspace=0)

    ax1.fill_between(sat.r_bins, n_sats['percentile'][2],
        n_sats['percentile'][3], color=CB3_1[0], alpha=0.3, linewidth=0)
    ax1.fill_between(sat.r_bins, n_sats['percentile'][0],
        n_sats['percentile'][1], color=CB3_1[0], alpha=0.4, linewidth=0)
    ax1.plot(sat.r_bins, n_sats[stat], color=CB3_1[0], label='isolated hosts')

    # sample and plot median of observations
    if MW_data:
        mw_dict, mw_samp = obs.mw_sample(table_dir='./', 
                                MW_data_file='mw_sats_distances.txt', 
                                sky_pos_file='mccon2012.vot',
                                n_iter=n_iter, 
                                rbins=sat.r_bins, 
                                star_mass=sat.star_mass[0])
        ax1.plot(sat.r_bins, mw_dict['median'], color='k', label='MW', linestyle='-')
    if M31_data:
        m31_dict, m31_samp = obs.m31_sample(table_dir='./', 
                                M31_data_file='conn_m31_sats_posteriors.txt', 
                                sky_pos_file='mccon2012.vot',
                                n_iter=n_iter, rbins=sat.r_bins, 
                                star_mass=sat.star_mass[0], M31_sm_file='m31_sats.txt')
        # shade M31's line to reflect observational incompleteness
        rbins = np.arange(0, 310, 10)
        m31_rbins1_mask = rbins <= 150
        m31_rbins2_mask = rbins >= 150
        m31_rbins1 = np.arange(0, 160, 10)
        m31_rbins2 = np.arange(150, 310, 10)
        ax1.plot(m31_rbins1, m31_dict['median'][m31_rbins1_mask], color='k', linestyle='--', label='M31')
        ax1.plot(m31_rbins2, m31_dict['median'][m31_rbins2_mask], color='k', linestyle='--', alpha=0.4)

    # not sure if this works?
    if norm_by:
        norm_sats, obs_sats = spa.coadd_distance_all_hosts(norm_by, mask_key=mask_key, norm=norm, bins=None)
        ax2.plot(sat.r_bins, np.array(n_sats[stat])/np.array(norm_sats[stat]), color=CB3_1[0],
            linestyle='-')
        ax2.set_ylabel('N$_{sat}/$N$_{baryon}$', fontsize=22)

    else:
        ax2.plot(sat.r_bins, np.array(n_sats[stat])/np.array(obs_sats['MW']), color=CB3_1[0],
            label='isolated hosts/MW', linestyle='-')
        ax2.plot(sat.r_bins, np.array(n_sats[stat])/np.array(obs_sats['M31']), color=CB3_1[0],
            label='isolated hosts/M31', linestyle='--')
        ax2.set_ylabel('N$_{sat}/$N$_{obs}$', fontsize=22)
        
    ax1.legend(loc=2, fontsize=16, title=leg_title)
    ax1.set_ylabel('N$_{sat}(<D)$', fontsize=22)
    ax2.set_xlabel('Distance from host [kpc]', fontsize=22)
    ax2.legend(loc=2, fontsize=16)
    ax1.tick_params(axis='both', which='major', labelsize=18)
    ax2.tick_params(axis='both', which='major', labelsize=18)
    plt.show()

    if figdir:
        fig.savefig(figdir+'coadd_all_{}_z{}_sub.pdf'.format(plot_name, max(sat.redshift)))
        plt.close(fig)

    return fig

def radial_dist_coadd_m12_and_lg_sub3(m12_sat, lg_sat, mask_key, MW_data=False, 
    M31_data=False, norm=False, figdir=None, stat='median', mass_cuts=None, 
    obs_sample=True, obs_dir='./', rbins=None, n_iter=1000, diff=False):
    """
    Makes the main figure from radial paper. Coadd all hosts at 3 satellite mass
    scales, normalized to observations in lower panel. Different lines for pairs
    vs. isolated hosts.
    """
    if rbins is None:
        rbins = m12_sat.r_bins
    if not mass_cuts:
        mass_cuts = [m12_sat.star_mass[0], m12_sat.star_mass[0]*10, m12_sat.star_mass[0]*100]
    
    fig = plt.figure(figsize=(14, 8))
    fig.set_tight_layout(False)
    gs = gridspec.GridSpec(3, 3, height_ratios=[4, 1, 1])

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1, sharey=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1, sharey=ax1)
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    ax5 = fig.add_subplot(gs[4], sharex=ax1, sharey=ax4)
    ax6 = fig.add_subplot(gs[5], sharex=ax1, sharey=ax4)
    ax7 = fig.add_subplot(gs[6], sharex=ax1)
    ax8 = fig.add_subplot(gs[7], sharex=ax1, sharey=ax7)
    ax9 = fig.add_subplot(gs[8], sharex=ax1, sharey=ax7)

    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.11, wspace=0, hspace=0)
    axlist = [[ax1, ax4, ax7], [ax2, ax5, ax8], [ax3, ax6, ax9]]
    for i, (ax, mass_cut) in enumerate(zip(axlist, mass_cuts)):
        if not mass_cut == m12_sat.star_mass[0]:
            # reset sm limits
            sio.reset_sm_limits(lg_sat, lower_lim=mass_cut)
            sio.reset_sm_limits(m12_sat, lower_lim=mass_cut)

        all_nsat, m12_nsat, lg_nsat = spa.coadd_distance_m12_and_lg(m12_sat, lg_sat, mask_key, norm=norm, bins=rbins, diff=diff)

        ax[0].fill_between(rbins, all_nsat['percentile'][2],
            all_nsat['percentile'][3], color=CB3_1[0], alpha=0.25, linewidth=0)
        ax[0].fill_between(rbins, all_nsat['percentile'][0],
            all_nsat['percentile'][1], color=CB3_1[0], alpha=0.25, linewidth=0)
        if diff is True:
            ax[0].plot(rbins, all_nsat[stat], color=CB3_1[0], linewidth=4, label='all hosts')
        else:
            ax[0].plot(rbins, all_nsat[stat], color=CB3_1[0], linewidth=3, label='all hosts')
        ax[0].plot(rbins, m12_nsat[stat], color=CB3_1[1], linewidth=3, label='isolated hosts')
        ax[0].plot(rbins, lg_nsat[stat], color=CB3_1[2], linewidth=3, label='paired hosts')

        # sample and plot median of observations
        if MW_data:
            mw_dict, mw_samp = obs.mw_sample(table_dir=obs_dir, 
                                    MW_data_file='mw_sats_distances.txt', 
                                    sky_pos_file='mccon2012.vot',
                                    n_iter=n_iter, 
                                    rbins=rbins, 
                                    star_mass=mass_cut,
                                    diff=diff)
            ax[0].plot(rbins, mw_dict['median'], color='k', label='MW', linestyle='-')
        if M31_data:
            m31_dict, m31_samp = obs.m31_sample(table_dir=obs_dir, 
                                    M31_data_file='conn_m31_sats_posteriors.txt', 
                                    sky_pos_file='mccon2012.vot',
                                    n_iter=n_iter, rbins=rbins, 
                                    star_mass=mass_cut,
                                    M31_sm_file='m31_sats.txt',
                                    diff=diff)
            # shade M31's line to reflect observational incompleteness
            #rbins = np.arange(0, 310, 10)
            m31_rbins1_mask = rbins <= 150
            m31_rbins2_mask = rbins >= 150
            m31_rbins1 = rbins[m31_rbins1_mask]#np.arange(0, 160, 10)
            m31_rbins2 = rbins[m31_rbins2_mask]#np.arange(150, 310, 10)
            ax[0].plot(m31_rbins1, m31_dict['median'][m31_rbins1_mask], color='k', linestyle='--', label='M31')
            ax[0].plot(m31_rbins2, m31_dict['median'][m31_rbins2_mask], color='k', linestyle='--', alpha=0.4)

        # calculate and plot observational ratios
        if obs_sample:
            obs_ratios = spa.coadd_distance_ratio(m12_sat, lg_sat, mask_key, 
                            mean_first=True, 
                            obs_samp={'MW':mw_samp, 'M31':m31_samp}, bins=rbins,
                            diff=diff)
        else:
            obs_ratios = spa.coadd_distance_ratio(m12_sat, lg_sat, mask_key)

        ax[1].plot(rbins, obs_ratios['MW'][stat], color=CB3_1[0],
                linestyle='-')
        ax[1].fill_between(rbins, obs_ratios['MW']['percentile'][2],
            obs_ratios['MW']['percentile'][3], color=CB3_1[0], alpha=0.25, linewidth=0)
        ax[1].fill_between(rbins, obs_ratios['MW']['percentile'][0],
            obs_ratios['MW']['percentile'][1], color=CB3_1[0], alpha=0.25, linewidth=0)
        a = np.empty(len(rbins))
        a.fill(1)
        ax[1].plot(rbins, a, color='k', linestyle=':', alpha=0.4)
        ax[1].set_ylim(0, 2)

        ax[2].plot(rbins, obs_ratios['M31'][stat], color=CB3_1[0],
                linestyle='--')
        ax[2].fill_between(rbins, obs_ratios['M31']['percentile'][2],
            obs_ratios['M31']['percentile'][3], color=CB3_1[0], alpha=0.25, linewidth=0)
        ax[2].fill_between(rbins, obs_ratios['M31']['percentile'][0],
            obs_ratios['M31']['percentile'][1], color=CB3_1[0], alpha=0.25, linewidth=0)
        a = np.empty(len(rbins))
        a.fill(1)
        ax[2].plot(rbins, a, color='k', linestyle=':', alpha=0.4)
        ax[2].set_ylim(0, 2)

    font = {'size'   : 23}
    plt.rc('font', **font)
    ax1.legend(handles=[None], labels=[None], loc=2, fontsize=22, 
        title=r'satellite M$_* > 10^5$ M$_{\odot}$', borderaxespad=0.7)
    ax2.legend(handles=[None], labels=[None], loc=2, fontsize=22, 
        title=r'satellite M$_* > 10^6$ M$_{\odot}$', borderaxespad=0.7)
    ax3.legend(loc=2, fontsize=22, title=r'satellite M$_* > 10^7$ M$_{\odot}$',
        borderaxespad=0.7, handlelength=1.2)

    font = {'size'   : 18}
    plt.rc('font', **font)
    ax4.legend(loc=4, title='all hosts/MW  ', borderaxespad=0.1)
    ax7.legend(loc=1, title='all hosts/M31', borderaxespad=0.5)

    for ax in [ax1,ax2,ax3]:
        if diff is True:
            ax.set_ylim(0,10)
            ax.set_yticks(np.arange(0,11,1), minor=False)
        else:
            ax.set_ylim(0,28)
            ax.set_yticks(np.arange(5,30,5), minor=False)
    for ax in [ax4, ax5, ax6, ax7, ax8, ax9]:
        ax.set_yticks(np.arange(0,2.25,.25), minor=True)
        ax.set_yticks([0.5, 1, 1.5], minor=False)
    for ax in [ax4, ax7]:
        ax.set_yticklabels(['0.5', '', '1.5'])
    for ax in [ax1,ax2,ax3,ax4,ax5,ax6]:
        ax.tick_params(axis='x', which='major', labelbottom=False)
    for ax in [ax2,ax3,ax5,ax6,ax8,ax9]:
        ax.tick_params(axis='y', which='major', labelleft=False)
    for ax in [ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8,ax9]:
        ax.set_xlim(0,300)
        ax.set_xticks([25,50,75,125,150,175,225,250,275], minor=True)
        ax.set_xticks([100,200], minor=False)
        ax.tick_params(axis='both', which='major', labelsize=22)

    if diff is True:
        ax1.set_ylabel(r'dN$_{\rm sat}/\rm{dr})$', fontsize=26)
        fig.text(0.01, 0.25, r'dN$_{\rm sim}$/dN$_{\rm obs}$', va='center', rotation='vertical', fontsize=26)
    else:
        ax1.set_ylabel(r'N$_{\rm sat}(<\rm{d})$', fontsize=26)
        fig.text(0.01, 0.25, r'N$_{\rm sim}$/N$_{\rm obs}$', va='center', rotation='vertical', fontsize=26)
    ax8.set_xlabel('Distance from host [kpc]', fontsize=26)

    # extra axis labels
    fig.text(0.07, 0.08, '0', va='center', fontsize=22)
    #fig.text(0.05, 0.95, '30', va='center', fontsize=22)
    fig.text(0.95, 0.08, '300', va='center', fontsize=22)

    plt.show()
    if figdir:
        fig.savefig(figdir+'coadd_all_{}_z{}_main.pdf'.format(stat, max(m12_sat.redshift)))
        plt.close(fig)

    return fig

def radial_dist_coadd_m12_and_lg_total_only(
    sat, sat_lg, mask_key, MW_data=False, M31_data=False, norm=False, 
    figdir=None, stat='median', separate=False, n_iter=1000, diff=False,
    obs_dir='./', bins=None):
    """
    Makes a clean presentation/proposal plot of all hosts/snapshots coadded.
    """
    if bins is None:
        bins = sat.r_bins
    fig, ax = plt.subplots(1, 1, figsize=(7.5,7))
    fig.set_tight_layout(False)
    fig.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.14)
    all_nsat, m12_nsat, lg_nsat = spa.coadd_distance_m12_and_lg(sat, sat_lg, mask_key, norm=norm, bins=bins, diff=diff)

    plt.fill_between(bins, all_nsat['percentile'][2],
        all_nsat['percentile'][3], color=CB3_1[0], alpha=0.3, linewidth=0)
    plt.fill_between(bins, all_nsat['percentile'][0],
        all_nsat['percentile'][1], color=CB3_1[0], alpha=0.3, linewidth=0)
    lw = 3
    if diff is True:
        lw = 4
    plt.plot(bins, all_nsat[stat], color=CB3_1[0], linewidth=lw, label='all hosts')

    if separate:
        plt.plot(bins, lg_nsat[stat], color=CB3_1[2], linewidth=3, label='paired hosts')
        plt.plot(bins, m12_nsat[stat], color=CB3_1[1], linewidth=3, linestyle='-.', label='isolated hosts')

    # sample and plot median of observations
    if MW_data:
        mw_dict, mw_samp = obs.mw_sample(table_dir=obs_dir, 
                                MW_data_file='mw_sats_distances.txt', 
                                sky_pos_file='mccon2012.vot',
                                n_iter=n_iter, 
                                rbins=bins, 
                                star_mass=sat.star_mass[0],
                                diff=diff)
        plt.plot(bins, mw_dict['median'], color='k', label='MW', linestyle=':')
    if M31_data:
        m31_dict, m31_samp = obs.m31_sample(table_dir=obs_dir, 
                                M31_data_file='conn_m31_sats_posteriors.txt', 
                                sky_pos_file='mccon2012.vot',
                                n_iter=n_iter, 
                                rbins=bins, 
                                star_mass=sat.star_mass[0], 
                                M31_sm_file='m31_sats.txt',
                                diff=diff)
        # shade M31's line to reflect observational incompleteness
        m31_rbins1_mask = bins <= 150
        m31_rbins2_mask = bins >= 150
        plt.plot(bins[m31_rbins1_mask], m31_dict['median'][m31_rbins1_mask], color='k', linestyle='--', label='M31')
        plt.plot(bins[m31_rbins2_mask], m31_dict['median'][m31_rbins2_mask], color='k', linestyle='--', alpha=0.4)

    plt.legend(loc=2, handlelength=1.3, fontsize=20, title=r'satellite M$_* > 10^5$ M$_{\odot}$', borderaxespad=0.7)
    plt.xlabel('Distance from host [kpc]', fontsize=26)
    if diff is True:
        plt.ylabel(r'dN$_{\rm sat}/\rm{dr}$', fontsize=26)
    else:
        plt.ylabel(r'N$_{\rm sat}(<\rm{d})$', fontsize=26)

    if diff is True:
        plt.ylim(0,11)
        ax.set_yticks(np.arange(1,11,1), minor=False)
        ax.tick_params(axis='y', which='minor', left=False, right=False)
    ax.set_xlim(0,300)
    ax.set_xticks([25,50,75,125,150,175,225,250,275], minor=True)
    ax.set_xticks([0,100,200,300], minor=False)
    ax.tick_params(axis='both', which='major', labelsize=22)


    #plt.show()
    if figdir:
        fig.savefig('{}total_radial_m12_lg_z01.pdf'.format(figdir))
        #plt.close(fig)

    return fig

def radial_dist_poster(
    sat, sat_lg, mask_key, MW_data=False, M31_data=False, norm=False, 
    figdir=None, stat='median', n_iter=1000, obs_dir='./'):
    """
    Makes a clean presentation/proposal plot of all hosts/snapshots coadded.
    """
    fig = plt.figure(figsize=(7,8))
    fig.set_tight_layout(False)
    gs = gridspec.GridSpec(3, 1, height_ratios=[4, 1, 1])

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1, sharey=ax2)
    fig.subplots_adjust(left=0.15, right=0.95, bottom=0.11, hspace=0)

    all_nsat, m12_nsat, lg_nsat = spa.coadd_distance_m12_and_lg(sat, sat_lg, mask_key, norm=norm, bins=None)

    ax1.fill_between(sat.r_bins, all_nsat['percentile'][2],
        all_nsat['percentile'][3], color=CB3_1[0], alpha=0.3, linewidth=0)
    ax1.fill_between(sat.r_bins, all_nsat['percentile'][0],
        all_nsat['percentile'][1], color=CB3_1[0], alpha=0.3, linewidth=0)
    ax1.plot(sat.r_bins, all_nsat[stat], color=CB3_1[0], linewidth=3, label='all hosts')

    ax1.plot(sat.r_bins, lg_nsat[stat], color=CB3_1[2], linewidth=3, label='paired hosts')
    ax1.plot(sat.r_bins, m12_nsat[stat], color=CB3_1[1], linewidth=3, label='isolated hosts')

    # sample and plot median of observations
    if MW_data:
        mw_dict, mw_samp = obs.mw_sample(table_dir=obs_dir, 
                                MW_data_file='mw_sats_distances.txt', 
                                sky_pos_file='mccon2012.vot',
                                n_iter=n_iter, 
                                rbins=sat.r_bins, 
                                star_mass=sat.star_mass[0])
        ax1.plot(sat.r_bins, mw_dict['median'], color='k', label='MW', linestyle='-')
    if M31_data:
        m31_dict, m31_samp = obs.m31_sample(table_dir=obs_dir, 
                                M31_data_file='conn_m31_sats_posteriors.txt', 
                                sky_pos_file='mccon2012.vot',
                                n_iter=n_iter, rbins=sat.r_bins, 
                                star_mass=sat.star_mass[0], M31_sm_file='m31_sats.txt')
        # shade M31's line to reflect observational incompleteness
        rbins = np.arange(0, 310, 10)
        m31_rbins1_mask = rbins <= 150
        m31_rbins2_mask = rbins >= 150
        m31_rbins1 = np.arange(0, 160, 10)
        m31_rbins2 = np.arange(150, 310, 10)
        ax1.plot(m31_rbins1, m31_dict['median'][m31_rbins1_mask], color='k', linestyle='--', label='M31')
        ax1.plot(m31_rbins2, m31_dict['median'][m31_rbins2_mask], color='k', linestyle='--', alpha=0.4)

    # calculate and plot observational ratios
    obs_ratios = spa.coadd_distance_ratio(sat, sat_lg, mask_key, 
                    mean_first=True, 
                    obs_samp={'MW':mw_samp, 'M31':m31_samp}, bins=rbins,
                    diff=False)

    ax2.plot(rbins, obs_ratios['MW'][stat], color=CB3_1[0],
            linestyle='-')
    ax2.fill_between(rbins, obs_ratios['MW']['percentile'][2],
        obs_ratios['MW']['percentile'][3], color=CB3_1[0], alpha=0.25, linewidth=0)
    ax2.fill_between(rbins, obs_ratios['MW']['percentile'][0],
        obs_ratios['MW']['percentile'][1], color=CB3_1[0], alpha=0.25, linewidth=0)
    a = np.empty(len(rbins))
    a.fill(1)
    ax2.plot(rbins, a, color='k', linestyle=':', alpha=0.4)
    ax2.set_ylim(0, 2)

    ax3.plot(rbins, obs_ratios['M31'][stat], color=CB3_1[0],
            linestyle='--')
    ax3.fill_between(rbins, obs_ratios['M31']['percentile'][2],
        obs_ratios['M31']['percentile'][3], color=CB3_1[0], alpha=0.25, linewidth=0)
    ax3.fill_between(rbins, obs_ratios['M31']['percentile'][0],
        obs_ratios['M31']['percentile'][1], color=CB3_1[0], alpha=0.25, linewidth=0)
    a = np.empty(len(rbins))
    a.fill(1)
    ax3.plot(rbins, a, color='k', linestyle=':', alpha=0.4)
    ax3.set_ylim(0, 2)


    font = {'size'   : 23}
    plt.rc('font', **font)
    ax1.legend(loc=2, fontsize=22, title=r'satellite M$_* > 10^5$ M$_{\odot}$',
        borderaxespad=0.6, handlelength=1.2)

    font = {'size'   : 18}
    plt.rc('font', **font)
    ax2.legend(loc=4, title='all hosts/MW  ', borderaxespad=0.1)
    ax3.legend(loc=1, title='all hosts/M31', borderaxespad=0.5)

    ax1.set_ylim(0,28)
    ax1.set_yticks(np.arange(5,30,5), minor=False)
    for ax in [ax2, ax3]:
        ax.set_yticks(np.arange(0,2.25,.25), minor=True)
        ax.set_yticks([0.5, 1, 1.5], minor=False)
        ax.set_yticklabels(['0.5', '', '1.5'])
    for ax in [ax1,ax2]:
        ax.tick_params(axis='x', which='major', labelbottom=False)
    for ax in [ax1,ax2,ax3]:
        ax.set_xlim(0,300)
        ax.set_xticks([25,50,75,125,150,175,225,250,275], minor=True)
        ax.set_xticks([0,100,200,300], minor=False)
        ax.tick_params(axis='both', which='major', labelsize=22)

    ax1.set_ylabel(r'N$_{\rm sat}(<\rm{d})$', fontsize=26)
    fig.text(0.01, 0.25, r'N$_{\rm sim}$/N$_{\rm obs}$', va='center', rotation='vertical', fontsize=26)
    ax3.set_xlabel('Distance from host [kpc]', fontsize=26)

    plt.show()
    if figdir:
        fig.savefig('{}main_poster_fig.pdf'.format(figdir))
        plt.close(fig)

def lowres_normed_sub(
    lowres_sat, hires_sat, mask_key, MW_data=False, M31_data=False,
    norm=False, figdir=None, stat='median', mass_cuts=None, bins=None,
    all_sims=True):
    """
    Plots cumulative profiles normaized (lowres to hires) at 2 mass cuts.
    """
    if bins is None:
        bins = lowres_sat.r_bins

    if not mass_cuts:
        mass_cuts = [lowres_sat.star_mass[0], lowres_sat.star_mass[0]*10, lowres_sat.star_mass[0]*100]
    
    fig = plt.figure(figsize=(6, 9))
    fig.set_tight_layout(False)
    gs = gridspec.GridSpec(len(mass_cuts), 1)

    ax1 = fig.add_subplot(gs[0])
    ax1.tick_params(axis='both', which='major', labelsize=16)

    ax2 = fig.add_subplot(gs[1], sharex=ax1, sharey=ax1)
    ax2.tick_params(axis='both', which='major', labelsize=16)

    fig.subplots_adjust(left=0.2, bottom=0.1, hspace=0)
    axlist = [ax1, ax2]#, ax3]
    for ax, mass_cut in zip(axlist, mass_cuts):
        if mask_key == 'star.mass':
            if not mass_cut == lowres_sat.star_mass[0]:
                # reset sm limits
                lowres_sat.star_mass = [mass_cut, 1e10]
                lowres_sat.tree_mask = sio.mask_tree(lowres_sat)
                lowres_sat.hal_label = sio.hal_label_names(lowres_sat)

                hires_sat.star_mass = [mass_cut, 1e10]
                hires_sat.tree_mask = sio.mask_tree(hires_sat)
                hires_sat.hal_label = sio.hal_label_names(hires_sat)

        elif mask_key == 'vel.circ.max':
            if not mass_cut == lowres_sat.vel_circ_max:
                lowres_sat.vel_circ_max = mass_cut
                lowres_sat.tree_mask = sio.mask_tree(lowres_sat)
                hires_sat.vel_circ_max = mass_cut
                hires_sat.tree_mask = sio.mask_tree(hires_sat)

        elif mask_key == 'v.peak':
            if not mass_cut == lowres_sat.v_peak:
                lowres_sat.v_peak = mass_cut
                lowres_sat.tree_mask = sio.mask_tree(lowres_sat)
                hires_sat.v_peak = mass_cut
                hires_sat.tree_mask = sio.mask_tree(hires_sat)

        elif mask_key == 'mass.peak':
            if not mass_cut == lowres_sat.mass_peak:
                lowres_sat.mass_peak = mass_cut
                lowres_sat.tree_mask = sio.mask_tree(lowres_sat)
                hires_sat.mass_peak = mass_cut
                hires_sat.tree_mask = sio.mask_tree(hires_sat)

        ratio_dict = spa.coadd_num_v_radius_bin_ratio(hires_sat, lowres_sat, 
            mask_key=mask_key, bins=bins, norm=norm, diff=False, all_sims=all_sims)

        if all_sims:
            ax.fill_between(bins, ratio_dict['percentile'][2],
                ratio_dict['percentile'][3], color='blue', alpha=0.25, linewidth=0)
            ax.fill_between(bins, ratio_dict['percentile'][0],
                ratio_dict['percentile'][1], color='blue', alpha=0.25, linewidth=0)
            ax.plot(bins, ratio_dict[stat], color='blue', linestyle='-')
        else:
            for i,host in enumerate(ratio_dict):
                ax.plot(bins, host, linestyle='-', label=hires_sat.hal_name[i])

        a = np.empty(len(bins))
        a.fill(1)
        ax.plot(bins, a, color='k', linestyle=':', alpha=0.35)

    if mask_key == 'v.peak':
        ax1.legend(loc=4, fontsize=18, title=r'v$_{peak}$'+' > {} km/s'.format(mass_cuts[0]))
        ax2.legend(loc=4, fontsize=18, title=r'v$_{peak}$'+' > {} km/s'.format(mass_cuts[1]))
    elif mask_key == 'mass.peak':
        m_mag = math.floor(np.log10(mass_cuts[0]))
        m_coeff = mass_cuts[0]/10**(m_mag)
        m_str = 'm$_{{peak}}$ > {:.1f} $\\times$ 10$^{}$ M$_{{\\odot}}$'.format(m_coeff, m_mag)
        ax1.legend(loc=4, fontsize=18, title=m_str)
        m_mag = math.floor(np.log10(mass_cuts[1]))
        m_coeff = mass_cuts[1]/10**(m_mag)
        m_str = 'm$_{{peak}}$ > {:.1f} $\\times$ 10$^{{{:.0f}}}$ M$_{{\\odot}}$'.format(m_coeff, m_mag)
        ax2.legend(loc=4, fontsize=18, title=m_str)
    elif mask_key == 'star.mass':
        m_mag = math.floor(np.log10(mass_cuts[0]))
        m_coeff = mass_cuts[0]/10**(m_mag)
        m_str = 'M$_{{*}}$ > {:.1f} $\\times$ 10$^{}$ M$_{{\\odot}}$'.format(m_coeff, m_mag)
        ax1.legend(loc=4, fontsize=18, title=m_str)
        m_mag = math.floor(np.log10(mass_cuts[1]))
        m_coeff = mass_cuts[1]/10**(m_mag)
        m_str = 'M$_{{*}}$ > {:.1f} $\\times$ 10$^{{}}$ M$_{{\\odot}}$'.format(m_coeff)#, m_mag)
        ax2.legend(loc=4, fontsize=18, title=m_str)

    ax1.tick_params(axis='x', which='major', labelbottom=False)
    ax1.set_ylabel('N$_{LR}$(<D)/N$_{HR}$(<D)', fontsize=22)
    ax2.set_ylabel('N$_{LR}$(<D)/N$_{HR}$(<D)', fontsize=22)
    ax2.set_xlabel('Distance from host [kpc]', fontsize=22)

    ax1.set_ylim(0,1.5)
    ax1.set_xlim(0,lowres_sat.r_range[1])
    ax2.set_ylim(0,1.5)
    ax2.set_xlim(0,lowres_sat.r_range[1])

    plt.show()

    if figdir:
        fig.savefig(figdir+'lowres_{}_z{}_{}.pdf'.format(stat, max(lowres_sat.redshift), mask_key))
        plt.close(fig)

    return fig

def nsat_v_time(sat, mask_key, radius_limit=None, obs_ratio=False):
    '''
    Plot the total number of satellites within the distance range of sat at each
    redshift.
    '''
    if radius_limit:
        radius_bin = [radius_limit]
    if not radius_limit:
        radius_bin = [sat.r_range[1]]

    n_sats, obs_dist = spa.cumulative_distance(sat, mask_key, bins=radius_bin, norm=False)

    fig = plt.figure()
    for i, hal_name in enumerate(n_sats.keys()):
        n_sat_new = [snap[0] for snap in n_sats[hal_name]]
        if obs_ratio:
            pass
            #MW_norm = obs_dist['MW'][0]
            #M31_norm = obs_dist['M31'][0]
            #plt.plot(sat.redshift, n_sat_new/MW_norm, color=CB_color_cycle[i],
            #     label=sat.hal_label[mask_key][i]+' MW ratio', alpha=0.9, linestyle='-')
            #plt.plot(sat.redshift, n_sat_new/M31_norm, color=CB_color_cycle[i],
            #     label=sat.hal_label[mask_key][i]+' M31 ratio', alpha=0.9, linestyle='--')
        else:
            plt.plot(sat.redshift, n_sat_new, color=CB_color_cycle[i],
                 label=sat.hal_label[mask_key][i], alpha=0.8)

    plt.xlabel('Redshift [z]')
    plt.ylabel('$N_{sat}$'+'(< {} kpc)'.format(radius_limit))
    plt.legend(handlelength=2)
    plt.show()
    
    return fig

def nsat_v_time2(sat_list, mask_keys):
    '''
    Plot the total number of satellites within the distance range of each sat in
    sat_list at each redshift. All on same plot, needs different linestyles.
    sat_list and mask_keys must be in the same order.
    '''
    plt.figure()
    for i, (sat, mask_key) in enumerate(zip(sat_list, mask_keys)):
        n_sats = sio.loop_hal(sat,
                        mask_key,
                        mf.cumulative_prop,
                        **{'hal_property':'host.distance.total',
                        'bins':[sat.r_range[1]],
                        'above':False,
                        'normalized':False})

        for j, hal_name in enumerate(n_sats.keys()):
            n_sat_new = [snap[0] for snap in n_sats[hal_name]]
            plt.plot(sat.redshift, n_sat_new, color=CB_color_cycle[j], linestyle=line_style[i], label=sat.hal_label[mask_key][i])

    plt.xlabel('Redshift [z]')
    plt.ylabel('$N_{sat}(< 350 kpc)$')
    plt.legend(handlelength=2)
    plt.show()

def nsat_vs_host_mass(sat_m12, sat_lg=None, mask_key='star.mass',
    mass_kind='star', fig_dir=None, lowres=False):
    """
    Plots number of satellites vs. host mass. Host masses have been hard coded
    from particle data at redshift 0.
    """
    # from host_properties.ipynb:
    m12_mass = {'star':{'m12i': 55227167000.0, 'm12f': 69300440000.0, 
                        'm12m': 99824200000.0, 'm12b': 72977000000.0,
                        'm12c': 50697490000.0, 'm12w': 48222050000.0, 
                        'm12r': 14670503000.0, 'm12z': 17643244000.0},
                'halo':{'m12i': 1040032595968.0, 'm12f': 1490549997568.0,
                        'm12m': 1333988950016.0, 'm12b': 1253360140288.0,
                        'm12c': 1194052026368.0, 'm12w': 972490276864.0,
                        'm12r': 1002674847744.0, 'm12z': 833242267648.0},
                '200m':{'m12i': 1180011585536.0, 'm12f': 1709764485120.0,
                        'm12m': 1584862093312.0, 'm12b': 1434223239168.0,
                        'm12c': 1352860315648.0, 'm12w': 1076698517504.0,
                        'm12r': 1103010316288.0, 'm12z': 924908029952.0}}

    lg_mass = {'star': {'Romeo':58674156000.0, 'Juliet':33552361000.0, 
                        'Thelma':63200907000.0, 'Louise':22782290000.0},
               'halo': {'Romeo':1134460010496.0, 'Juliet':973228998656.0,
                        'Thelma':1221716213760.0, 'Louise':999390183424.0},
                '200m':{'Romeo': 1320067047424.0, 'Juliet': 1104791445504.0,
                        'Thelma': 1432856199168.0, 'Louise': 1152683382784.0}}
    m12_mass_lowres = {'star': {'m12i LR': 55227167000.0, 'm12f LR': 69300440000.0,
                                'm12m LR': 99824200000.0, 'm12b LR': 72977000000.0,
                                'm12c LR': 50697490000.0, 'm12w LR': 48222050000.0,
                                'm12r LR': 14670503000.0, 'm12z LR': 17643244000.0},
                       'halo': {'m12i LR': 1040036200448.0, 'm12f LR': 1490583683072.0,
                                'm12m LR': 1333989343232.0, 'm12b LR': 1253356863488.0,
                                'm12c LR': 1194053206016.0, 'm12w LR': 972494798848.0,
                                'm12r LR': 1002674323456.0, 'm12z LR': 833129218048.0}}


    if sat_lg:
        sat_list = [sat_m12, sat_lg]
        mass_list = [m12_mass, lg_mass]
    else:
        if lowres:
            sat_list = [sat_m12]
            mass_list = [m12_mass_lowres]
        else:
            sat_list = [sat_m12]
            mass_list = [m12_mass]

    cc50 = []
    err50_up = []
    err50_low = []
    cc100 = []
    err100_up = []
    err100_low = []
    #cc300 = []
    #err300_up = []
    #err300_low = []
    mass_ = []

    fig, ax = plt.subplots(1,1, figsize=(7,6))
    fig.set_tight_layout(False)
    fig.subplots_adjust(left=0.15, bottom=0.16, right=0.95, top=0.95)
    for sat, masses in zip(sat_list, mass_list):
        n_sats_50, obs_dist = spa.cumulative_distance(sat, mask_key, bins=[50],
                                                    coadd=True, norm=False)
        n_sats_100, obs_dist = spa.cumulative_distance(sat, mask_key, bins=[100],
                                                    coadd=True, norm=False)
        #n_sats_300, obs_dist = spa.cumulative_distance(sat, mask_key, bins=[300],
        #                                            coadd=True, norm=False)

        for j, hal_name in enumerate(n_sats_50.keys()):
            if mass_kind == 'star':
                mass = masses['star'][hal_name]/1e10
                mw_mass = 5e10/1e10
                m31_mass = 1e11/1e10
                x_label = r'Host M$_{*}$ [$10^{10}$ M$_{\odot}$]'
                fit_pts = np.arange(1, 13, 1)
                x_lims = (1, 12)
                x_scale = 'linear'
            elif mass_kind == 'sm.hm':
                mass = masses['star'][hal_name]/masses['halo'][hal_name]
                x_label = r'Host M$_{*}$/M$_{200m}$'
                x_lims = (0, .11)
                x_scale = 'linear'
            elif mass_kind == 'dark':
                mass = masses['halo'][hal_name]
                x_label = r'Host M$_{h}$ [M$_{\odot}$]'
                x_lims = (7.5e11, 1.8e12)
                x_scale = 'log'
            elif mass_kind == '200m':
                mass = masses['200m'][hal_name]/1e12
                mw_mass = 1.4
                m31_mass = 1.6
                x_label = r'Host M$_{\rm 200m}$ [$10^{12}$ M$_{\odot}$]'
                fit_pts = np.arange(0.85, 1.85, 0.1)
                x_lims = (0.85, 1.75)
                x_scale = 'linear'

            '''
            n300_err = (n_sats_300[hal_name]['median']-n_sats_300[hal_name]['percentile'][0],
                        n_sats_300[hal_name]['percentile'][1]-n_sats_300[hal_name]['median'])
            plt.errorbar(mass, n_sats_300[hal_name]['median'], n300_err, capsize=0,
                        color='g', alpha=0.6)
            sim300, = plt.plot(mass, n_sats_300[hal_name]['median'], color='g',
                      marker='.', markersize=12)
            err300_low.append(n300_err[0][0])
            err300_up.append(n300_err[1][0])
            cc300.append(n_sats_300[hal_name]['median'][0])
            '''

            n100_err = (n_sats_100[hal_name]['median']-n_sats_100[hal_name]['percentile'][0],
                        n_sats_100[hal_name]['percentile'][1]-n_sats_100[hal_name]['median'])
            plt.errorbar(mass, n_sats_100[hal_name]['median'], n100_err, capsize=0,
                        color='#005AB5', alpha=0.6)
            sim100, = plt.plot(mass, n_sats_100[hal_name]['median'], color='#005AB5',
                    marker='.', markersize=14)
            err100_low.append(n100_err[0][0])
            err100_up.append(n100_err[1][0])
            cc100.append(n_sats_100[hal_name]['median'][0])

            n50_err = (n_sats_50[hal_name]['median']-n_sats_50[hal_name]['percentile'][0],
                        n_sats_50[hal_name]['percentile'][1]-n_sats_50[hal_name]['median'])
            plt.errorbar(mass, n_sats_50[hal_name]['median'], n50_err, alpha=0.6,
                        color='#DC3220', capsize=0)
            sim50, = plt.plot(mass, n_sats_50[hal_name]['median'], color='#DC3220',
                    marker='.', markersize=14)
            err50_low.append(n50_err[0][0])
            err50_up.append(n50_err[1][0])
            cc50.append(n_sats_50[hal_name]['median'][0])
            mass_.append(mass)

    #sim50 = ax.scatter(mass_, cc50, c=color_by_mass, s=14, cmap=cm.plasma)
    #sim100 = ax.scatter(mass_, cc100, c=color_by_mass, s=14, cmap=cm.plasma)
    # Add a colorbar
    #fig.colorbar(sim50, ax=ax)

    # plot linear fits to simulation data
    def lin_fit(x, a, b):
        return a*x + b

    sig50 = np.nanmean([err50_up, err50_low], axis=0)
    for i,j in enumerate(sig50):
        if j == 0:
            sig50[i] = 1
    popt50, pcov50 = curve_fit(f=lin_fit, xdata=np.array(mass_), ydata=np.array(cc50), sigma=sig50)
    print('50 kpc fit parameters: ', popt50)
    plt.plot(fit_pts, lin_fit(np.array(fit_pts), *popt50), color='#DC3220', alpha=0.4, linestyle='--')

    sig100 = np.nanmean([err100_up, err100_low], axis=0)
    for i,j in enumerate(sig100):
        if j == 0:
            sig100[i] = 1
    popt100, pcov100 = curve_fit(f=lin_fit, xdata=np.array(mass_), ydata=np.array(cc100), sigma=sig100)
    print('100 kpc fit parameters: ', popt100)
    plt.plot(fit_pts, lin_fit(np.array(fit_pts), *popt100), color='#005AB5', alpha=0.4, linestyle='--')

    '''
    sig300 = np.nanmean([err300_up, err300_low], axis=0)
    for i,j in enumerate(sig300):
        if j == 0:
            sig300[i] = 1
    popt300, pcov300 = curve_fit(f=lin_fit, xdata=np.array(mass_), ydata=np.array(cc300), sigma=sig300)
    plt.plot(fit_pts, lin_fit(np.array(fit_pts), *popt300), color='g', alpha=0.4, linestyle='--')
    '''
    # plot MW and M31 data
    plt.errorbar(mw_mass, 6, yerr=[[0], [1]], alpha=0.7, color='k', capsize=0)
    mw100, = plt.plot(mw_mass, 6, color='#005AB5', marker='^', markersize=11, 
             label='D$_{MW}<100$ kpc', markeredgecolor='k', markeredgewidth=2)
    plt.errorbar(mw_mass, 1, yerr=[[1], [0]], alpha=0.7, color='k', capsize=0)
    mw50, = plt.plot(mw_mass, 1, color='#DC3220', marker='^', markersize=11, 
            label='D$_{MW}<50$ kpc', markeredgecolor='k', markeredgewidth=2)
    
    plt.errorbar(m31_mass, 5, yerr=[[1], [1]], alpha=0.7, color='k', capsize=0)
    m31100, = plt.plot(m31_mass, 5, color='#005AB5', marker='s', markersize=11, 
                        label='D$_{M31}<100$ kpc',
                        markeredgecolor='k', markeredgewidth=2)
    plt.errorbar(m31_mass, 2, yerr=[[0], [1]], alpha=0.7, color='k', capsize=0)
    m3150, = plt.plot(m31_mass, 2, color='#DC3220', marker='s', markersize=11, 
                        label='D$_{M31}<50$ kpc',
                        markeredgecolor='k', markeredgewidth=2)

    # custom legend and formatting
    plt.xlabel(x_label, fontsize=26)
    plt.xscale(x_scale)
    if x_lims:
        plt.xlim(x_lims)
    if mass_kind == 'star':
        plt.xlim(1,11)
        ax.set_xticks(np.arange(1,11,0.5), minor=True)
        ax.set_xticks([2,4,6,8,10], minor=False)
    plt.ylim((0,11))
    ax.set_yticks(np.arange(0,11,1), minor=True)
    plt.yticks([2,4,6,8,10])
    ax.tick_params(axis='both', which='major', labelsize=26)
    ax.tick_params(axis='x', which='minor', labelbottom=False)
    plt.ylabel(r'N$_{\rm sat}(<\rm{d})$', fontsize=28)
    plt.legend(handles=(sim50, sim100, mw50, mw100, m3150, m31100),
        labels=(r'd$_{\rm sim}<$50 kpc', r'd$_{\rm sim}<$100 kpc', r'MW',
                r'MW', r'M31', r'M31'),
        fontsize=22, loc=1, ncol=3, handlelength=0.1, handletextpad=0.6,
        borderaxespad=0.18)
    plt.show()

    print(mass_kind, '50 kpc', np.corrcoef(mass_, cc50))
    print(mass_kind, '100 kpc', np.corrcoef(mass_, cc100))
    #print(mass_kind, '300 kpc', np.corrcoef(mass_, cc300))

    if fig_dir:
        fig.savefig('{}nsat_vs_host_{}_mass_z{}_final.pdf'.format(fig_dir, mass_kind, max(sat.redshift)))
        plt.close(fig)

def nsat_vs_host_mass_colorbar(sat_m12, sat_lg=None, mask_key='star.mass',
    mass_kind='star', fig_dir=None, lowres=False, color_points_by=None):
    """
    Plots number of satellites vs. host mass. Host masses have been hard coded
    from particle data at redshift 0.
    """
    # from host_properties.ipynb:
    m12_mass = {'star':{'m12i': 55227167000.0, 'm12f': 69300440000.0, 
                        'm12m': 99824200000.0, 'm12b': 72977000000.0,
                        'm12c': 50697490000.0, 'm12w': 48222050000.0, 
                        'm12r': 14670503000.0, 'm12z': 17643244000.0},
                'halo':{'m12i': 1040032595968.0, 'm12f': 1490549997568.0,
                        'm12m': 1333988950016.0, 'm12b': 1253360140288.0,
                        'm12c': 1194052026368.0, 'm12w': 972490276864.0,
                        'm12r': 1002674847744.0, 'm12z': 833242267648.0},
                '200m':{'m12i': 1180011585536.0, 'm12f': 1709764485120.0,
                        'm12m': 1584862093312.0, 'm12b': 1434223239168.0,
                        'm12c': 1352860315648.0, 'm12w': 1076698517504.0,
                        'm12r': 1103010316288.0, 'm12z': 924908029952.0}}

    lg_mass = {'star': {'Romeo':58674156000.0, 'Juliet':33552361000.0, 
                        'Thelma':63200907000.0, 'Louise':22782290000.0},
               'halo': {'Romeo':1134460010496.0, 'Juliet':973228998656.0,
                        'Thelma':1221716213760.0, 'Louise':999390183424.0},
                '200m':{'Romeo': 1320067047424.0, 'Juliet': 1104791445504.0,
                        'Thelma': 1432856199168.0, 'Louise': 1152683382784.0}}

    if sat_lg:
        sat_list = [sat_m12, sat_lg]
        mass_list = [m12_mass, lg_mass]
    else:
        sat_list = [sat_m12]
        mass_list = [m12_mass]

    cc50 = []
    err50_up = []
    err50_low = []
    cc100 = []
    err100_up = []
    err100_low = []

    mass_ = []
    color_by_mass = []

    fig, ax = plt.subplots(1,1, figsize=(9,6))
    fig.set_tight_layout(False)
    fig.subplots_adjust(left=0.15, bottom=0.16, right=0.95, top=0.95)
    for sat, masses in zip(sat_list, mass_list):
        n_sats_50, obs_dist = spa.cumulative_distance(sat, mask_key, bins=[50],
                                                    coadd=True, norm=False)
        n_sats_100, obs_dist = spa.cumulative_distance(sat, mask_key, bins=[100],
                                                    coadd=True, norm=False)
        for j, hal_name in enumerate(n_sats_50.keys()):
            if mass_kind == 'star':
                mass = masses['star'][hal_name]/1e10
                mw_mass = 5e10/1e10
                m31_mass = 1e11/1e10
                x_label = r'Host M$_{*}$ [$10^{10}$ M$_{\odot}$]'
                fit_pts = np.arange(1, 13, 1)
                x_lims = (1, 12)
                x_scale = 'linear'
            elif mass_kind == '200m':
                mass = masses['200m'][hal_name]/1e12
                mw_mass = 1.4
                m31_mass = 1.6
                x_label = r'Host M$_{\rm 200m}$ [$10^{12}$ M$_{\odot}$]'
                fit_pts = np.arange(0.85, 1.85, 0.1)
                x_lims = (0.85, 1.75)
                x_scale = 'linear'

            color_by_mass.append(masses['star'][hal_name]/1e10)

            n100_err = (n_sats_100[hal_name]['median']-n_sats_100[hal_name]['percentile'][0],
                        n_sats_100[hal_name]['percentile'][1]-n_sats_100[hal_name]['median'])
            #plt.errorbar(mass, n_sats_100[hal_name]['median'], n100_err, capsize=0,
            #            color='#005AB5', alpha=0.6)
            err100_low.append(n100_err[0][0])
            err100_up.append(n100_err[1][0])
            cc100.append(n_sats_100[hal_name]['median'][0])

            n50_err = (n_sats_50[hal_name]['median']-n_sats_50[hal_name]['percentile'][0],
                        n_sats_50[hal_name]['percentile'][1]-n_sats_50[hal_name]['median'])
            #plt.errorbar(mass, n_sats_50[hal_name]['median'], n50_err, alpha=0.6,
            #            color='#DC3220', capsize=0)
            err50_low.append(n50_err[0][0])
            err50_up.append(n50_err[1][0])
            cc50.append(n_sats_50[hal_name]['median'][0])
            mass_.append(mass)


    sim50 = ax.scatter(mass_, cc50, c=color_by_mass, s=144, marker='o', cmap=cm.plasma)
    sim100 = ax.scatter(mass_, cc100, c=color_by_mass, s=144, marker='^', cmap=cm.plasma)
    print(cc50, cc100)
    # Add a colorbar
    cbar = fig.colorbar(sim50, ax=ax, label=r'Host M$_{*}$ [$10^{10}$ M$_{\odot}$]')
    #cbar.ax.set_ylabel(r'Host M$_{*}$ [$10^{10}$ M$_{\odot}$]', rotation=270)

    # plot linear fits to simulation data
    def lin_fit(x, a, b):
        return a*x + b

    sig50 = np.nanmean([err50_up, err50_low], axis=0)
    for i,j in enumerate(sig50):
        if j == 0:
            sig50[i] = 1
    popt50, pcov50 = curve_fit(f=lin_fit, xdata=np.array(mass_), ydata=np.array(cc50), sigma=sig50)
    print(popt50)
    plt.plot(fit_pts, lin_fit(np.array(fit_pts), *popt50), color='k', alpha=0.4, linestyle='--')

    sig100 = np.nanmean([err100_up, err100_low], axis=0)
    for i,j in enumerate(sig100):
        if j == 0:
            sig100[i] = 1
    popt100, pcov100 = curve_fit(f=lin_fit, xdata=np.array(mass_), ydata=np.array(cc100), sigma=sig100)
    plt.plot(fit_pts, lin_fit(np.array(fit_pts), *popt100), color='k', alpha=0.4, linestyle='--')

    # plot MW and M31 data
    #plt.errorbar(mw_mass, 6, yerr=[[0], [1]], alpha=0.7, color='k', capsize=0)
    mw100, = plt.plot(mw_mass, 6, color='#D8261B', marker='^', markersize=12, 
             label='D$_{MW}<100$ kpc', markeredgecolor='k', markeredgewidth=2)
    #plt.errorbar(mw_mass, 1, yerr=[[1], [0]], alpha=0.7, color='k', capsize=0)
    mw50, = plt.plot(mw_mass, 1, color='#D8261B', markersize=12, marker='o',
            label='D$_{MW}<50$ kpc', markeredgecolor='k', markeredgewidth=2)
    
    #plt.errorbar(m31_mass, 5, yerr=[[1], [1]], alpha=0.7, color='k', capsize=0)
    m31100, = plt.plot(m31_mass, 5, color='#1E8875', marker='^', markersize=12, 
                        label='D$_{M31}<100$ kpc',
                        markeredgecolor='k', markeredgewidth=2)
    #plt.errorbar(m31_mass, 2, yerr=[[0], [1]], alpha=0.7, color='k', capsize=0)
    m3150, = plt.plot(m31_mass, 2, color='#1E8875', markersize=12, 
                        label='D$_{M31}<50$ kpc', marker='o',
                        markeredgecolor='k', markeredgewidth=2)

    # custom legend and formatting
    plt.xlabel(x_label, fontsize=26)
    plt.xscale(x_scale)
    if x_lims:
        plt.xlim(x_lims)
    if mass_kind == 'star':
        plt.xlim(1,11)
        ax.set_xticks(np.arange(1,11,0.5), minor=True)
        ax.set_xticks([2,4,6,8,10], minor=False)
    plt.ylim((0,11))
    ax.set_yticks(np.arange(0,11,1), minor=True)
    plt.yticks([2,4,6,8,10])
    ax.tick_params(axis='both', which='major', labelsize=26)
    ax.tick_params(axis='x', which='minor', labelbottom=False)
    plt.ylabel(r'N$_{\rm sat}(<\rm{d})$', fontsize=28)
    plt.legend(handles=(sim50, sim100, mw50, mw100, m3150, m31100),
        labels=(r'd$_{\rm sim}<$50 kpc', r'd$_{\rm sim}<$100 kpc', r'MW',
                r'MW', r'M31', r'M31'),
        fontsize=22, loc=1, ncol=3, handlelength=0.1, handletextpad=0.6,
        borderaxespad=0.18)
    plt.show()

    print(mass_kind, '50 kpc', np.corrcoef(mass_, cc50))
    print(mass_kind, '100 kpc', np.corrcoef(mass_, cc100))
    #print(mass_kind, '300 kpc', np.corrcoef(mass_, cc300))

    if fig_dir:
        fig.savefig('{}nsat_vs_host_{}_mass_z{}_final.pdf'.format(fig_dir, mass_kind, max(sat.redshift)))
        plt.close(fig)

def nsat_vs_stellar_mass(sat, mask_key, star_mass_lims=None, radius_limit=None, 
    norm=False, obs_sys=None, fig_dir=None):
    '''
    Plot the coadded number of satellites within the distance limit, for each
    stellar mass cut. Each host gets its own plot.
    '''
    cum_sats_v_sm, obs_norm = spa.nsat_vs_stellar_mass(sat, mask_key,
                                  star_mass_limits=star_mass_lims,
                                  radius_limit=radius_limit, obs_sys=obs_sys)
    
    if norm:
        for j, hal_name in enumerate(cum_sats_v_sm.keys()):
            fig = plt.figure()
            plt.fill_between(star_mass_lims,
                             cum_sats_v_sm[hal_name]['percent_95'][0]/obs_norm,
                             cum_sats_v_sm[hal_name]['percent_95'][1]/obs_norm,
                             alpha=0.3, color=CB_color_cycle[j], linewidth=0)

            plt.fill_between(star_mass_lims,
                             cum_sats_v_sm[hal_name]['percent_68'][0]/obs_norm,
                             cum_sats_v_sm[hal_name]['percent_68'][1]/obs_norm,
                             alpha=0.3, color=CB_color_cycle[j], linewidth=0)

            plt.plot(star_mass_lims, cum_sats_v_sm[hal_name]['mean']/obs_norm,
                     color=CB_color_cycle[j], label=hal_name+' '+obs_sys+'-normed')
            
            plt.xlabel(r'Minimum satellite $M_{*}$ [$M_{\odot}$]', fontsize=18)
            plt.xscale('log')
            plt.ylabel('$N_{sat}$'+'(< {} kpc)'.format(radius_limit), fontsize=18)
            plt.legend(handlelength=2, fontsize=14)
            plt.show()
            if fig_dir:
                fig.savefig('{}coadd_N_{}norm_vs_sm_{}kpc_{}.pdf'.format(fig_dir, obs_sys, radius_limit, hal_name))
                plt.close(fig)
        
    else:    
        for j, hal_name in enumerate(cum_sats_v_sm.keys()):
            fig = plt.figure()
            plt.fill_between(star_mass_lims,
                             cum_sats_v_sm[hal_name]['percent_95'][0],
                             cum_sats_v_sm[hal_name]['percent_95'][1],
                             alpha=0.3, color=CB_color_cycle[j], linewidth=0)

            plt.fill_between(star_mass_lims,
                             cum_sats_v_sm[hal_name]['percent_68'][0],
                             cum_sats_v_sm[hal_name]['percent_68'][1],
                             alpha=0.4, color=CB_color_cycle[j], linewidth=0)

            plt.plot(star_mass_lims, cum_sats_v_sm[hal_name]['mean'],
                     color=CB_color_cycle[j], label=hal_name)
            plt.plot(star_mass_lims, obs_norm, color='k', label=obs_sys)
            
            plt.xlabel(r'Stellar mass [$M_{\odot}$]')
            plt.xscale('log')
            plt.ylabel('$N_{sat}$'+'(< {} kpc)'.format(radius_limit))
            plt.legend(handlelength=2)
            plt.show()
            if fig_dir:
                fig.savefig('{}coadd_N_vs_sm_{}kpc_{}_{}.pdf'.format(fig_dir, radius_limit, obs_sys, hal_name))
                plt.close(fig)

def nsat_vs_stellar_mass_all_hosts(sat, mask_key, star_mass_limits=None, 
    radius_limit=None, norm=False, fig_dir=None, stat='median'):
    '''
    Plot the coadded number of satellites within the distance limit for all 
    hosts togehter, at each stellar mass cut.
    '''
    if sat.sat_type == 'tree':
        plot_name = 'm12'
    elif sat.sat_type == 'tree.lg':
        plot_name = 'LG'

    cum_sats_v_sm, obs_norm = spa.nsat_vs_stellar_mass_all_hosts(sat, mask_key,
                                  star_mass_limits=star_mass_limits,
                                  radius_limit=radius_limit)

    low_68 = [percent[0] for percent in cum_sats_v_sm['percentile']]
    high_68 = [percent[1] for percent in cum_sats_v_sm['percentile']]
    low_95 = [percent[2] for percent in cum_sats_v_sm['percentile']]
    high_95 = [percent[3] for percent in cum_sats_v_sm['percentile']]    

    fig = plt.figure(figsize=(7, 9))
    fig.set_tight_layout(False)
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, .5])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    fig.subplots_adjust(hspace=0)
    ax1.tick_params(axis='both', which='major', labelsize=20)
    ax2.tick_params(axis='both', which='major', labelsize=20)

    ax1.fill_between(star_mass_limits,
                     low_95,
                     high_95,
                     alpha=0.25, color=CB3_1[0], linewidth=0)
    ax1.fill_between(star_mass_limits,
                        low_68,
                        high_68,
                        alpha=0.25, color=CB3_1[0], linewidth=0)
    ax1.plot(star_mass_limits, cum_sats_v_sm[stat],
                color=CB3_1[0], linewidth=3, label='all hosts')
    ax1.plot(star_mass_limits, obs_norm['MW'], color='k', linestyle='-', label='Milky Way')
    ax1.plot(star_mass_limits, obs_norm['M31'], color='k', linestyle='--', label='M31')
    ax1.legend(loc=1, fontsize=16)
    ax1.set_ylabel('$N_{sat}$', fontsize=22)

    ax2.plot(star_mass_limits, np.array(cum_sats_v_sm[stat])/np.array(obs_norm['MW']), color=CB3_0[0],
            label='isolated hosts/MW', linestyle='-')
    ax2.plot(star_mass_limits, np.array(cum_sats_v_sm[stat])/np.array(obs_norm['M31']), color=CB3_0[0],
            label='isolated hosts/M31', linestyle='--')

    ax2.legend(loc=1, fontsize=16)
    ax2.set_ylabel('$N_{sat}/N_{observed}$', fontsize=22)

    
    plt.xlabel(r'Minimum M$_{*}$ of satellites [$M_{\odot}$]', fontsize=22)
    plt.xscale('log')
    plt.show()
    if fig_dir:
        fig.savefig('{}coadd_nsat_vs_sm_{}kpc_z{}_all_hosts.pdf'.format(fig_dir, radius_limit, max(sat.redshift)))
        plt.close(fig)

    return fig

def nsat_vs_stellar_mass_m12_and_lg(
    sat, sat_lg, mask_key, star_mass_limits=None, radius_limit=None,
    norm=False, fig_dir=None, stat='median', n_iter=1000, obs_dir='./'):
    '''
    Plot the coadded number of satellites within the distance limit, for each
    stellar mass cut.
    '''
    sm_dicts = spa.nsat_vs_stellar_mass_m12_and_lg(sat, sat_lg, mask_key, 
                    star_mass_limits, radius_limit, n_iter=n_iter, stat=stat,
                    obs_dir=obs_dir)
    all_sm_dict, m12_sm_dict, lg_sm_dict, obs_norm, obs_ratio = sm_dicts

    low_68 = [percent[0] for percent in all_sm_dict['percentile']]
    high_68 = [percent[1] for percent in all_sm_dict['percentile']]
    low_95 = [percent[2] for percent in all_sm_dict['percentile']]
    high_95 = [percent[3] for percent in all_sm_dict['percentile']]
    
    fig = plt.figure(figsize=(6, 9))
    fig.set_tight_layout(False)
    gs = gridspec.GridSpec(3, 1, height_ratios=[3.5, 1, 1])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1, sharey=ax2)
    ax1.tick_params(axis='both', which='major', labelsize=22)
    ax2.tick_params(axis='both', which='major', labelsize=22)
    ax3.tick_params(axis='both', which='major', labelsize=22)
    ax1.tick_params(axis='x', which='major', labelbottom=False)
    ax2.tick_params(axis='x', which='major', labelbottom=False)
    fig.subplots_adjust(left=0.17, right=0.92, top=0.96, bottom=0.12, hspace=0)

    ax1.fill_between(star_mass_limits, low_95, high_95,
                     alpha=0.25, color=CB3_1[0], linewidth=0)
    ax1.fill_between(star_mass_limits, low_68, high_68,
                     alpha=0.25, color=CB3_1[0], linewidth=0)

    ax1.plot(star_mass_limits, all_sm_dict[stat],
                color=CB3_1[0], linewidth=3, label='all hosts')
    ax1.plot(star_mass_limits, obs_norm['MW'], color='k', linestyle='-', label='MW')
    ax1.plot(star_mass_limits, obs_norm['M31'], color='k', linestyle='--', label='M31')
    ax1.legend(loc=1, fontsize=20, title='d < {} kpc'.format(radius_limit))
    ax1.set_ylabel(r'N$_{\rm sat}$(d$<$100)', fontsize=24)

    # reset shaded regions for ratio
    low_68 = [percent[0] for percent in obs_ratio['MW']['percentile']]
    high_68 = [percent[1] for percent in obs_ratio['MW']['percentile']]
    low_95 = [percent[2] for percent in obs_ratio['MW']['percentile']]
    high_95 = [percent[3] for percent in obs_ratio['MW']['percentile']]

    a = np.empty(len(star_mass_limits))
    a.fill(1)
    ax2.plot(star_mass_limits, a, color='k', linestyle=':', alpha=0.4)

    ax2.fill_between(star_mass_limits, low_95, high_95,
                     alpha=0.25, color=CB3_1[0], linewidth=0)
    ax2.fill_between(star_mass_limits, low_68, high_68,
                     alpha=0.25, color=CB3_1[0], linewidth=0)

    ax2.plot(star_mass_limits, obs_ratio['MW'][stat], color=CB3_0[0],
            linestyle='-')

    # reset shaded regions for ratio
    low_68 = [percent[0] for percent in obs_ratio['M31']['percentile']]
    high_68 = [percent[1] for percent in obs_ratio['M31']['percentile']]
    low_95 = [percent[2] for percent in obs_ratio['M31']['percentile']]
    high_95 = [percent[3] for percent in obs_ratio['M31']['percentile']]

    ax3.fill_between(star_mass_limits, low_95, high_95,
                     alpha=0.25, color=CB3_1[0], linewidth=0)
    ax3.fill_between(star_mass_limits, low_68, high_68,
                     alpha=0.25, color=CB3_1[0], linewidth=0)
    ax3.plot(star_mass_limits, obs_ratio['M31'][stat], color=CB3_0[0],
            linestyle='--')

    ax3.plot(star_mass_limits, a, color='k', linestyle=':', alpha=0.4)

    ax1.set_ylim(0,10)
    ax1.set_yticks(np.arange(2,12,2), minor=False)
    for ax in [ax2, ax3]:
        ax.set_yticks([0.5, 1, 1.5], minor=False)
        ax.set_yticklabels(['0.5', '', '1.5'])
        ax.set_yticks(np.arange(0,2,0.25), minor=True)
        ax.set_ylim(0, 2)
    for ax in [ax1, ax2, ax3]:
        ax.set_xlim(star_mass_limits[0], star_mass_limits[-1])
    ax2.legend(labels=[None], handles=[None], loc=1, fontsize=20, borderaxespad=0.1, title='all hosts/MW ')
    ax3.legend(labels=[None], handles=[None], loc=1, fontsize=20, borderaxespad=0.1, title='all hosts/M31')
    fig.text(0.02, 0.29, r'N$_{\rm sim}$/N$_{\rm obs}$', va='center', rotation='vertical', fontsize=24)
    
    plt.xlabel(r'Minimum M$_{*}$ of satellites [M$_{\odot}$]', fontsize=23)
    plt.xscale('log')
    plt.show()
    if fig_dir:
        fig.savefig('{}nsat_vs_sm_{}kpc_z{}.pdf'.format(fig_dir, radius_limit, max(sat.redshift)))
        plt.close(fig)

    return fig

def radial_proj_coadd_m12_and_lg(
    m12_sat, lg_sat, mask_key, MW_data=False, M31_data=False, stat='median',
    saga_data=False, n_iter=1000, rbins=None, mw_shade=False, saga_bin=False,
    sm_lim=5e6):
    """
    Plot of the coadded 2D projection of radial profiles across hosts/snapshots.
    """

    if not sm_lim == m12_sat.star_mass[0]:
        m12_sat = sio.reset_sm_limits(m12_sat, lower_lim=sm_lim, obs=False)
        lg_sat = sio.reset_sm_limits(lg_sat, lower_lim=sm_lim, obs=False)
    
    if rbins is None:
        rbins = m12_sat.r_bins
    
    # simulated data
    m12_2d = sio.loop_hal(m12_sat, 'star.mass', spa.cumul_2d_proj,
                                    **{'rbins':rbins, 'n_iter':n_iter})
    lg_2d = sio.loop_hal(lg_sat, 'star.mass', spa.cumul_2d_proj,
                                    **{'rbins':rbins, 'n_iter':n_iter})

    all_coadd_2d = mf.coadd_redshift({**m12_2d, **lg_2d}, all_sims=True)
    
    # LG observational data
    m31_2d = [0, 2, 2, 3, 3, 4, 5, 5, 7, 7, 7, 9, 9, 10, 11, 11]
    mw_2d = [0, 0, 1, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4]
    
    # saga data is in the form of distance of sats from their host, vs cumulative number of sats at that distance
    saga_hosts = ['NGC 6181', 'NGC 7541', 'NGC 2543', 'PGC068743', 'NGC 7716', 'NGC 1015', 'NGC 5962', 'NGC 5750']
    saga_cumul = [[np.array([  0,  37,  47,  65,  80, 186, 198, 255, 273, 285]), np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])], 
                  [np.array([  0,  34,  43,  68, 120, 133, 259]), np.array([0, 1, 2, 3, 4, 5, 6])], 
                  [np.array([  0,  37, 246]), np.array([0, 1, 2])], 
                  [np.array([  0,  98, 119, 164, 186]), np.array([0, 1, 2, 3, 4])], 
                  [np.array([  0, 144, 201, 213]), np.array([0, 1, 2, 3])], 
                  [np.array([  0,  51, 253]), np.array([0, 1, 2])],
                  [np.array([  0,  81, 206]), np.array([0, 1, 2])],
                  [np.array([  0, 105]), np.array([0, 1])]]
    
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    fig.set_tight_layout(False)
    fig.subplots_adjust(left=0.12, bottom=0.12, right=0.96, top=0.98)

    if not mw_shade:
        plt.fill_between(rbins, all_coadd_2d['percentile'][2],
            all_coadd_2d['percentile'][3], color=CB3_1[0], alpha=0.3, linewidth=0)
        plt.fill_between(rbins, all_coadd_2d['percentile'][0],
            all_coadd_2d['percentile'][1], color=CB3_1[0], alpha=0.3, linewidth=0)#, label='all simulated hosts')

    plt.plot(rbins, all_coadd_2d[stat], color=CB3_1[0], linewidth=4, label='all simulated hosts')
    
    if saga_data:
        cm_subsection = np.linspace(0.2, 0.55, len(saga_hosts)) 
        colors = [cm.plasma(x) for x in cm_subsection]
        if saga_bin:
            sbin = [[0, 0, 1, 2, 4, 4, 4, 4, 4, 4, 6, 6, 6, 7, 8, 9], 
                    [0, 0, 1, 2, 3, 3, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6], 
                    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2], 
                    [0, 0, 0, 0, 0, 1, 2, 2, 2, 3, 4, 4, 4, 4, 4, 4], 
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 3, 3, 3, 3, 3], 
                    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2], 
                    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2], 
                    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
            for i, saga_host in enumerate(saga_hosts):
                plt.plot(np.arange(0,320,20), sbin[i], label=saga_host, 
                    color=colors[i], alpha=0.4, linewidth=2)

        else:
            for i, saga_host in enumerate(saga_hosts):
                plt.plot(saga_cumul[i][0], saga_cumul[i][1], label=saga_host, 
                    color=colors[i], alpha=0.4, linewidth=2)

    if MW_data:
        plt.plot(rbins, mw_2d, color='k', label='MW', linestyle=':', linewidth=2)
        if mw_shade:
            mw2d_perc = [[0, 0, 0, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4], 
                        [0, 0, 1, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4], 
                        [0, 0, 0, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4], 
                        [0, 1, 2, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]]
            plt.fill_between(rbins, mw2d_perc[2],
                mw2d_perc[3], color='k', alpha=0.25, linewidth=0)
            plt.fill_between(rbins, mw2d_perc[0],
                mw2d_perc[1], color='k', alpha=0.25, 
                linewidth=0)

    if M31_data:
        plt.plot(rbins, m31_2d, color='k', label='M31', linestyle='--', linewidth=2)
    
    plt.xlabel('Projected distance from host [kpc]', fontsize=22)
    plt.ylabel('N$_{sat}$(<D)', fontsize=22)
    plt.legend(loc=2, fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    ax.set_yticks(np.arange(1,12,1), minor=True)
    ax.set_yticks(np.arange(0,12,2), minor=False)
    ax.set_xticks([0,100,200,300], minor=False)
    ax.set_xticks(np.arange(25,300,25), minor=True)
    plt.xlim(0,300)
    plt.ylim(0,11.5)
    plt.show()
    
    return fig

def radial_proj_coadd_m12_and_lg_subplots(
    m12_sat, lg_sat, mask_key, MW_data=False, M31_data=False, stat='median',
    saga_data=False, n_iter=1000, rbins=None):
    """
    Plot of 2D profile projections for each host.
    """
    
    if rbins is None:
        rbins = m12_sat.r_bins

    # simulated data
    m12_2d = sio.loop_hal(m12_sat, 'star.mass', spa.cumul_2d_proj,
                                    **{'rbins':rbins, 'n_iter':n_iter})
    lg_2d = sio.loop_hal(lg_sat, 'star.mass', spa.cumul_2d_proj,
                                    **{'rbins':rbins, 'n_iter':n_iter})
    all_coadd_2d = mf.coadd_redshift({**m12_2d, **lg_2d}, all_sims=False)
    
    # observational data
    m31_2d = [0, 2, 2, 3, 3, 4, 5, 5, 7, 7, 7, 9, 9, 10, 11, 11]
    mw_2d = []
    
    # saga data is in the form of distance of sats from their host, vs cumulative number of sats at that distance
    saga_hosts = ['NGC 6181', 'NGC 7541', 'NGC 2543', 'PGC068743', 'NGC 7716', 'NGC 5750', 'NGC 5962', 'NGC 1015']
    saga_cumul = [[np.array([  0,  37,  47,  65,  80, 186, 198, 255, 273, 285]), np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])], 
                  [np.array([  0,  34,  43,  68, 120, 133, 259]), np.array([0, 1, 2, 3, 4, 5, 6])], 
                  [np.array([  0,  37, 246]), np.array([0, 1, 2])], 
                  [np.array([  0,  98, 119, 164, 186]), np.array([0, 1, 2, 3, 4])], 
                  [np.array([  0, 144, 201, 213]), np.array([0, 1, 2, 3])], 
                  [np.array([  0, 105]), np.array([0, 1])], 
                  [np.array([  0,  81, 206]), np.array([0, 1, 2])], 
                  [np.array([  0,  51, 253]), np.array([0, 1, 2])]]
    

    fig, ax = plt.subplots(nrows=3, ncols=4, sharex=True, sharey=True, figsize=(12, 9))
    fig.set_tight_layout(False)

    # order hosts from most stellar mass to least
    name_iter = ['m12m', 'm12b', 'm12f', 'Thelma', 'Romeo', 'm12i', 'm12c',
                 'm12w', 'Juliet', 'Louise', 'm12z', 'm12r']
    for i, name in enumerate(name_iter):
        if 'm12' in name:
            plot_color = CB3_1[1]
        else:
            plot_color = CB3_1[2]
        j = math.floor(i/4)
        k = i - math.floor(i/4)*4
        ax[j, k].fill_between(rbins, all_coadd_2d[name]['percentile'][2], all_coadd_2d[name]['percentile'][3], 
                              alpha=0.25, color=plot_color, linewidth=0)
        ax[j, k].fill_between(rbins, all_coadd_2d[name]['percentile'][0], all_coadd_2d[name]['percentile'][1], 
                              alpha=0.25, color=plot_color, linewidth=0)
        ax[j, k].plot(rbins, all_coadd_2d[name][stat], color=plot_color, linewidth=2, label=name)

        if MW_data:
            ax[j, k].plot(rbins, mw_2d, color='k', linestyle='-')
        if M31_data:
            ax[j, k].plot(rbins, m31_2d, color='k', linestyle='--')
            
        if saga_data:   
            saga_opacity = np.arange(.3, 1.1, .1)
            for i, saga_host in enumerate(saga_hosts):
                ax[j, k].plot(saga_cumul[i][0], saga_cumul[i][1], label=saga_host, color='#0408B5', alpha=saga_opacity[i])

        ax[j, k].legend(loc=2, fontsize=16)
        ax[j, k].tick_params(axis='both', which='major', labelsize=18)
        ax[j, k].set_xlim((-15,315))
        ax[j, k].set_ylim((-0.5,12))
        #ax[j, k].set_xticks([25, 50, 75, 125, 150, 175, 225, 250, 275], minor=True)
        ax[j, k].set_xticks([50, 150, 250], minor=True)
        ax[j, k].set_xticks([0, 100, 200, 300], minor=False)
        ax[j, k].set_yticks([0, 5, 10], minor=False)
        ax[j, k].set_yticks([1,2,3,4,6,7,8,9,11], minor=True)
        #ax[j, k].tick_params(axis='both', which='minor', bottom=False, top=False, left=False, right=False)4

    if MW_data:
        ax[0, 0].plot(rbins, mw_2d, color='k', label='MW', linestyle='-')
    if M31_data:
        ax[0, 0].plot(rbins, m31_2d, color='k', label='M31', linestyle='--')

    ax[0, 0].legend(loc=2, fontsize=16)
    fig.text(0.52, 0.02, 'Projected distance from host [kpc]', ha='center', fontsize=22)
    fig.text(0.02, 0.54, 'N$_{sat}(<D)$', va='center', rotation='vertical', fontsize=22)
    fig.subplots_adjust(left=0.1, bottom=0.12, wspace=0.1, hspace=0.1)
    plt.show()
    
    return fig

def radial_2d_vs_3d_subplots(
    m12_sat, lg_sat, mask_key, stat='median', n_iter=1000, rbins=None, rlim=150,
    M31_data=True, all_sims=False, complete_limit=1e6):
    """
    Mock PAndAS effects on measuring 3D profile in simulations.
    """
    
    if rbins is None:
        rbins = m12_sat.r_bins

    # true 3d distribution for each host
    if all_sims is True:
        m12_nsat, obs_sats = spa.cumulative_distance(m12_sat, mask_key, coadd=False, bins=rbins)
        lg_nsat, obs_sats = spa.cumulative_distance(lg_sat, mask_key, coadd=False, bins=rbins)
        all_nsat = mf.coadd_redshift({**m12_nsat, **lg_nsat}, all_sims=all_sims)
    else:
        m12_nsat, obs_sats = spa.cumulative_distance(m12_sat, mask_key, coadd=True, bins=rbins)
        lg_nsat, obs_sats = spa.cumulative_distance(lg_sat, mask_key, coadd=True, bins=rbins)
        all_nsat = {**m12_nsat, **lg_nsat}
    
    # 3d distribution from masking in 2d
    m12_2d = sio.loop_hal(m12_sat, mask_key, spa.mask_on_2d_proj,
                                    **{'rbins':rbins, 'n_iter':n_iter, 
                                    'rlim':rlim, 'complete_limit':complete_limit})
    lg_2d = sio.loop_hal(lg_sat, mask_key, spa.mask_on_2d_proj,
                                    **{'rbins':rbins, 'n_iter':n_iter, 
                                    'rlim':rlim, 'complete_limit':complete_limit})
    all_coadd_2d = mf.coadd_redshift({**m12_2d, **lg_2d}, all_sims=all_sims)

    # M31 distribution
    if M31_data:
        m31_dict, m31_samp = obs.m31_sample(table_dir='./', 
                                M31_data_file='conn_m31_sats_posteriors.txt', 
                                sky_pos_file='mccon2012.vot',
                                n_iter=n_iter, rbins=rbins, 
                                star_mass=m12_sat.star_mass[0], M31_sm_file='m31_sats.txt')
        #rbins = np.arange(0, 310, 10)
        m31_rbins1_mask = rbins <= 150
        m31_rbins2_mask = rbins >= 150
        m31_rbins1 = np.arange(0, 160, 10)
        m31_rbins2 = np.arange(150, 310, 10)
    
    if all_sims is True:
        plot_color = CB3_1[0]
        fig, ax = plt.subplots(
            nrows=1, ncols=1, sharex=True, sharey=True, figsize=(7, 6))
        fig.set_tight_layout(False)
        ax.fill_between(rbins, all_coadd_2d['percentile'][2], 
            all_coadd_2d['percentile'][3], alpha=0.25, 
            color=plot_color, linewidth=0)
        ax.fill_between(rbins, all_coadd_2d['percentile'][0], 
            all_coadd_2d['percentile'][1], alpha=0.25, color=plot_color, 
            linewidth=0)
        ax.plot(rbins, all_coadd_2d[stat], color=plot_color, linewidth=2, 
            label='all hosts 2D-selected')
        ax.plot(rbins, all_nsat[stat], color=plot_color, alpha=0.6, linewidth=2, 
            linestyle='--', label='all hosts true 3D')
        ax.plot(m31_rbins1, m31_dict['median'][m31_rbins1_mask], color='k', 
            linestyle='--')
        ax.plot(m31_rbins2, m31_dict['median'][m31_rbins2_mask], color='k', 
            linestyle='--', alpha=0.4)
        ax.legend(loc=2, fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.set_xlim((0,np.max(rbins)))
        ax.set_ylim((0,30))
        fig.subplots_adjust(left=0.13, bottom=0.12, wspace=0.1, hspace=0.1)
        ax.set_yticks(np.arange(5,35,5), minor=False)
        ax.set_xticks(np.arange(50,350,50), minor=False)

    else:
        fig, ax = plt.subplots(nrows=3, ncols=4, sharex=True, sharey=True, 
            figsize=(12, 9))
        fig.set_tight_layout(False)

        # order hosts from most stellar mass to least
        name_iter = ['m12m', 'm12b', 'm12f', 'Thelma', 'Romeo', 'm12i', 'm12c',
                    'm12w', 'Juliet', 'Louise', 'm12z', 'm12r']
        for i, name in enumerate(name_iter):
            if 'm12' in name:
                plot_color = CB3_1[1]
            else:
                plot_color = CB3_1[2]
            j = math.floor(i/4)
            k = i - math.floor(i/4)*4
            ax[j, k].fill_between(rbins, all_coadd_2d[name]['percentile'][2], 
                all_coadd_2d[name]['percentile'][3], alpha=0.25, 
                color=plot_color, linewidth=0)
            ax[j, k].fill_between(rbins, all_coadd_2d[name]['percentile'][0], 
                all_coadd_2d[name]['percentile'][1], alpha=0.25, 
                color=plot_color, linewidth=0)
            ax[j, k].plot(rbins, all_coadd_2d[name][stat], color=plot_color, 
                linewidth=2, label=name+' 2D-selected')
            ax[j, k].plot(rbins, all_nsat[name][stat], color=plot_color, 
                alpha=0.6, linewidth=2, linestyle='--', label=name+' true 3D')
            ax[j, k].plot(m31_rbins1, m31_dict['median'][m31_rbins1_mask], 
                color='k', linestyle='--')
            ax[j, k].plot(m31_rbins2, m31_dict['median'][m31_rbins2_mask], 
                color='k', linestyle='--', alpha=0.4)
            ax[j, k].legend(loc=2, fontsize=12)
            ax[j, k].tick_params(axis='both', which='major', labelsize=16)
            ax[j, k].set_xlim((0,np.max(rbins)))
            ax[j, k].set_ylim((0,30))
            ax[j, k].set_yticks([10,20], minor=False)
            ax[j, k].set_xticks(np.arange(25,325,25), minor=True)
            ax[j, k].set_xticks([100,200], minor=False)
        fig.subplots_adjust(left=0.1, bottom=0.12, wspace=0, hspace=0)

        ax[0, 0].plot(m31_rbins1, m31_dict['median'][m31_rbins1_mask], 
            color='k', linestyle='--', label='M31')
        ax[0, 0].legend(loc=2, fontsize=12)
    fig.text(0.52, 0.02, 'Distance from host [kpc]', ha='center', fontsize=22)
    fig.text(0.02, 0.54, r'N$_{\rm sat}(<\rm{d})$', va='center', rotation='vertical', 
        fontsize=22)
    plt.show()

    return fig

### most of above is for satellite radial distribution paper

def plot_axis_ratio(sat, mask_key, title=''):
    '''
    Plot the axis ratio of all satellites from their MOI tensor axes at each
    redshift and their isotropic versions for comparison.
    '''
    axis_ratios = sio.loop_hal(sat, mask_key, spa.axis_ratio)
    iso_ratios = sio.loop_iso(sat, mask_key, iso.iso_axis_ratio)

    plt.figure()
    for i, hal_name in enumerate(axis_ratios.keys()):
        plt.plot(sat.redshift, axis_ratios[hal_name], color=CB_color_cycle[i], label=sat.hal_label[mask_key][i])
        plt.plot(sat.redshift, iso_ratios[hal_name], color='k', alpha=0.5, linestyle=line_style[i], label=sat.hal_label[mask_key][i]+' isotropic')
    plt.xlabel('z')
    plt.ylabel('Minor to major axis ratio ($c/a$)')
    plt.title(title)
    plt.legend(handlelength=2)
    plt.show()

def plot_axis_ratio2(sat_list, mask_keys, title=''):
    '''
    Plot the axis ratio of all satellites from their MOI tensor axes at each
    redshift and their isotropic versions for comparison for each sat object in
    sat_list.
    '''
    fig, ax = plt.subplots(1, len(mask_keys), sharex=True, sharey=True, figsize=(15, 5))
    for i, (sat, mask_key) in enumerate(zip(sat_list, mask_keys)):
        axis_ratios = sio.loop_hal(sat, mask_key, spa.axis_ratio)
        iso_ratios = sio.loop_iso(sat, mask_key, iso.iso_axis_ratio)
        for j, hal_name in enumerate(axis_ratios.keys()):
            ax[i].plot(sat.redshift, axis_ratios[hal_name], color=CB_color_cycle[j], label=sat.hal_label[mask_key][i])
            ax[i].plot(sat.redshift, iso_ratios[hal_name], color='k', alpha=0.5, linestyle=line_style[j], label=sat.hal_label[mask_key][i]+' isotropic')
        ax[i].legend(handlelength=2)
        ax[i].set_xlabel('Redshift (z)', fontsize=14)
    ax[0].set_ylabel('Minor to major axis ratio ($c/a$)', fontsize=14)
    plt.title(title)
    plt.show()

def plot_rms_height(sat, mask_key, title=''):
    '''
    Plot rms height and radius of all satellites in sat for each redshift.
    '''
    rms_dict = sio.loop_hal(sat, mask_key, spa.rms_distance)

    plt.figure()
    for i, hal_name in enumerate(rms_dict.keys()):
        rmsz = [rms_dict[hal_name][j]['rmsz'] for j in range(len(sat.redshift))]
        plt.plot(sat.redshift, rmsz, color=CB_color_cycle[i], label=sat.hal_label[mask_key][i])
    plt.xlabel('Redshift [z]')
    plt.ylabel('rms height of satellite disk [kpc]', fontsize=12)
    plt.legend()
    plt.title(title)	
    plt.show()

    plt.figure()
    for i, hal_name in enumerate(rms_dict.keys()):
        rmsx = [rms_dict[hal_name][j]['rmsx'] for j in range(len(sat.redshift))]
        plt.plot(sat.redshift, rmsx, color=CB_color_cycle[i], label=sat.hal_label[mask_key][i])
    plt.xlabel('Redshift [z]')
    plt.ylabel('rms radius of satellite disk [kpc]', fontsize=12)
    plt.legend()
    plt.title(title)
    plt.show()

def plot_rms_height2(sat_list, mask_keys):
    '''
    Plot rms height for sat and darksat, and repeat for rms radius.
    '''
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(15, 5))
    for i, (sat, mask_key) in enumerate(zip(sat_list, mask_keys)):
        rms_dict = sio.loop_hal(sat, mask_key, spa.rms_distance)
        for j, hal_name in enumerate(sat.hal_name):
            rmsz = [rms_dict[hal_name][j]['rmsz'] for j in range(len(sat.redshift))]
            rmsx = [rms_dict[hal_name][j]['rmsx'] for j in range(len(sat.redshift))]
            ax[0].plot(sat.redshift, rmsz, color=CB_color_cycle[j], linestyle=line_style[i], label=sat.hal_label[mask_key][i])
            ax[1].plot(sat.redshift, rmsx, color=CB_color_cycle[j], linestyle=line_style[i], label=sat.hal_label[mask_key][i])
    
    ax[0].set_xlabel('Redshift (z)')
    ax[0].set_ylabel('rms height of satellite disk [kpc]', fontsize=12)
    ax[0].legend(handlelength=2)
    ax[1].set_xlabel('Redshift (z)')
    ax[1].set_ylabel('rms radius of satellite disk [kpc]', fontsize=12)
    ax[1].legend(handlelength=2)	
    plt.show()

def plot_rms_z_r_vs_frac(sat, mask_key, title=''):
    '''
    Plot rms height vs fraction of satellites enclosed, and same thing in another
    plot for rms radius.
    '''
    frac_enclosed = sio.loop_hal(sat, mask_key, ang.fraction_open_angle, **{'angle_bins':sat.a_bins})
    rms_dict = sio.loop_hal(sat, mask_key, spa.rms_v_fraction, **{'angle_bins':sat.a_bins})

    rmsz_coadd = mf.coadd_redshift(rms_dict, 'rmsz')
    rmsx_coadd = mf.coadd_redshift(rms_dict, 'rmsx')

    plt.figure()
    for i, hal_name in enumerate(rmsz_coadd.keys()):
        plt.plot(frac_enclosed[hal_name][i], rmsz_coadd[hal_name]['mean'], label=sat.hal_label[mask_key][i])
    plt.ylabel('rms height of disk [kpc]')
    plt.xlabel('Fraction of satellites enclosed')
    plt.title(title)
    plt.legend()
    plt.show()

    plt.figure()
    for i, hal_name in enumerate(rmsx_coadd.keys()):
        plt.plot(frac_enclosed[hal_name][i], rmsx_coadd[hal_name]['mean'], label=sat.hal_label[mask_key][i])
    plt.ylabel('rms radius of disk [kpc]')
    plt.xlabel('Fraction of satellites enclosed')
    plt.title(title)
    plt.legend()
    plt.show()

# needs work
def plot_coadd_rms_z_r_vs_frac(sat, mask_key, title=''):
    '''
    Plot coadded rms height and radius of sat vs fraction enclosed.
    '''
    frac_enclosed = sio.loop_hal(sat, mask_key, ang.fraction_open_angle, **{'angle_bins':sat.a_bins})
    frac_coadd = mf.coadd_redshift(frac_enclosed)
    rms_dict = sio.loop_hal(sat, mask_key, spa.rms_v_fraction, **{'angle_bins':sat.a_bins})
    rmsz_coadd = mf.coadd_redshift(rms_dict, 'rmsz')
    rmsx_coadd = mf.coadd_redshift(rms_dict, 'rmsx')

    pg.plot_coadd_comparison(frac_coadd, rmsz_coadd, isotropic_y_data=None, host_halo_names=sat.hal_label[mask_key],
        plot_title=title, xlabel='Average fraction of satellites enclosed', ylabel='Average rms height',
        color_list=CB_color_cycle, xscale='linear', location=None)
    
    pg.plot_coadd_comparison(frac_coadd, rmsx_coadd, isotropic_y_data=None, host_halo_names=sat.hal_label[mask_key],
        plot_title=title, xlabel='Average fraction of satellites enclosed', ylabel='Average rms radius',
        color_list=CB_color_cycle, xscale='linear', location=None)

def plot_coadd_rms_z_r_vs_frac2(sat_list, mask_keys, title=''):
    '''
    Plot coadded rms height and radius vs fraction enclosed for each sat in sat_list.
    '''
    fig_h, ax_h = plt.subplots(len(mask_keys), 3, sharex=True, sharey=True)
    fig_r, ax_r = plt.subplots(len(mask_keys), 3, sharex=True, sharey=True)

    for i, (sat, mask_key) in enumerate(zip(sat_list, mask_keys)):
        frac_enclosed = sio.loop_hal(sat, mask_key, ang.fraction_open_angle, **{'angle_bins':sat.a_bins})
        frac_coadd = mf.coadd_redshift(frac_enclosed)
        rms_dict = sio.loop_hal(sat, mask_key, spa.rms_v_fraction, **{'angle_bins':sat.a_bins})

        rmsz = mf.coadd_redshift(rms_dict, 'rmsz')
        rmsx = mf.coadd_redshift(rms_dict, 'rmsx')

        for j, name in enumerate(sat.hal_name):
            ax_h[i, j].fill_between(frac_coadd[name]['mean'], rmsz[name]['percentile'][2], rmsz[name]['percentile'][3], alpha=0.3,
                color=CB_color_cycle[j], linewidth=0)
            ax_h[i, j].fill_between(frac_coadd[name]['mean'], rmsz[name]['percentile'][0], rmsz[name]['percentile'][1], alpha=0.5,
                color=CB_color_cycle[j], linewidth=0)
            ax_h[i, j].plot(frac_coadd[name]['mean'], rmsz[name]['mean'], color=CB_color_cycle[j], label=sat.hal_label[mask_key][j])
            ax_h[i, j].legend()

        for j, name in enumerate(sat.hal_name):
            ax_r[i, j].fill_between(frac_coadd[name]['mean'], rmsx[name]['percentile'][2], rmsx[name]['percentile'][3], alpha=0.3,
                color=CB_color_cycle[j], linewidth=0)
            ax_r[i, j].fill_between(frac_coadd[name]['mean'], rmsx[name]['percentile'][0], rmsx[name]['percentile'][1], alpha=0.5,
                color=CB_color_cycle[j], linewidth=0)
            ax_r[i, j].plot(frac_coadd[name]['mean'], rmsx[name]['mean'], color=CB_color_cycle[j], label=sat.hal_label[mask_key][j])
            ax_r[i, j].legend()

    ax_h[0, 0].set_ylabel('rms height [kpc]', fontsize=12)
    ax_h[1, 1].set_xlabel('Fraction of satellites enclosed', fontsize=12)
    ax_r[0, 0].set_ylabel('rms radius [kpc]', fontsize=12)
    ax_r[1, 1].set_xlabel('Fraction of satellites enclosed', fontsize=12)
    plt.show()

def plot_rms_vs_r_frac(sat_list, mask_key_list, radius_frac=0.68, rfrac1=0.9, rfrac2=0.1, ax_key='moi'):
    '''
    Plot rms height vs. concentration for the given enclosing fraction. Points are
    color-coded by redshift and the correlation coefficient is included in the
    legend.
    get r%, then get rms height of satellites within r%
    '''
    ps_arr = np.array((['^', 's', 'p'],['+', 'x', '3'],['s','p','h']))
    cm = plt.cm.get_cmap('viridis')
    fig = plt.figure(figsize=(14,6))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1,1,0.05])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1, sharey=ax1)
    ax3 = fig.add_subplot(gs[2])
    ax = [ax1, ax2, ax3]

    for i, (sat, mask_key) in enumerate(zip(sat_list, mask_key_list)):
        r_ratio = sio.loop_hal(sat, mask_key, spa.rfrac_vs_rfrac, **{'frac1':rfrac1, 'frac2':rfrac2, 'r_bins':sat.r_bins})
     
        if ax_key == 'moi':
            rms_dict = sio.loop_hal(sat, mask_key, spa.rms_vs_r_frac, **{'r_frac':radius_frac,'radius_bins':sat.r_bins})
            rmsz = sio.single_property_dict(rms_dict, 'rmsz')
            # experimental section
            #r_ratio = ba.single_property_dict(rms_dict, 'rmsx')
            #r_ratio = ba.loop_hal(sat, mask_key, spa.r_fraction, **{'frac':radius_frac, 'radius_bins':sat.r_bins})

            iso_rms_dict = sio.loop_iso(sat, mask_key, iso.iso_rms_vs_r_frac, **{'r_frac':radius_frac,'radius_bins':sat.r_bins})
            iso_rmsz = sio.single_property_dict(iso_rms_dict, 'rmsz')
        elif ax_key == 'rand':
            rms_dict = sio.loop_hal(sat, mask_key, ra.rand_rms_min, **{'r_frac':radius_frac, 'radius_bins':sat.r_bins, 'n_iter':sat.n_iter})
            rmsz = sio.single_property_dict(rms_dict, 'rmsz')

            iso_rms_dict = sio.loop_iso(sat, mask_key, iso.rand_iso_rms_min, **{'r_frac':radius_frac, 'r_bins':sat.r_bins, 'n_iter':sat.n_iter})
            iso_rmsz = sio.single_property_dict(iso_rms_dict, 'rmsz')

        for j, name in enumerate(sat.hal_name):
            label_str = r'{}: $R^2$ = {:.3f}'.format(sat.hal_label[mask_key][j], np.corrcoef(r_ratio[name], rmsz[name])[0,1]**2)
            pts = ax[0].scatter(r_ratio[name], rmsz[name], c=sat.redshift, marker=ps_arr[i, j], s=100, label=label_str, cmap=cm)
            ax[0].legend()

            iso_label_str = r'{}: $R^2$ = {:.3f}'.format(sat.hal_label[mask_key][j], np.corrcoef(r_ratio[name], iso_rmsz[name])[0,1]**2)
            pts = ax[1].scatter(r_ratio[name], iso_rmsz[name], c=sat.redshift, marker=ps_arr[i, j], s=100, label=iso_label_str, cmap=cm)
            ax[1].legend()

    ax[2].tick_params(axis=u'both', which=u'both',length=0)
    cbar = fig.colorbar(pts, cax=ax[2], orientation='vertical')
    cbar.set_label('Redshift [z]')

    rms_frac_str = int(radius_frac*100)
    rfrac1_str = int(rfrac1*100)
    rfrac2_str = int(rfrac2*100)
    ax[0].set_xlabel('$R{}$/$R{}$'.format(rfrac1_str, rfrac2_str), fontsize=18)
    ax[1].set_xlabel('$R{}$/$R{}$'.format(rfrac1_str, rfrac2_str), fontsize=18)
    ax[0].set_ylabel('rms height of {}$\%$ of sats [kpc]'.format(rms_frac_str), fontsize=18)
    ax[1].set_title('Isotropic $N_{iter}=$'+str(sat_list[0].n_iter), fontsize=18)
    ax[0].set_title(' ', fontsize=12)
    plt.show()

    return fig

def plot_rms_vs_r_frac2(sat_list, mask_key_list, radius_frac=0.68, rfrac1=0.9, rfrac2=0.1, ax_key='moi'):
    '''
    Plot rms height vs. concentration for the given enclosing fraction,
    and the correlation coefficient is included in the legend.
    (get r%, then get rms height of satellites within r%)
    '''
    ps_arr = np.array(['^', 's', 'p', 'h', '+', 'x', '3', 's','p','h'])
    colors = np.array([CB_color_cycle, ['darkgray','darkgray','darkgray']])
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(12,6))

    for i, (sat, mask_key) in enumerate(zip(sat_list, mask_key_list)):
        r_ratio = sio.loop_hal(sat, mask_key, spa.rfrac_vs_rfrac, **{'frac1':rfrac1, 'frac2':rfrac2, 'r_bins':sat.r_bins})
     
        if ax_key == 'moi':
            rms_dict = sio.loop_hal(sat, mask_key, spa.rms_vs_r_frac, **{'r_frac':radius_frac,'radius_bins':sat.r_bins})
            rmsz = sio.single_property_dict(rms_dict, 'rmsz')
            # experimental section
            #r_ratio = ba.single_property_dict(rms_dict, 'rmsx')
            #r_ratio = ba.loop_hal(sat, mask_key, spa.r_fraction, **{'frac':radius_frac, 'radius_bins':sat.r_bins})

            #iso_rms_dict = sio.loop_iso(sat, mask_key, iso.iso_rms_vs_r_frac, **{'r_frac':radius_frac,'radius_bins':sat.r_bins})
            #iso_rmsz = sio.single_property_dict(iso_rms_dict, 'rmsz')

        elif ax_key == 'rand':
            rms_dict = sio.loop_hal(sat, mask_key, ra.rand_rms_min, **{'r_frac':radius_frac, 'radius_bins':sat.r_bins, 'n_iter':sat.n_iter})
            rmsz = sio.single_property_dict(rms_dict, 'rmsz')

            #iso_rms_dict = ba.loop_iso(sat, mask_key, iso.rand_iso_rms_min, **{'r_frac':radius_frac, 'r_bins':sat.r_bins, 'n_iter':sat.n_iter})
            #iso_rmsz = ba.single_property_dict(iso_rms_dict, 'rmsz')

        for j, name in enumerate(sat.hal_name):
            label_str = r'{}: $r$ = {:.3f}'.format(sat.hal_label[mask_key][j], np.corrcoef(r_ratio[name], rmsz[name])[0,1])
            ax[0].plot(r_ratio[name], rmsz[name], color=colors[i][j], alpha=0.8, marker=ps_arr[j], markeredgecolor='k', markeredgewidth=1, markersize=10, linestyle='None', label=label_str)
            ax[0].legend(fontsize=16)
            print(name, np.mean(rmsz[name]))

            #iso_label_str = r'{}: $r$ = {:.3f}'.format(sat.hal_label[mask_key][j], np.corrcoef(r_ratio[name], iso_rmsz[name])[0,1])
            #ax[1].plot(r_ratio[name], iso_rmsz[name], color=colors[i][j], alpha=0.8, marker=ps_arr[j], markeredgecolor='k', markeredgewidth=1, markersize=10, linestyle='None', label=iso_label_str)
            #ax[1].legend(fontsize=16)

    rfrac1_str = int(rfrac1*100)
    rfrac2_str = int(rfrac2*100)
    ax[0].set_xlabel(r'$R{}$/$R{}$'.format(rfrac1_str, rfrac2_str), fontsize=18)
    ax[1].set_xlabel(r'$R{}$/$R{}$'.format(rfrac1_str, rfrac2_str), fontsize=18)
    ax[0].set_ylabel(r'rms height of plane [kpc]', fontsize=18)
    plt.show()

    return fig

def plot_axratio_vs_r_frac2(sat_list, mask_key_list, rfrac1=0.9, rfrac2=0.1, ax_key='moi'):
    '''
    Plot rms height vs. concentration for the given enclosing fraction,
    and the correlation coefficient is included in the legend.
    (get r%, then get rms height of satellites within r%)
    '''
    ps_arr = np.array(['^', 's', 'p', 'h', '+', 'x', '3', 's','p','h'])
    colors = np.array([CB_color_cycle, ['darkgray','darkgray','darkgray']])
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(12,6))

    for i, (sat, mask_key) in enumerate(zip(sat_list, mask_key_list)):
        r_ratio = sio.loop_hal(sat, mask_key, spa.rfrac_vs_rfrac, **{'frac1':rfrac1, 'frac2':rfrac2, 'r_bins':sat.r_bins})
     
        if ax_key == 'moi':
            axis_ratio = sio.loop_hal(sat, mask_key, spa.axis_ratio)
            #iso_axis_ratio = sio.loop_iso(sat, mask_key, iso.iso_axis_ratio)
        '''
        elif ax_key == 'rand':
            rms_dict = ba.loop_hal(sat, mask_key, ra.rand_rms_min, **{'r_frac':radius_frac, 'radius_bins':sat.r_bins, 'n_iter':500})
            rmsz = ba.single_property_dict(rms_dict, 'rmsz')

            iso_rms_dict = ba.loop_iso(sat, mask_key, iso.rand_iso_rms_min, **{'r_frac':radius_frac, 'r_bins':sat.r_bins, 'n_iter':500})
            iso_rmsz = ba.single_property_dict(iso_rms_dict, 'rmsz')
        '''
        for j, name in enumerate(sat.hal_name):
            label_str = r'{}: $r$ = {:.3f}'.format(sat.hal_label[mask_key][j], np.corrcoef(r_ratio[name], axis_ratio[name])[0,1])
            ax[0].plot(r_ratio[name], axis_ratio[name], color=colors[i][j], alpha=0.8, marker=ps_arr[j], markeredgecolor='k', markeredgewidth=1, markersize=10, linestyle='None', label=label_str)
            ax[0].legend(fontsize=16)
            print(name, np.mean(axis_ratio[name]))

            #iso_label_str = r'{}: $r$ = {:.3f}'.format(sat.hal_label[mask_key][j], np.corrcoef(r_ratio[name], iso_axis_ratio[name])[0,1])
            #ax[1].plot(r_ratio[name], iso_axis_ratio[name], color=colors[i][j], alpha=0.8, marker=ps_arr[j], markeredgecolor='k', markeredgewidth=1, markersize=10, linestyle='None', label=iso_label_str)
            #ax[1].legend(fontsize=16)

    rfrac1_str = int(rfrac1*100)
    rfrac2_str = int(rfrac2*100)
    ax[0].set_xlabel(r'$R{}$/$R{}$'.format(rfrac1_str, rfrac2_str), fontsize=18)
    ax[1].set_xlabel(r'$R{}$/$R{}$'.format(rfrac1_str, rfrac2_str), fontsize=18)
    ax[0].set_ylabel(r'axis ratio of plane (c/a)', fontsize=18)
    #ax[1].set_title(r'Isotropic $N_{iter}=$'+str(sat_list[0].n_iter), fontsize=14)
    #ax[0].set_title(' ', fontsize=12)
    plt.show()

    return fig

###############################
def plot_concentration_corr(sat_list, mask_keys, sat_frac=1.0, rfrac1=0.9, rfrac2=0.1, ax_key='moi', metric='angle'):
    '''
    Generate a two panel comparison plot of the angle enclosing angle_frac fraction
    of the satellites for a given measure of concentration, r1/r2, where r1 and
    r2 are input as fractions and calculated as radii enclosing their
    respective fractions of the satellite distribution.
    '''
    ps_arr = np.array(['^', 's', 'p', 'h', '+', 'x', '3', 's','p','h'])
    colors = np.array([CB_color_cycle, ['darkgray','darkgray','darkgray']])
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(7,7))

    for i, (sat, mask_key) in enumerate(zip(sat_list, mask_keys)):
        r_ratio = sio.loop_hal(sat, mask_key, spa.rfrac_vs_rfrac, **{'frac1':rfrac1, 'frac2':rfrac2, 'r_bins':sat.r_bins})
        # experimental section
        #rms_dict = ba.loop_hal(sat, mask_key, spa.rms_vs_r_frac, **{'r_frac':angle_frac,'radius_bins':sat.r_bins})
        #r_ratio = ba.single_property_dict(rms_dict, 'rmsx')
        #r_ratio = ba.loop_hal(sat, mask_key, spa.r_fraction, **{'frac':0.5, 'radius_bins':sat.r_bins})
        
        if ax_key == 'moi':
            if metric == 'angle':
                iso_metric = sio.loop_iso(sat, mask_key, iso.iso_angle_width, **{'threshold_fraction':sat_frac, 'angle_bins':sat.a_bins})
            elif metric == 'rms':
                iso_metric_dict = sio.loop_iso(sat, mask_key, iso.iso_rms_vs_r_frac, **{'r_frac':sat_frac,'radius_bins':sat.r_bins})
                iso_metric = sio.single_property_dict(iso_metric_dict, 'rmsz')
            elif metric == 'axis.ratio':
                iso_metric = sio.loop_iso(sat, mask_key, iso.iso_axis_ratio)

        elif ax_key == 'rand':
            if metric == 'angle':
                iso_metric_dict = sio.loop_iso(sat, mask_key, iso.iso_rand_angle_width, **{'threshold_fraction':sat_frac, 'angle_range':sat.a_bins, 'n_iter':sat.n_iter})
                iso_metric = sio.single_property_dict(iso_metric_dict, 'angle')
            elif metric == 'rms':
                iso_metric_dict = sio.loop_iso(sat, mask_key, iso.rand_iso_rms_min, **{'r_frac':sat_frac, 'r_bins':sat.r_bins, 'n_iter':sat.n_iter})
                iso_metric = sio.single_property_dict(iso_metric_dict, 'rmsz')

        for j, hal_name in enumerate(iso_metric.keys()):
            iso_label_str = r'{}: $r$ = {:.3f}'.format(sat.hal_label[mask_key][j], np.corrcoef(r_ratio[hal_name], iso_metric[hal_name])[0,1])
            ax.plot(r_ratio[hal_name], iso_metric[hal_name], color=colors[i][j], alpha=0.8, marker=ps_arr[j], markeredgecolor='k', markeredgewidth=1, markersize=10, linestyle='None', label=iso_label_str)
            ax.legend(fontsize=16)

    rfrac1_str = int(rfrac1*100)
    rfrac2_str = int(rfrac2*100)
    ax.set_xlabel('$R{}$/$R{}$'.format(rfrac1_str, rfrac2_str), fontsize=18)
    metric_label_dict = {'angle':'Enclosing angle of plane [deg]', 'rms':'rms height of plane [kpc]', 'axis.ratio':'axis ratio of plane (c/a)'}
    ax.set_ylabel(metric_label_dict[metric], fontsize=18)

    plt.show()

    return fig

# I don't think this works anymore
def plot_coadd_rmsz_vs_r(sat_list, mask_keys):
    '''
    Plot coadded rms height vs enclosing radius for each sat in sat_list.
    Correlation coefficient is included in the legend.
    '''
    fig, ax = plt.subplots(len(sat_list[0].hal_name), len(sat_list), sharex=True, sharey=True, figsize=(10,15))

    for i, (sat, mask_key) in enumerate(zip(sat_list, mask_keys)):
        rms_z_radii_dict, coadd_rms_z = spa.rmsz_vs_r(sat, isotropic=False)
        iso_rms_z_radii_dict, iso_coadd_rms_z = spa.rmsz_vs_r(sat, isotropic=True)

        for j, name in enumerate(sat.hal_name):
            ax[j, i].fill_between(rms_z_radii_dict[name]['radii'], coadd_rms_z[name]['percentile'][2], coadd_rms_z[name]['percentile'][3], alpha=0.3,
                color=CB_color_cycle[j], linewidth=0)
            ax[j, i].fill_between(rms_z_radii_dict[name]['radii'], coadd_rms_z[name]['percentile'][0], coadd_rms_z[name]['percentile'][1], alpha=0.5,
                color=CB_color_cycle[j], linewidth=0)
            label_str = r'{}: $R^2$ = {:.3f}'.format(sat.hal_label[mask_key][j], np.corrcoef(rms_z_radii_dict[name]['radii'], coadd_rms_z[name]['mean'])[0,1]**2)
            ax[j, i].plot(rms_z_radii_dict[name]['radii'], coadd_rms_z[name]['mean'], color=CB_color_cycle[j], label=label_str)
            iso_label_str = r'isotropic: $R^2$ = {:.3f}'.format(np.corrcoef(iso_rms_z_radii_dict[name]['radii'], iso_coadd_rms_z[name]['mean'])[0,1]**2)
            ax[j, i].plot(iso_rms_z_radii_dict[name]['radii'], iso_coadd_rms_z[name]['mean'], alpha=0.8, color='k', label=iso_label_str)
            ax[j, i].legend()

    ax[1, 0].set_ylabel('rms height [kpc]', fontsize=12)
    ax[2, 0].set_xlabel('Enclosing radius [kpc]', fontsize=12)
    ax[2, 1].set_xlabel('Enclosing radius [kpc]', fontsize=12)
    plt.show()

    return fig
