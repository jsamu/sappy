import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from satellite_analysis import satellite_io as sio
from satellite_analysis import math_funcs as mf
from satellite_analysis import spatial as spa


CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

line_style = ['-', '--', '-.', ':']


def radial_distribution(sat, mask_key, redshift_index=0, norm=False, bins=None):
    if bins is None:
        bins = sat.r_bins

    radial_dist = spa.cumulative_distance(sat, mask_key, norm=norm, bins=bins)
    sim_dist = radial_dist[0]
    MW_dist = radial_dist[1]
    M31_dist = radial_dist[2]

    plt.figure(figsize=(6,6))

    for i, hal_name in enumerate(sat.hal_name):
        plt.plot(bins, sim_dist[hal_name][redshift_index], color=CB_color_cycle[i], label=sat.hal_label[mask_key][i])
    plt.plot(bins, MW_dist, color='k', label='Milky Way', linestyle='-')
    plt.plot(bins, M31_dist, color='k', label='M31', linestyle='--')

    plt.legend(handlelength=2, fontsize=14)
    plt.xlabel('Distance from host galaxy [kpc]', fontsize=18)
    plt.ylabel('$N_{sat}(<d)$', fontsize=18)
    plt.show()
"""
def stellar_mass_dist(sat, mask_key, redshift_index=0, norm=False, bins=None):
    if bins is None:
        bins = np.logspace(5, 9, 15)

    stellar_mass = stellar_mass_func(sat, mask_key, norm=norm, bins=bins)
    sim_sm = stellar_mass[0]
    MW_sm = stellar_mass[1]
    M31_sm = stellar_mass[2]

    plt.figure(figsize=(6,6))

    for i, hal_name in enumerate(sat.hal_name):
        plt.plot(bins, sim_sm[hal_name][redshift_index], color=CB_color_cycle[i], label=sat.hal_label[mask_key][i])
    plt.plot(bins, MW_sm, color='k', label='Milky Way', linestyle='-')
    plt.plot(bins, M31_sm, color='k', label='M31', linestyle='--')

    plt.xscale('log')
    plt.legend(handlelength=2, fontsize=14)
    plt.xlabel('Stellar mass [$M_{\odot}$]', fontsize=18)
    plt.ylabel('$N_{sat}(>M_{star})$', fontsize=18)
    plt.show()

def stellar_mass_func(sat, mask_key, norm=False, bins=None):
    '''
    Calculate the stellar mass distribution of sim subhalos at z = 0, and for
    observed satellites today.
    '''
    if bins is None:
        bins = np.logspace(4.5, 9.5, 10)

    sim_sm = sio.loop_hal(sat, mask_key, mf.cumulative_prop, **{'hal_property':'star.mass', 'bins':bins, 'above':True, 'normalized':norm})

    #mask observational data
    MW_sm = sat.observation.prop('star.mass')[sat.MW_mask]
    M31_sm = sat.observation.prop('star.mass')[sat.M31_mask]

    #cumulative observational quantities
    MW_sm_cum = mf.obs_cumulative_prop(MW_sm, bins, above=True, normalized=norm)
    M31_sm_cum = mf.obs_cumulative_prop(M31_sm, bins, above=True, normalized=norm)

    sm_list = [sim_sm, MW_sm_cum, M31_sm_cum]

    return sm_list
"""