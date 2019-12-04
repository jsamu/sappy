import numpy as np
from scipy import stats
from numba import jit
import utilities as ut
from satellite_analysis import satellite_io as sio
from satellite_analysis import isotropic as iso
from satellite_analysis import angular as ang
from satellite_analysis import math_funcs as mf
from satellite_analysis import observation as obs
from satellite_analysis import rand_axes as ra

def ks_test_distance(
    hal, hal_mask=None, host_str='host.', sig_lvl=95, n_samp=1000, 
    stat='median', star_mass_lim=1e5, table_dir='./'):
    mw_dict, mw_n = obs.mw_sample(n_iter=n_samp, star_mass=star_mass_lim, no_bin=True, table_dir=table_dir)
    m31_dict, m31_n = obs.m31_sample(n_iter=n_samp, star_mass=star_mass_lim, no_bin=True, table_dir=table_dir)
    mw_ = mw_dict[stat]
    m31_ = m31_dict[stat]
    
    sat_rad = hal.prop(host_str+'distance.total')[hal_mask]
    ks_mw = stats.ks_2samp(mw_, sat_rad)
    ks_m31 = stats.ks_2samp(m31_, sat_rad)
    
    c_alpha = {'99':1.62762, '95':1.35810, '90':1.22385, '85':1.13795, '80':1.07275}
    stat_crit_mw = c_alpha[str(sig_lvl)]*np.sqrt((len(mw_)+len(sat_rad))/((len(mw_)*len(sat_rad))))
    stat_crit_m31 = c_alpha[str(sig_lvl)]*np.sqrt((len(m31_)+len(sat_rad))/((len(m31_)*len(sat_rad))))
            
    return {'MW':[ks_mw[0], ks_mw[1], stat_crit_mw], 'M31':[ks_m31[0], ks_m31[1], stat_crit_m31]}

def ad_test_distance(
    hal, hal_mask=None, host_str='host.', n_samp=1000, stat='median', 
    star_mass_lim=1e5, table_dir='./'):
    mw_dict, mw_n = obs.mw_sample(n_iter=n_samp, star_mass=star_mass_lim, no_bin=True, table_dir=table_dir)
    m31_dict, m31_n = obs.m31_sample(n_iter=n_samp, star_mass=star_mass_lim, no_bin=True, table_dir=table_dir)
    mw_ = mw_dict[stat]
    m31_ = m31_dict[stat]
    
    sat_rad = hal.prop(host_str+'distance.total')[hal_mask]
    ad_mw = stats.anderson_ksamp([mw_, sat_rad])
    ad_m31 = stats.anderson_ksamp([m31_, sat_rad])
            
    return {'MW':[ad_mw[0], ad_mw[1], ad_mw[2]], 'M31':[ad_m31[0], ad_m31[1], ad_m31[2]]}

@jit
def get_satellite_principal_axes(hal, hal_mask=None, host_str='host.', mass_kind=None):
    '''
    Find the inertia tensor and maj/med/min axes of the satellite distribution
    (hal) that meets the criteria of (hal_mask).
    '''
    distance_vectors = hal.prop(host_str+'distance')[hal_mask]

    # ut.coordinate.get_principal_axes returns: eigen_vectors, eigen_values, axis_ratios
    if mass_kind == None:
        # use this if you only care about geometric distribution
        moi_quantities = ut.coordinate.get_principal_axes(distance_vectors)

    else:
        # use this if you want to weight positions by mass or another property
        mass = hal.prop(mass_kind)[hal_mask]
        moi_quantities = ut.coordinate.get_principal_axes(distance_vectors, mass)

    return moi_quantities

def num_v_radius_bin(hal, hal_mask=None, host_str='host.', bins=None):
    # differential satellite counts vs distance
    radial_distance = hal.prop(host_str+'distance.total')[hal_mask]
    
    dn_dr = []
    for i in range(len(bins)):
        if i == 0:
            dn = (radial_distance <= bins[i]) & (radial_distance > 0)
        else:
            dn = (radial_distance <= bins[i]) & (radial_distance > bins[i-1])
        dn_dr.append(np.sum(dn))

    return dn_dr

def coadd_num_v_radius_bin_ratio(
    dmo_m12, m12_sat, dmo_lg=None, lg_sat=None, mask_key=None, bins=None,
    diff=False, norm=False, all_sims=False):
    """
    DMO and baryon versions must have the simulations listed in the same order.
    """
    if bins is None:
        bins = dmo_m12.r_bins

    if diff:
        dmo_m12_dist = sio.loop_hal(dmo_m12, mask_key, num_v_radius_bin, **{'bins':bins})
        m12_dist = sio.loop_hal(m12_sat, mask_key, num_v_radius_bin, **{'bins':bins})
        if not dmo_lg:
            dmo_lg_dist = {}
            lg_dist = {}
        else:
            dmo_lg_dist = sio.loop_hal(dmo_lg, mask_key, num_v_radius_bin, **{'bins':bins})
            lg_dist = sio.loop_hal(lg_sat, mask_key, num_v_radius_bin, **{'bins':bins})
        
    else:
        dmo_m12_dist = sio.loop_hal(dmo_m12, mask_key, mf.cumulative_prop,
                                **{'hal_property':'distance.total',
                                'bins':bins, 'above':False, 'normalized':norm})
        m12_dist = sio.loop_hal(m12_sat, mask_key, mf.cumulative_prop,
                                **{'hal_property':'distance.total',
                                'bins':bins, 'above':False, 'normalized':norm})
        if not dmo_lg:
            dmo_lg_dist = {}
            lg_dist = {}
        else:  
            dmo_lg_dist = sio.loop_hal(dmo_lg, mask_key, mf.cumulative_prop,
                                **{'hal_property':'distance.total',
                                'bins':bins, 'above':False, 'normalized':norm})   
            lg_dist = sio.loop_hal(lg_sat, mask_key, mf.cumulative_prop,
                                **{'hal_property':'distance.total',
                                'bins':bins, 'above':False, 'normalized':norm})

    dmo_dist = {**dmo_m12_dist, **dmo_lg_dist}
    baryon_dist = {**m12_dist, **lg_dist}

    # time average distributions for each host
    dmo_avg_host = {}
    baryon_avg_host = {}
    for dmo_name, baryon_name in zip(dmo_dist.keys(), baryon_dist.keys()):
        dmo_avg_host[dmo_name] = np.nanmean(dmo_dist[dmo_name], axis=0)
        baryon_avg_host[baryon_name] = np.nanmean(baryon_dist[baryon_name], axis=0)

    # take ratio of time averaged baryon to dmo for each host
    ratio = []
    for dmo_name, baryon_name in zip(dmo_avg_host.keys(), baryon_avg_host.keys()):
        pre_ratio = np.array(baryon_avg_host[baryon_name]/dmo_avg_host[dmo_name])
        pre_ratio[np.isinf(pre_ratio)] = np.nan
        ratio.append(pre_ratio)

    # coadd over all hosts
    dist_dict = {}
    dist_dict['mean'] = np.nanmean(ratio, axis=0)
    dist_dict['median'] = np.nanmedian(ratio, axis=0)
    dist_dict['percentile'] = np.nanpercentile(ratio, [16, 84, 2.5, 97.5], axis=0)

    if all_sims:
        return dist_dict
    else:
        return ratio

def total_sats(darksat, baryon_sat, mask_key, radius):
    '''
    Find the total number of subhalos (which meet the criteria of catalog_mask[mask_key])
    that are inside the set radius of the host at z=0.
    Used for creation of darksat masks in halo_reader
    '''
    cumul_sats = sio.loop_hal(baryon_sat, mask_key, mf.cumulative_prop, **{'hal_property':'distance.total', 'bins':[radius]})
    total_sats = {}

    for d_host, b_host in zip(sio.hal_name_iter(darksat), cumul_sats.keys()):
        total_sats[d_host] = [snap[0] for snap in cumul_sats[b_host]]

    return total_sats

def coadd_distance_all_hosts(sat, mask_key, norm=False, bins=None):
    '''
    Coadd cumulative distributions of distance from host galaxy, over all hosts
    and over all redshifts.
    '''
    if bins is None:
        bins = sat.r_bins
    if sat.sat_type in ['hal', 'tree']:
        dist_cum = sio.loop_hal(sat, mask_key, mf.cumulative_prop,
                                **{'hal_property':'distance.total',
                                'bins':bins, 'above':False, 'normalized':norm})
    elif sat.sat_type == 'tree.lg':
        dist_cum = sio.loop_hal(sat, mask_key, mf.cumulative_prop_lg,
                                **{'hal_property':'distance', 'bins':bins,
                                'above':False, 'normalized':norm})

    obs_dict = obs.obs_cumulative_prop(sat.observation, sat.observation_mask,
                                       'distance.host', bins, normalized=norm)

    all_dist = []
    for name in dist_cum.keys():
        for j in range(len(sat.redshift)):
            all_dist.append(dist_cum[name][j])

    dist_dict = {}
    dist_dict['mean'] = np.average(all_dist, axis=0)
    dist_dict['median'] = np.median(all_dist, axis=0)
    dist_dict['percentile'] = np.nanpercentile(all_dist, [16, 84, 2.5, 97.5], axis=0)

    return dist_dict, obs_dict

def coadd_distance_m12_and_lg(
    sat, sat_lg, mask_key, norm=False, bins=None, diff=False):
    '''
    Coadd cumulative or differential distributions of number of satellites vs 
    distance from host galaxy, over all hosts and over all redshifts.
    '''
    if bins is None:
        bins = sat.r_bins

    if diff is True:
        m12_radial_dist = sio.loop_hal(sat, mask_key, num_v_radius_bin, **{'bins':bins})
        lg_radial_dist = sio.loop_hal(sat_lg, mask_key, num_v_radius_bin, **{'bins':bins})
    else:
        m12_radial_dist = sio.loop_hal(sat, mask_key, mf.cumulative_prop,
                                    **{'hal_property':'distance.total',
                                    'bins':bins, 'above':False, 'normalized':norm})
        lg_radial_dist = sio.loop_hal(sat_lg, mask_key, mf.cumulative_prop,
                                    **{'hal_property':'distance.total', 'bins':bins,
                                    'above':False, 'normalized':norm})

    #obs_dict = obs.obs_cumulative_prop(sat.observation, sat.observation_mask,
    #                                   'distance.host', bins, normalized=norm)

    all_dist = []
    m12_dist = []
    lg_dist = []
    for name in m12_radial_dist.keys():
        for j in range(len(sat.redshift)):
            all_dist.append(m12_radial_dist[name][j])
            m12_dist.append(m12_radial_dist[name][j])
    for name in lg_radial_dist.keys():
        for j in range(len(sat.redshift)):
            all_dist.append(lg_radial_dist[name][j])
            lg_dist.append(lg_radial_dist[name][j])

    # sort into separate dictionaries and a common dictionary for statistics
    m12_dist_dict = {}
    m12_dist_dict['mean'] = np.nanmean(m12_dist, axis=0)
    m12_dist_dict['median'] = np.nanmedian(m12_dist, axis=0)
    m12_dist_dict['percentile'] = np.nanpercentile(m12_dist, [16, 84, 2.5, 97.5], axis=0)

    lg_dist_dict = {}
    lg_dist_dict['mean'] = np.nanmean(lg_dist, axis=0)
    lg_dist_dict['median'] = np.nanmedian(lg_dist, axis=0)
    lg_dist_dict['percentile'] = np.nanpercentile(lg_dist, [16, 84, 2.5, 97.5], axis=0)

    all_dist_dict = {}
    all_dist_dict['mean'] = np.nanmean(all_dist, axis=0)
    all_dist_dict['median'] = np.nanmedian(all_dist, axis=0)
    all_dist_dict['percentile'] = np.nanpercentile(all_dist, [16, 84, 2.5, 97.5], axis=0)

    return all_dist_dict, m12_dist_dict, lg_dist_dict#, obs_dict

def coadd_distance_ratio(
    sat, sat_lg, mask_key, norm=False, bins=None, mean_first=False, 
    obs_samp=None, diff=False):
    if bins is None:
        bins = sat.r_bins

    if diff is True:
        m12_radial_dist = sio.loop_hal(sat, mask_key, num_v_radius_bin, **{'bins':bins})
        lg_radial_dist = sio.loop_hal(sat_lg, mask_key, num_v_radius_bin, **{'bins':bins})
    else:
        m12_radial_dist = sio.loop_hal(sat, mask_key, mf.cumulative_prop,
                                    **{'hal_property':'distance.total',
                                    'bins':bins, 'above':False, 'normalized':norm})
        lg_radial_dist = sio.loop_hal(sat_lg, mask_key, mf.cumulative_prop,
                                    **{'hal_property':'distance.total', 'bins':bins,
                                    'above':False, 'normalized':norm})

    if not mean_first:
        obs_dict = obs.obs_cumulative_prop(sat.observation, sat.observation_mask,
                                       'distance.host', bins, normalized=norm)
        all_dist = []
        for name in m12_radial_dist.keys():
            for j in range(len(sat.redshift)):
                all_dist.append(m12_radial_dist[name][j])
        for name in lg_radial_dist.keys():
            for j in range(len(sat.redshift)):
                all_dist.append(lg_radial_dist[name][j])

        all_ratios = {'MW':[np.array(dist)/np.array(obs_dict['MW']) for dist in all_dist],
                      'M31':[np.array(dist)/np.array(obs_dict['M31']) for dist in all_dist]}

        all_ratio_dict = {}
        for obs_sys in ['MW', 'M31']:
            ratio_dist_dict = {}
            ratio_dist_dict['mean'] = np.nanmean(all_ratios[obs_sys], axis=0)
            ratio_dist_dict['median'] = np.nanmedian(all_ratios[obs_sys], axis=0)
            ratio_dist_dict['percentile'] = np.nanpercentile(all_ratios[obs_sys], [16, 84, 2.5, 97.5], axis=0)
            all_ratio_dict[obs_sys] = ratio_dist_dict

    else:
        all_dist = []
        for name in m12_radial_dist.keys():
            all_dist.append(np.nanmean([m12_radial_dist[name][j] for j in range(len(sat.redshift))], axis=0))
        for name in lg_radial_dist.keys():
            all_dist.append(np.nanmean([lg_radial_dist[name][j] for j in range(len(sat.redshift))], axis=0))
        
        all_host_ratio = {}
        for obs_sys in obs_samp.keys():
            host_ratio = []
            for host in all_dist:
                for obs_i in obs_samp[obs_sys]:
                    host_ratio.append(np.array(host/obs_i))
            all_host_ratio[obs_sys] = host_ratio

        all_ratio_dict = {}
        for obs_sys in obs_samp.keys():
            ratio_dist_dict = {'mean':np.nanmean(all_host_ratio[obs_sys], axis=0),
                                'median':np.nanmedian(all_host_ratio[obs_sys], axis=0),
                                'percentile':np.nanpercentile(all_host_ratio[obs_sys], [16, 84, 2.5, 97.5], axis=0)}
            all_ratio_dict[obs_sys] = ratio_dist_dict

    return all_ratio_dict

def cumulative_distance(sat, mask_key, norm=False, coadd=False, bins=None, obs=False):
    '''
    Calculate the cumulative distribution of the radial distances of subhalos 
    at z = 0. Does the same for the supplied observational data.
    '''
    if bins is None:
        bins = sat.r_bins
    #if sat.sat_type in ['hal', 'tree']:
    dist_cum = sio.loop_hal(sat, mask_key, mf.cumulative_prop,
                                **{'hal_property':'distance.total',
                                'bins':bins, 'above':False, 'normalized':norm})
    #elif sat.sat_type == 'tree.lg':
    #    dist_cum = sio.loop_hal(sat, mask_key, mf.cumulative_prop_lg,
    #                            **{'hal_property':'distance.total', 'bins':bins,
    #                            'above':False, 'normalized':norm})

    if obs:
        obs_dict = obs.obs_cumulative_prop(sat.observation, sat.observation_mask,
                                       'distance.host', bins, normalized=norm)
    else:
        obs_dict = {}

    if coadd == True:
        dist_dict = mf.coadd_redshift(dist_cum)
        return dist_dict, obs_dict

    elif coadd == False:
        return dist_cum, obs_dict

def nsat_vs_stellar_mass(sat, mask_key, star_mass_limits, radius_limit, obs_sys, host_str='host.'):
    if radius_limit:
        radius_bin = [radius_limit]
    if not radius_limit:
        radius_bin = [sat.r_range[1]]

    calc_sat_list = []
    obs_norm = []
    # get values at each stellar mass cut
    for sm in star_mass_limits:
        sat.star_mass = [sm, 1e10]
        sat.hal_label = sio.hal_label_names(sat)
        sat.observation_mask = obs.obs_mask(sat.observation, star_mass=sat.star_mass, r_range=sat.r_range)

        if sat.sat_type == 'tree':
            sat.tree_mask = sio.mask_tree(sat)
            n_sats = sio.loop_hal(sat,
                                mask_key,
                                mf.cumulative_prop,
                                **{'hal_property':'distance.total',
                                    'bins':radius_bin,'above':False,
                                    'normalized':False})
        elif sat.sat_type == 'tree.lg':
            sat.tree_mask = sio.mask_tree_lg(sat)
            n_sats = sio.loop_hal(sat,
                                mask_key,
                                mf.cumulative_prop_lg,
                                **{'hal_property':'distance.total',
                                    'bins':radius_bin,'above':False,
                                    'normalized':False})
        
        nsats_dict = mf.coadd_redshift(n_sats)
        calc_sat_list.append(nsats_dict)
        
        # get same values for the MW and M31
        obs_dist = obs.obs_cumulative_prop(sat.observation, sat.observation_mask, 'distance.host', radius_bin)
        if obs_sys == 'MW':
            obs_norm.append(obs_dist['MW'][0])
        elif obs_sys == 'M31':
            obs_norm.append(obs_dist['M31'][0])
        else:
            pass
        
    obs_norm = np.array(obs_norm)
        
    # organize values at each stellar mass cut into a single array for plotting
    calc_sat_dict = {}
    name_iter = sio.hal_name_iter(sat)
    for hal_name in name_iter:
        mean = []
        upper_68_lims = [] 
        lower_68_lims = []
        upper_95_lims = []
        lower_95_lims = []
        for k, sm in enumerate(star_mass_limits):
            mean.append(calc_sat_list[k][hal_name]['mean'][0])
            upper_68_lims.append(calc_sat_list[k][hal_name]['percentile'][1][0])
            lower_68_lims.append(calc_sat_list[k][hal_name]['percentile'][0][0])
            upper_95_lims.append(calc_sat_list[k][hal_name]['percentile'][3][0])
            lower_95_lims.append(calc_sat_list[k][hal_name]['percentile'][2][0])
            
        calc_sat_dict[hal_name] = {'mean':np.array(mean), 'percent_68':np.array([lower_68_lims, upper_68_lims]),
                                   'percent_95':np.array([lower_95_lims, upper_95_lims])}

    return calc_sat_dict, obs_norm

def nsat_vs_stellar_mass_all_hosts(sat, mask_key, star_mass_limits, radius_limit):
    if radius_limit:
        radius_bin = [radius_limit]
    if not radius_limit:
        radius_bin = [sat.r_range[1]]

    calc_sat_dict = {}
    obs_norm = {'MW':[], 'M31':[]}
    # get values at each stellar mass cut
    for sm in star_mass_limits:
        sat.star_mass = [sm, 1e10]
        sat.hal_label = sio.hal_label_names(sat)
        sat.observation_mask = obs.obs_mask(sat.observation, star_mass=sat.star_mass, r_range=sat.r_range)

        if sat.sat_type == 'tree':
            sat.tree_mask = sio.mask_tree(sat)
            n_sats = sio.loop_hal(sat,
                                mask_key,
                                mf.cumulative_prop,
                                **{'hal_property':'distance.total',
                                    'bins':radius_bin,'above':False,
                                    'normalized':False})
        elif sat.sat_type == 'tree.lg':
            sat.tree_mask = sio.mask_tree_lg(sat)
            n_sats = sio.loop_hal(sat,
                                mask_key,
                                mf.cumulative_prop_lg,
                                **{'hal_property':'distance.total',
                                    'bins':radius_bin,'above':False,
                                    'normalized':False})
        
        calc_sat_dict[str(sm)] = (n_sats)

        # get same values for the MW and M31
        obs_dist = obs.obs_cumulative_prop(sat.observation, sat.observation_mask, 'distance.host', radius_bin)
        obs_norm['MW'].append(obs_dist['MW'][0])
        obs_norm['M31'].append(obs_dist['M31'][0])

    mean = []
    median = []
    percentile = [] 

    name_iter = sio.hal_name_iter(sat)
    for sm in calc_sat_dict.keys():
        sm_calc_list = []
        for hal_name in name_iter:
            hal_array = calc_sat_dict[sm][hal_name]
            for ha in hal_array:
                sm_calc_list.append(ha[0])
        mean.append(np.average(sm_calc_list))
        median.append(np.median(sm_calc_list))
        percentile.append(np.nanpercentile(sm_calc_list, [16, 84, 2.5, 97.5], axis=0))

    calc_sm_dict = {'mean':mean, 'median':median, 'percentile':percentile}

    return calc_sm_dict, obs_norm

def nsat_vs_stellar_mass_m12_and_lg(
    sat, sat_lg, mask_key, star_mass_limits, radius_limit, obs_sample=True, 
    obs_dir='./', n_iter=1000, stat='median'):
    if radius_limit:
        radius_bin = [radius_limit]
    if not radius_limit:
        radius_bin = [sat.r_range[1]]

    calc_sat_dict = {}
    m12_sat_dict = {}
    lg_sat_dict = {}
    obs_norm = {'MW':[], 'M31':[]}
    obs_samps = {'MW':[], 'M31':[]}
    # get values at each stellar mass cut
    for sm in star_mass_limits:
        if not sm == sat.star_mass[0]:
            sat = sio.reset_sm_limits(sat, sm, obs=False)
            sat_lg = sio.reset_sm_limits(sat_lg, sm, obs=False)

        m12_nsat = sio.loop_hal(sat,
                                mask_key,
                                mf.cumulative_prop,
                                **{'hal_property':'distance.total',
                                    'bins':radius_bin,'above':False,
                                    'normalized':False})

        lg_nsat = sio.loop_hal(sat_lg,
                                mask_key,
                                mf.cumulative_prop,
                                **{'hal_property':'distance.total',
                                    'bins':radius_bin,'above':False,
                                    'normalized':False})
        
        m12_sat_dict[str(sm)] = m12_nsat
        lg_sat_dict[str(sm)] = lg_nsat
        # merge m12 and lg dictionaries
        calc_sat_dict[str(sm)] = {**m12_nsat, **lg_nsat}
        
        # get same values for the MW and M31
        if obs_sample:
            mw_sm_dict, mw_sm_samp = obs.mw_sample(table_dir=obs_dir,
                                                    n_iter=n_iter, 
                                                    rbins=[radius_limit], 
                                                    star_mass=sm)
            m31_sm_dict, m31_sm_samp = obs.m31_sample(table_dir=obs_dir,
                                                    n_iter=n_iter, 
                                                     rbins=[radius_limit], 
                                                     star_mass=sm)
            #print(m31_sm_dict, m31_sm_samp)
            obs_norm['MW'].append(mw_sm_dict[stat][0])
            obs_norm['M31'].append(m31_sm_dict[stat][0])
            obs_samps['MW'].append([mws[0] for mws in mw_sm_samp])
            obs_samps['M31'].append([m31s[0] for m31s in m31_sm_samp])
        else:
            obs_dist = obs.obs_cumulative_prop(sat.observation, 
                                              sat.observation_mask, 
                                              'distance.host', 
                                              radius_bin)
            obs_norm['MW'].append(obs_dist['MW'][0])
            obs_norm['M31'].append(obs_dist['M31'][0])

    name_combine = sio.hal_name_iter(sat) + sio.hal_name_iter(sat_lg)
    all_sm_dict = sio.sort_by_key_coadd(sat, calc_sat_dict, name_iter=name_combine)
    m12_sm_dict = sio.sort_by_key_coadd(sat, m12_sat_dict)
    lg_sm_dict = sio.sort_by_key_coadd(sat_lg, lg_sat_dict)

    if obs_sample:
        mw_ratio_dict = sio.sort_by_key_coadd(sat, calc_sat_dict, 
                                             name_iter=name_combine,
                                             obs_sample=obs_samps['MW'])
        m31_ratio_dict = sio.sort_by_key_coadd(sat, calc_sat_dict, 
                                              name_iter=name_combine,
                                              obs_sample=obs_samps['M31'])
        obs_ratio_dict = {'MW':mw_ratio_dict, 'M31':m31_ratio_dict}

    else:
        mw_ratio_dict = sio.sort_by_key_coadd(sat, calc_sat_dict, 
                                             name_iter=name_combine, 
                                             obs_sat_dict=obs_norm, 
                                             obs_sys='MW')
        m31_ratio_dict = sio.sort_by_key_coadd(sat, calc_sat_dict, 
                                              name_iter=name_combine, 
                                              obs_sat_dict=obs_norm, 
                                              obs_sys='M31')
        obs_ratio_dict = {'MW':mw_ratio_dict, 'M31':m31_ratio_dict}

    return all_sm_dict, m12_sm_dict, lg_sm_dict, obs_norm, obs_ratio_dict

def sat_align_moi(hal, hal_mask=None):
    '''
    Take the positions and velocities of subhalos from the simulation and rotate
    them onto the axes defined by the inertia tensor of the satellite distribution.
    '''
    hal_mask = sio.default_mask(hal, hal_mask)
    sat_axes = get_satellite_principal_axes(hal, hal_mask)

    sat_coords = hal.prop('host.distance')[hal_mask]
    sat_prime_coords = ut.basic.coordinate.get_coordinates_rotated(sat_coords, rotation_tensor=sat_axes[0])

    sat_vels = hal.prop('host.velocity')[hal_mask]
    sat_prime_vels = ut.basic.coordinate.get_coordinates_rotated(sat_vels, rotation_tensor=sat_axes[0])

    return sat_prime_coords, sat_prime_vels

@jit
def axis_ratio(hal, hal_mask=None, host_str='host.', return_ax=False):
    '''
    Get the axis ratio (minor/major) for the total distribution of satellites
    within the fiducial virial radius of the host halo.
    '''
    hal_mask = sio.default_mask(hal, hal_mask)
    sat_axes = get_satellite_principal_axes(hal, hal_mask, host_str=host_str)

    if return_ax is True:
        return {'axis.ratio':sat_axes[2][0], 'ax':sat_axes[0][2]}
    else:
        return sat_axes[2][0]
        
@jit
def rms_distance(hal, hal_mask=None, host_str='host.'):
    '''
    Get the rms of major (plane 'x', or width) and minor (plane 'z', or height)
    axes for the total distribution of satellites within the fiducial virial
    radius of the host halo.
    '''
    hal_mask = sio.default_mask(hal, hal_mask)
    sat_coords = hal.prop(host_str+'distance')[hal_mask]
    
    sat_axes = get_satellite_principal_axes(hal, hal_mask)
    sat_prime_coords = ut.basic.coordinate.get_coordinates_rotated(sat_coords, rotation_tensor=sat_axes[0])
    rms_z = np.sqrt(np.mean(sat_prime_coords[:,2]**2))
    rms_x = np.sqrt(np.mean(sat_prime_coords[:,0]**2))
            
    return {'rmsx':rms_x, 'rmsz':rms_z}

@jit
def rms_v_fraction(hal, hal_mask=None, host_str='host.', angle_bins=None):
    '''
    Find the rms (height and projected 'radius') as a function of enclosing
    angle. Satellite distribution MOI axes are recomputed for those satellites
    within each enclosing angle.
    '''
    hal_mask = sio.default_mask(hal, hal_mask)
    angle_range, angle_mask = ang.open_angle_mask(hal, hal_mask=hal_mask, angle_bins=angle_bins)

    rms_major = []
    rms_minor = []

    for k, l in enumerate(angle_range):
        sat_coords = hal.prop(host_str+'distance')[hal_mask]
        sat_coords = sat_coords[angle_mask[k]]

        sat_axes = {}
        sat_axes[str(l)] = ut.coordinate.get_principal_axes(sat_coords)
        sat_prime_coords = ut.basic.coordinate.get_coordinates_rotated(sat_coords, rotation_tensor=sat_axes[str(l)][0])

        rms_major.append(np.sqrt(np.mean(sat_prime_coords[:,0]**2)))
        rms_minor.append(np.sqrt(np.mean(sat_prime_coords[:,2]**2)))

    return {'rmsx':rms_major, 'rmsz':rms_minor}

def r_fraction(hal, hal_mask=None, host_str='host.', frac=0.1, radius_bins=None):
    '''
    Find the radius (r) that encloses the given fraction of satellites (frac).
    Form a list of these radii for each redshift (host_r) for each host (r_all).
    '''
    hal_mask = sio.default_mask(hal, hal_mask)
    prop_str = '{}{}'.format(host_str, 'distance.total')
    radii = hal.prop(prop_str)[hal_mask]

    if type(frac) == list:
        frac_r = []
        for f in frac:
            for rad in radius_bins:
                radius_mask = radii <= rad
                frac_enclosed = float(len(radii[radius_mask]))/float(len(radii))
                
                if frac_enclosed >= f:
                    frac_r.append(rad)
                    break

                else:
                    pass

        return frac_r

    elif type(frac) == float:
        for rad in radius_bins:
            radius_mask = radii <= rad
            frac_enclosed = float(len(radii[radius_mask]))/float(len(radii))
            
            if frac_enclosed >= frac:
                break
            else:
                pass

        return rad

def rfrac_vs_rfrac(hal, hal_mask=None, host_str='host.', frac1=0.9, frac2=0.1, radius_bins=None):
    '''
    Find the ratio of the radii enclosing frac1 and frac2 of the satellites.
    (R1/R2).
    '''
    rf1 = r_fraction(hal, hal_mask=hal_mask, host_str=host_str, frac=frac1, radius_bins=radius_bins)
    rf2 = r_fraction(hal, hal_mask=hal_mask, host_str=host_str, frac=frac2, radius_bins=radius_bins)

    return rf1/rf2

def rms_maj(sat_coords):
    '''
    Calculates rms deviation of the given coordinates along the major axis of
    the inertia tensor defined by those coordinates.
    '''
    sat_axes = ut.coordinate.get_principal_axes(sat_coords)
    sat_prime_coords = ut.basic.coordinate.get_coordinates_rotated(sat_coords, rotation_tensor=sat_axes[0])
    rms_maj = np.sqrt(np.mean(sat_prime_coords[:,0]**2))
    #rms_maj = np.std(abs(sat_prime_coords[:,0]))

    return rms_maj

def rms_min(sat_coords):
    '''
    Calculates rms deviation of the given coordinates along the minor axis of
    the inertia tensor defined by those coordinates.
    '''
    sat_axes = ut.coordinate.get_principal_axes(sat_coords)
    sat_prime_coords = ut.basic.coordinate.get_coordinates_rotated(sat_coords, rotation_tensor=sat_axes[0])
    rms_min = np.sqrt(np.mean(sat_prime_coords[:,2]**2))
    #rms_min = np.std(abs(sat_prime_coords[:,2]))

    return rms_min

def rms_vs_r_frac(hal, hal_mask=None, host_str='host.', r_frac=0.68, radius_bins=None):
    '''
    Find the rms (height and 'radius') of satellites within a radius that encloses
    a given (single) fraction of satellites (r_frac).
    '''
    rads = r_fraction(hal, hal_mask=hal_mask, host_str=host_str, frac=r_frac, radius_bins=radius_bins)
    tot_str = '{}{}'.format(host_str, 'distance.total')
    dist3d_str = '{}{}'.format(host_str, 'distance')
    # make rads iterable if it is not already
    try:
        iter(rads)
        rms_major = []
        rms_minor = []
        for r in rads:
            radial_mask = hal.prop(tot_str) <= r
            sat_coords = hal.prop(dist3d_str)[hal_mask & radial_mask]

            rms_major.append(rms_maj(sat_coords))
            rms_minor.append(rms_min(sat_coords))
    except TypeError:
        for r in np.array([rads]):
            radial_mask = hal.prop(tot_str) <= r
            sat_coords = hal.prop(dist3d_str)[hal_mask & radial_mask]

            rms_major = rms_maj(sat_coords)
            rms_minor = rms_min(sat_coords)

    return {'rmsx':rms_major, 'rmsz':rms_minor, 'radii':rads}

def rand_2d_proj(hal, hal_mask=None, host_str='host.', n_iter=1000):
    dist3d_str = '{}{}'.format(host_str, 'distance')
    sat_coords = hal.prop(dist3d_str)[hal_mask]
    rot_vecs, rot_mats = ra.rand_rot_vec(n_iter)
    proj_2d_dist = []

    for n in range(n_iter):
        sat_coords_rot = ut.basic.coordinate.get_coordinates_rotated(sat_coords, rotation_tensor=rot_vecs[n])
        proj_2d = np.sqrt(sat_coords_rot[:,1]**2 + sat_coords_rot[:,2]**2)
        proj_2d_dist.append(proj_2d)

    return proj_2d_dist

def cumul_2d_proj(hal, hal_mask=None, host_str='host.', n_iter=1000, rbins=None):
    proj_2d_dist = rand_2d_proj(hal, hal_mask, host_str, n_iter)
    cumul_proj_2d_n = []
    for proj_2d in proj_2d_dist:
        cumul_proj_2d = []
        for _bin in rbins:
            bin_mask = proj_2d <= _bin
            cumul_proj_2d.append(len(proj_2d[bin_mask]))
        cumul_proj_2d_n.append(cumul_proj_2d)

    return cumul_proj_2d_n

def mask_on_2d_proj(
    hal, hal_mask=None, host_str='host.', n_iter=1000, rbins=None, rlim=None,
    complete_limit=1e7):
    if rlim is None:
        rlim = np.max(rbins)
    dist3d_str = '{}{}'.format(host_str, 'distance')
    sat_coords = hal.prop(dist3d_str)[hal_mask]
    sat_3d_dist = hal.prop(host_str+'distance.total')[hal_mask]
    rot_vecs, rot_mats = ra.rand_rot_vec(n_iter)
    all_3d_proj_masked_profiles = []
    sat_star_mass = hal.prop('star.mass')[hal_mask]

    for n in range(n_iter):
        sat_coords_rot = ut.basic.coordinate.get_coordinates_rotated(sat_coords, rotation_tensor=rot_vecs[n])
        proj_2d = np.sqrt(sat_coords_rot[:,1]**2 + sat_coords_rot[:,2]**2)
        proj_mask = proj_2d <= rlim
        outer_comp_mask = (sat_star_mass > complete_limit) & (proj_2d > rlim)
        sat_3d_dist_proj_masked = sat_3d_dist[proj_mask | outer_comp_mask]

        cumulative_dist = []
        for _bin in rbins:
            bin_mask = sat_3d_dist_proj_masked <= _bin
            cumulative_dist.append(np.sum(bin_mask))
        all_3d_proj_masked_profiles.append(cumulative_dist)

    return all_3d_proj_masked_profiles

### below needs development/checking

# write a version that doesn't need to be coadded later to avoid having to know about 
# more that one hal at a time. for now, just use this with loop_sat (not loop_hal)
def rms_min_vs_r(sat, mask_key, hal_name, redshift_index):
    '''
    Find the rms height of satellites within a range of radii using the rms deviation
    of their positions along the minor axis of the satellite distribution inertia
    tensor. 
    '''
    hal_rms_minor = []
    hal = sat.hal_catalog[hal_name][redshift_index]
    hal_mask = sat.catalog_mask[mask_key][hal_name][redshift_index]

    host_hals = sat.hal_catalog[hal_name]
    host_masks = sat.catalog_mask[mask_key][hal_name]

    rad_min = max([np.sort(host_hals[k].prop('distance.total')[host_masks[k]])[2] for k in range(len(sat.redshift))])
    rad_min_round = mf.round_up(rad_min, base=25)
    rads = np.arange(rad_min_round, sat.r_range[1]+25, 25)
    rads = [r for r in rads]

    for r in rads:
        #snap_name = str(r)
        radial_mask = hal.prop('distance.total') <= r
        sat_coords = hal.prop('host.distance')[hal_mask & radial_mask]
        rms_z = rms_min(sat_coords)
        hal_rms_minor.append(rms_z)

    return {'rmsz':np.array(hal_rms_minor), 'radii':rads}

def rmsz_vs_r(sat, isotropic=False):
    '''
    Wrapper for the functions called to be implemented in a loop inside of a
    plot_spatial function. Calculates rms height as a function of radius and
    and coadds this over redshift.
    '''
    if isotropic == True:
        rms_z_radii_dict = sio.loop_sat(sat, iso.iso_rms_min_vs_r)
        coadd_rms_z = mf.coadd_redshift(rms_z_radii_dict, dict_key='rmsz')
    
    elif isotropic == False:
        rms_z_radii_dict = sio.loop_sat(sat, rms_min_vs_r)
        coadd_rms_z = mf.coadd_redshift(rms_z_radii_dict, dict_key='rmsz')

    else:
        raise ValueError('\'isotropic\' keyword must be True or False, currently set to: \'{}\''.format(isotropic))

    return rms_z_radii_dict, coadd_rms_z
