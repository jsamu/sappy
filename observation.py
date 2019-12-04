import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from astropy.table import Table
from astropy.io import ascii as asc
import astropy.coordinates as coord
from astropy.coordinates import SkyCoord
import astropy.units as u
from satellite_analysis import math_funcs as mf
from satellite_analysis import satellite_io as sio
from satellite_analysis import spatial as spa

def load_obs(
    table_dir='./',
    MW_data_file='mw_sats.txt',
    M31_data_file='m31_sats.txt'):

    observation = {}
    observation['MW'] = pd.read_csv(table_dir+MW_data_file, sep=' ')
    observation['M31'] = pd.read_csv(table_dir+M31_data_file, sep=' ')

    return observation

def cumul_dist(radial_dist, bins=np.arange(0, 310, 10)):
    if type(radial_dist) is not np.ndarray:
        radial_dist = np.array(radial_dist)
    cumulative_dist = []
    for _bin in bins:
        bin_mask = radial_dist <= _bin
        cumulative_dist.append(np.sum(bin_mask))
    return cumulative_dist

def diff_radial_profile(radial_distance, bins=np.arange(0, 310, 10)):
    radial_distance = np.array(radial_distance)
    dn_dr = []
    for i in range(len(bins)):
        if i == 0:
            dn = (radial_distance <= bins[i]) & (radial_distance > 0)
        else:
            dn = (radial_distance <= bins[i]) & (radial_distance > bins[i-1])
        dn_dr.append(np.sum(dn))
    return np.array(dn_dr)

def mw_sample(
    table_dir='./', MW_data_file='mw_sats_distances.txt', sky_pos_file='mccon2012.vot',
    n_iter=1000, rbins=np.arange(0,310,10), star_mass=1e5, no_bin=False, diff=False):

    # load observational data
    mw_dists = pd.read_csv(table_dir+MW_data_file, sep=' ')

    # positions on the sky from McConnachie 2012
    mccon_tab = Table.read(table_dir+sky_pos_file, format='votable')
    mccon_tab.keep_columns(['Name', 'RAJ2000', 'DEJ2000'])
    mccon_tab.add_row(['Crater 2', 177.310*u.degree, -18.413*u.degree])
    mccon_tab.add_row(['Antlia 2', 143.8868*u.degree, -36.7673*u.degree])
    mccon_tab['coo'] = SkyCoord(mccon_tab['RAJ2000'], mccon_tab['DEJ2000'], unit=(u.degree, u.degree))

    # get indices of MW satellites in McConnachie table
    mw_idx = []
    for key in mw_dists['Name']:
        lkey = key.lower()
        for i, nm in enumerate(mccon_tab['Name']):
            if nm == 'The Galaxy':
                nm = 'MW'
            if nm.replace(' ', '').lower() == lkey:
                mw_idx.append(i)
                break
            elif nm == key:
                mw_idx.append(i)
                break
        else:
            print(f'Warning: did not find {key}!')
            mw_idx.append(-10)
    mw_dists['midx'] = mw_idx

    # generate random 3D projection convolving in uncertainties on distances
    dist_samples = {}
    for j, nm in enumerate(mw_dists['Name']):
        # mask on stellar mass
        if mw_dists['Star_mass'][j] > star_mass:
            midx = mw_dists['midx'][j]
            # get heliocentric coords from ra, dec, and distance
            hc = coord.ICRS(ra=mccon_tab[midx]['coo'].ra,
                            dec=mccon_tab[midx]['coo'].dec,
                            distance=np.random.normal(loc=mw_dists['D_sun'][j], 
                            scale=mw_dists['D_sun_unc'][j], size=n_iter)*u.kpc)
            # get galactocentric coords
            gc = hc.transform_to(coord.Galactocentric)
            sep_r = gc.separation_3d(coord.Galactocentric(0*u.kpc,0*u.kpc,0*u.kpc))
            dist_samples[nm] = sep_r

    # bin the observational radial profiles to compare at fixed distances
    mw_n = []
    for i in range(n_iter):
        rad = []
        for nm in dist_samples.keys():
            rad.append(dist_samples[nm][i].kpc)
        mw_n.append(rad)

    mw_n_samp = {'mean':np.nanmean(mw_n, axis=0),
                'median':np.nanmedian(mw_n, axis=0),
                'percentile':np.nanpercentile(mw_n, 
                        [16, 84, 2.5, 97.5], axis=0, interpolation='nearest')}
            
    binned_profile_mw = []
    if diff is True:
        for i in range(n_iter):
            binned_profile_mw.append(diff_radial_profile(mw_n[i], bins=rbins))
    else:
        for i in range(n_iter):
            binned_profile_mw.append(cumul_dist(mw_n[i], bins=rbins))
        
    mw_samp = {'mean':np.nanmean(binned_profile_mw, axis=0),
                'median':np.nanmedian(binned_profile_mw, axis=0),
                'percentile':np.nanpercentile(binned_profile_mw, 
                        [16, 84, 2.5, 97.5], axis=0, interpolation='nearest')}

    if no_bin:
        return mw_n_samp, mw_n
    else:
        return mw_samp, binned_profile_mw

def m31_sample(
    table_dir='./', M31_data_file='conn_m31_sats_posteriors.txt', 
    sky_pos_file='mccon2012.vot', n_iter=1000, rbins=np.arange(0,310,10), 
    star_mass=1e5, M31_sm_file='m31_sats.txt', no_bin=False, diff=False):

    # get stellar masses of M31 satellites
    m31_sm = pd.read_csv(table_dir+M31_sm_file, sep=' ')
    m31_sm_names = list(m31_sm['name'][m31_sm['star.mass'] >= star_mass])
    m31_sm_names.append('M31')
    if star_mass <= 9.64e5:
        m31_sm_names.append('LGS 3 (Local Group Suspect 3)')

    # read in M31 satellite distance posteriors from Conn et al 2012
    conntab = Table.read(table_dir+M31_data_file, format='ascii')
    grouped_tab = conntab.group_by('Name')
    gal_tabs = {g['Name'][0]:g for g in grouped_tab.groups}
    # by-hand insert M32 and NGC205 with the same distance distribution as M31
    gal_tabs['M32'] = gal_tabs['M31'].copy()
    gal_tabs['NGC205'] = gal_tabs['M31'].copy()
    # insert distributions for other satellites with pnly published distances 
    # and uncertainties, will replace with their own distributions later
    gal_tabs['IC10'] = gal_tabs['M31'].copy()
    gal_tabs['AndromedaVI'] = gal_tabs['M31'].copy()
    gal_tabs['AndromedaVII'] = gal_tabs['M31'].copy()
    gal_tabs['AndromedaXXIX'] = gal_tabs['M31'].copy()
    gal_tabs['LGS 3 (Local Group Suspect 3)'] = gal_tabs['M31'].copy()
    gal_tabs['AndromedaXXXI'] = gal_tabs['M31'].copy()
    gal_tabs['AndromedaXXXII'] = gal_tabs['M31'].copy()

    # get sky coordinates of M31 satellites from McConnachie 2012
    mccon_tab = Table.read(table_dir+sky_pos_file, format='votable')
    # add satellites with coords published later
    mccon_tab.keep_columns(['Name', 'RAJ2000', 'DEJ2000'])
    mccon_tab.add_row(['Andromeda XXX', '0 36 21', '+49 36 40'])
    mccon_tab.add_row(['Andromeda XXXI', '22 58 16.3', '+41 17 28'])
    mccon_tab.add_row(['Andromeda XXXII', '0 35 59.4', '+51 33 35'])
    mccon_tab['coo'] = SkyCoord(mccon_tab['RAJ2000'], mccon_tab['DEJ2000'], unit=(u.hour, u.degree))

    # get indices of M31 satellites in McConnachie table
    for key, tab in gal_tabs.items():
        lkey = key.lower()
        for i, nm in enumerate(mccon_tab['Name']):
            if nm == 'Andromeda':
                nm = 'M31'
            elif nm == 'Triangulum':
                nm = 'M33'
            if nm.replace(' ', '').lower() == lkey:
                tab.mccon_idx = i
                break
            elif nm == key:
                tab.mccon_idx = i
                break
        else:
            print(f'Warning: did not find {key}!')
            tab.mccon_idx = None

    # sample random distances convolving in uncertainties,
    # generate distributions for those satellites not in Conn et al 2012
    sat_names_gen = ['IC10', 'AndromedaVI', 'AndromedaVII', 'AndromedaXXIX', 'LGS 3 (Local Group Suspect 3)', 
                    'AndromedaXXXI', 'AndromedaXXXII']
    sat_dist_gen = {'IC10':794, 'AndromedaVI':783, 'AndromedaVII':762, 'AndromedaXXIX':731, 
                        'LGS 3 (Local Group Suspect 3)':769, 'AndromedaXXXI':756, 'AndromedaXXXII':772}
    sat_dist_errs_gen = {'IC10':44, 'AndromedaVI':25, 'AndromedaVII':35, 'AndromedaXXIX':74, 
                            'LGS 3 (Local Group Suspect 3)':25, 'AndromedaXXXI':44, 'AndromedaXXXII':61}

    dist_samples = {}
    for nm, tab in gal_tabs.items():
        if nm in m31_sm_names:
            if nm in sat_names_gen:
                dist_samples[nm] = np.random.normal(loc=sat_dist_gen[nm], scale=sat_dist_errs_gen[nm], size=n_iter)
            else:
                percs = np.random.uniform(0, 100, size=n_iter)
                dist_samples[nm] = np.interp(percs, tab['Per'], tab['Dis(no)'])
        else:
            pass

    # convert to 3D distances from M31
    nm = 'M31'
    midx = gal_tabs[nm].mccon_idx
    m31sc = SkyCoord(mccon_tab[midx]['RAJ2000'], mccon_tab[midx]['DEJ2000'], unit=(u.hour, u.degree), 
                    distance=dist_samples[nm]*u.kpc)
    dist_3ds = {}
    for nm, tab in gal_tabs.items():
        if nm == 'M31':
            continue
        midx = tab.mccon_idx
        if midx is None:
            continue
        if nm in m31_sm_names:
            galsc = SkyCoord(mccon_tab[midx]['RAJ2000'], mccon_tab[midx]['DEJ2000'], unit=(u.hour, u.degree), 
                            distance=dist_samples[nm]*u.kpc)
            dist_3ds[nm] = galsc.separation_3d(m31sc)
    
    # string together each satellite's sample to make radial distributions
    m31_n = []
    #for i in range(n_iter):
    #    m31_rd = [m31sat[i] for m31sat in list(dist_3ds.values())]
    #    m31_n.append(np.array([rd.value for rd in m31_rd[i]]))
    for i in range(n_iter):
        rad = []
        for nm in dist_3ds.keys():
            rad.append(dist_3ds[nm][i].kpc)
        m31_n.append(rad)

    m31_n_samp = {'mean':np.nanmean(m31_n, axis=0),
                'median':np.nanmedian(m31_n, axis=0),
                'percentile':np.nanpercentile(m31_n, 
                        [16, 84, 2.5, 97.5], axis=0, interpolation='nearest')}

    binned_profile_m31 = []
    if diff is True:
        for i in range(n_iter):
            binned_profile_m31.append(diff_radial_profile(m31_n[i], bins=rbins))
    else:
        for i in range(n_iter):
            binned_profile_m31.append(cumul_dist(m31_n[i], bins=rbins))
        
    m31_samp = {'mean':np.nanmean(binned_profile_m31, axis=0),
                'median':np.nanmedian(binned_profile_m31, axis=0),
                'percentile':np.nanpercentile(binned_profile_m31, 
                        [16, 84, 2.5, 97.5], axis=0, interpolation='nearest')}

    if no_bin:
        return m31_n_samp, m31_n
    else:
        return m31_samp, binned_profile_m31

def obs_mask(observation, star_mass=[1e5, 1e10], r_range=[5,300]):
    '''
    Create masks for observational data loaded in by localgroup_analysis.
    '''
    masks = {}
    for gal in observation.keys():
        mask_star_mass_low = np.array(observation[gal]['star.mass']) >= star_mass[0]
        mask_star_mass_high = np.array(observation[gal]['star.mass']) <= star_mass[1]
        distance_mask = np.array(observation[gal]['distance.host']) <= r_range[1]
        masks[gal] = mask_star_mass_low & mask_star_mass_high & distance_mask

    return masks

def obs_cumulative_prop(observation, obs_mask, sat_property, bin_list, above=False, normalized=False):
    '''
    Get a cumulative distribution of an observational quantity loaded in by
    localgroup_analysis.
    '''
    obs_cumulative_property = {}
    for gal in observation.keys():
        cumulative_property = []
        if above == False:
            for _bin in bin_list:
                bin_mask = np.array(observation[gal][sat_property]) <= _bin
                cumulative_property.append(np.sum(bin_mask & obs_mask[gal]))

        elif above == True:
            for _bin in bin_list:
                bin_mask = np.array(observation[gal][sat_property]) >= _bin
                cumulative_property.append(np.sum(bin_mask & obs_mask[gal]))

        if normalized == True:
            cumulative_property = cumulative_property/np.max(cumulative_property)

        obs_cumulative_property[gal] = cumulative_property

    return obs_cumulative_property

### used with old lg files

def obs_mask_old(gal):
    '''
    Create masks for observational data loaded in by localgroup_analysis.
    '''
    mask_star_mass_low = gal.prop('star.mass') >= 10**5
    mask_star_mass_high = gal.prop('star.mass') <= 10**10
    sm_nan = ~np.isnan(gal.prop('star.mass'))
    #mask_star_density = gal.prop('star.density.50') >= 10**4
    #sd_nan = ~np.isnan(gal.prop('star.density.50'))
    distance_mask = gal.prop('host.distance.total') <= 300
    d_nan = ~np.isnan(gal.prop('host.distance.total'))
    combined_mask = mask_star_mass_low & mask_star_mass_high & distance_mask & sm_nan & d_nan
    MW_name = gal.prop('host.name') == b'MW'
    MW_mask = MW_name & combined_mask
    M31_name = gal.prop('host.name') == b'M31'
    M31_mask = M31_name & combined_mask

    return MW_mask, M31_mask

def obs_cumulative_prop_old(property_list, bin_list, above=False, normalized=False):
    '''
    Get a cumulative distribution of an observational quantity loaded in by
    localgroup_analysis.
    '''
    cumulative_property = []
    if above == False:

        for _bin in bin_list:
            bin_mask = property_list <= _bin
            cumulative_property.append(len(property_list[bin_mask]))

    elif above == True:

        for _bin in bin_list:
            bin_mask = property_list >= _bin
            cumulative_property.append(len(property_list[bin_mask]))

    if normalized == True:
        cumulative_property = cumulative_property/np.max(cumulative_property)

    return cumulative_property

def mw_match(hal, hal_mask=None, host_str='host.', mw_value=10):
    rad_profile = mf.cumulative_prop(hal, hal_mask, 
        hal_property='distance.total', host_str=host_str, 
        bins=np.arange(0,310,10))
    if rad_profile[15] == mw_value:
        return rad_profile

def mw_predict(
    m12_sat, lg_sat, rbins=np.arange(0,310,10), title=None, ylim=(0,25), 
    figdir='./', table_dir='./'):

    font = {'size'   : 22}
    plt.rc('font', **font)

    CB3_0 = ['#09B0C1', '#175F02', '#FF9D07']
    CB3_1 = ['#0DBDC7', '#B906AF', '#E49507']

    mw_dict, mw_samp = mw_sample(table_dir=table_dir, 
                            MW_data_file='mw_sats_distances.txt', 
                            sky_pos_file='mccon2012.vot',
                            n_iter=1000, 
                            rbins=rbins, 
                            star_mass=m12_sat.star_mass[0])
    fig, ax = plt.subplots(1,1, figsize=(7,6))
    fig.set_tight_layout(False)
    ax.fill_between(rbins, mw_dict['percentile'][2], mw_dict['percentile'][3], color='k', 
                    alpha=0.25, linewidth=0)
    ax.fill_between(rbins, mw_dict['percentile'][0], mw_dict['percentile'][1], color='k', 
                    alpha=0.25, linewidth=0)
    ax.plot(rbins, mw_dict['median'], color='k', label='Milky Way', linestyle='-')
    mw_at_150_kpc = mw_dict['median'][15]

    m12_mw_match = sio.loop_hal(m12_sat, 'star.mass', mw_match, **{'mw_value':mw_at_150_kpc})
    lg_mw_match = sio.loop_hal(lg_sat, 'star.mass', mw_match, **{'mw_value':mw_at_150_kpc})
    all_mw_match = {**m12_mw_match, **lg_mw_match}

    filtered_mw_match = {}
    for host in all_mw_match.keys():
        filtered_list = []
        for z in all_mw_match[host]:
            if z is not None:
                filtered_list.append(z)
        if len(filtered_list) > 0:
            filtered_mw_match[host] = filtered_list
    coadd_mw_match = mf.coadd_redshift(filtered_mw_match, all_sims=True)


    ax.fill_between(rbins, coadd_mw_match['percentile'][2], 
        coadd_mw_match['percentile'][3], alpha=0.25, 
        color=CB3_1[0], linewidth=0)
    ax.fill_between(rbins, coadd_mw_match['percentile'][0], 
        coadd_mw_match['percentile'][1], alpha=0.25, color=CB3_1[0], 
        linewidth=0)
    ax.plot(rbins, coadd_mw_match['median'], color=CB3_1[0], linewidth=2, 
        label='hosts matched to MW at 150 kpc')

    plt.legend(loc=2, fontsize=20, title=title)
    plt.xlabel('Distance from host [kpc]', fontsize=24)
    plt.ylabel(r'N$_{\rm sat}(<\rm{d})$', fontsize=26)
    ax.tick_params(axis='both', which='major', labelsize=22)
    ax.set_xlim((0,np.max(rbins)))
    fig.subplots_adjust(left=0.15, bottom=0.15, wspace=0.1, hspace=0.1)
    ax.set_yticks(np.arange(5,30,5), minor=False)
    ax.set_xticks(np.arange(0,350,50), minor=False)
    ax.set_ylim(ylim)
    fig.savefig(figdir+'mw_prediction_10_{}.pdf'.format(np.log10(m12_sat.star_mass[0])))

def m31_predict(
    m12_sat, lg_sat, rbins=np.arange(0,310,10), title=None, ylim=(0,25),
    figdir='./', table_dir='./', M31_data=False):
    
    # change default font size
    font = {'size'   : 22}
    plt.rc('font', **font)

    CB3_0 = ['#09B0C1', '#175F02', '#FF9D07']
    CB3_1 = ['#0DBDC7', '#B906AF', '#E49507']  
    rbins = m12_sat.r_bins
    comp_limit_title = [r'M$_{*, \rm{complete}}$(d$>$150 kpc)$ = 10^6$ M$_{\odot}$',
        r'M$_{*, \rm{complete}}$(d$>$150 kpc)$ = 10^7$ M$_{\odot}$']
    if M31_data is True:
        # M31 data
        m31_dict, m31_samp = m31_sample(table_dir=table_dir, 
                                M31_data_file='conn_m31_sats_posteriors.txt', 
                                sky_pos_file='mccon2012.vot',
                                n_iter=1000, rbins=rbins, 
                                star_mass=1e5, M31_sm_file='m31_sats.txt')
        rbins = np.arange(0, 310, 10)
        m31_rbins1_mask = rbins <= 150
        m31_rbins2_mask = rbins >= 150
        m31_rbins1 = np.arange(0, 160, 10)
        m31_rbins2 = np.arange(150, 310, 10)


    name = 'm12m'
    m31_like_profiles = ['m12m', 'm12c', 'm12w', 'Juliet', 'Louise']
    for j, comp_limit in enumerate([1e6, 1e7]):
        '''
        all_subplots = ps.radial_2d_vs_3d_subplots(
            m12_sat, lg_sat, 'star.mass', stat='median', n_iter=1000, rbins=None, 
            rlim=150, M31_data=True, all_sims=False, complete_limit=comp_limit)
        all_subplots.savefig(figdir+'all_subplots_10_{}_z{}.pdf'.format(np.log10(comp_limit), np.max(redshifts_01)))

        all_coadd_plot = ps.radial_2d_vs_3d_subplots(
            m12_sat, lg_sat, 'star.mass', stat='median', n_iter=1000, rbins=None, 
            rlim=150, M31_data=True, all_sims=True, complete_limit=comp_limit)
        all_coadd_plot.savefig(figdir+'mock_pandas_all_hosts_10_{}_z{}.pdf'.format(np.log10(comp_limit), np.max(redshifts_01)))
        '''
        # true 3d distribution for each host and total
        m12_nsat, obs_sats = spa.cumulative_distance(m12_sat, mask_key='star.mass', bins=rbins)
        lg_nsat, obs_sats = spa.cumulative_distance(lg_sat, mask_key='star.mass', bins=rbins)
        all_nsat = {**m12_nsat, **lg_nsat}
        m31_like_dict_3d = {}
        for host in m31_like_profiles:
            m31_like_dict_3d[host] = all_nsat[host]
        m31_like_coadd_3d = mf.coadd_redshift(m31_like_dict_3d, all_sims=True)

        m12_nsat, obs_sats = spa.cumulative_distance(m12_sat, mask_key='star.mass', coadd=True, bins=rbins)
        lg_nsat, obs_sats = spa.cumulative_distance(lg_sat, mask_key='star.mass', coadd=True, bins=rbins)
        all_nsat = {**m12_nsat, **lg_nsat}

        # 3d distribution from masking in 2d
        m12_2d = sio.loop_hal(m12_sat, 'star.mass', spa.mask_on_2d_proj,
                                        **{'rbins':rbins, 'n_iter':1000, 'rlim':150,
                                            'complete_limit':comp_limit})
        lg_2d = sio.loop_hal(lg_sat, 'star.mass', spa.mask_on_2d_proj,
                                        **{'rbins':rbins, 'n_iter':1000, 'rlim':150,
                                            'complete_limit':comp_limit})
        all_coadd_2d = mf.coadd_redshift({**m12_2d, **lg_2d}, all_sims=False)
        m31_like_dict_2d = {}
        for host in m31_like_profiles:
            m31_like_dict_2d[host] = {**m12_2d, **lg_2d}[host]
        m31_like_coadd_2d = mf.coadd_redshift(m31_like_dict_2d, all_sims=True)

        # make figure combining all m31-like hosts
        plot_color = CB3_1[0]
        fig0, ax0 = plt.subplots(nrows=1, ncols=1, figsize=(7, 6))
        fig0.set_tight_layout(False)
        fig0.subplots_adjust(left=0.15, bottom=0.15)
        ax0.fill_between(rbins, m31_like_coadd_2d['percentile'][2], 
            m31_like_coadd_2d['percentile'][3], alpha=0.25, 
            color=plot_color, linewidth=0)
        ax0.fill_between(rbins, m31_like_coadd_2d['percentile'][0], 
            m31_like_coadd_2d['percentile'][1], alpha=0.25, color=plot_color, 
            linewidth=0)
        ax0.plot(rbins, m31_like_coadd_2d['median'], color=plot_color, linewidth=2, 
            label='all hosts mock survey')
        ax0.plot(rbins, m31_like_coadd_3d['median'], color=plot_color, linewidth=2, 
            linestyle='--', label='all hosts true')
        if M31_data is True:
            ax0.plot(m31_rbins1, m31_dict['median'][m31_rbins1_mask], color='k', 
                linestyle='--', label='M31')
            ax0.plot(m31_rbins2, m31_dict['median'][m31_rbins2_mask], color='k', 
                linestyle='--', alpha=0.4)
        ax0.tick_params(axis='both', which='major', labelsize=20)
        ax0.set_xlim((0,np.max(rbins)))
        ax0.set_xticks(np.arange(0,350,50), minor=False)
        ax0.set_ylim((0,30))
        ax0.set_yticks(np.arange(0,30,1), minor=True)
        ax0.set_yticks(np.arange(5,30,5), minor=False)
        #ax0.set_yticklabels(['','10','','20',''], minor=False)

        ax0.legend(loc=2, fontsize=20, title=comp_limit_title[j], borderaxespad=0.2)
        ax0.set_xlabel('Distance from host [kpc]', fontsize=24)
        ax0.set_ylabel(r'N$_{\rm sat}$($<$d)', fontsize=26)

        fig0.savefig(figdir+'mock_pandas_all_hosts_10_{}_z{}.pdf'.format(np.log10(comp_limit), np.max(m12_sat.redshift)))
