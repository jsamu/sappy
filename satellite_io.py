import numpy as np
import pandas as pd
from numba import jit
from matplotlib import pyplot as plt
import math
import utilities as ut
import halo_analysis as halo
from collections import defaultdict
from sappy import isotropic as iso
from sappy import kinematics as kin

###########################################
### load halo catalogs and merger trees ###
###########################################

def load_hals(
    sim_directories, snapshot_index, sim_names, baryon_frac=None, 
    file_kind='hdf5', host_number=1, assign_species=True):
    ''' 
    Loads simulation halo catalogs and returns a dictionary where each value
    is a list of halo catalogs at the specified redshifts for a certain
    host halo.
    '''
    if assign_species is True:
        species_ = ['star']
    hal_dict = {}
    for directory, name in zip(sim_directories, sim_names):
        if type(snapshot_index) is int:
            # may be irrelevant with array default snapshot in halo_reader
            # format a single snapshot as a list for consistency
            snap_list = [halo.io.IO.read_catalogs(snapshot_value_kind='index',
                snapshot_values=snapshot_index, simulation_directory=directory, 
                file_kind=file_kind, host_number=host_number, species=species_)]
        else:
            snapshot_index = np.array(snapshot_index).astype(int)
            snap_list = halo.io.IO.read_catalogs(snapshot_value_kind='index',
                snapshot_values=snapshot_index, simulation_directory=directory, 
                file_kind=file_kind, host_number=host_number, species=species_)
            # grab one snapshot/catalog to get indices from
            snap_list = [snap_list[snap_index] for snap_index in snapshot_index]

        # add in baryon fraction-corrected properties for DMO catalogs
        if baryon_frac is True:
            for snap_dict in snap_list:
                fb_correction = 1 - snap_dict.Cosmology['baryon.fraction']
                snap_dict['vel.circ.max.fbcorr'] = snap_dict['vel.circ.max']*np.sqrt(fb_correction)
                snap_dict['mass.fbcorr'] = snap_dict['mass']*fb_correction
                snap_dict['mass.bound.fbcorr'] = snap_dict['mass.bound']*fb_correction
                if file_kind == 'hdf5':
                    snap_dict['vel.circ.peak.fbcorr'] = snap_dict['vel.circ.peak']*np.sqrt(fb_correction)
                    snap_dict['mass.peak.fbcorr'] = snap_dict['mass.peak']*fb_correction

        hal_dict[name] = snap_list

    return hal_dict

def load_trees(
    sim_directories, sim_names, prop_subset=None, dmo=False, host_number=1,
    snapshot_ind=None, assign_species=True):
    """
    Load in merger trees using halo_analysis package.

    Parameters
    ----------
    sim_dirs : list
        List of strings specifying where the simulation data for each host is
        located. Number of directories should match number of host names, unless
        simulation is of a Local Group system (then 2 host names per directoy).
    sim_names : list
        List of strings indentifying each simulated host halo.
    prop_subset : list
        List of strings specifying properties output by halo_analysis package to be
        used in satellite analysis. If None, all properties kept in tree. If a
        list is supplied then all other properties in tree that are not listed
        in prop_subset will be deleted from the tree.
    dmo : boolean
        Flag used to correct circular velocities of dark matter only
        simulations with universal baryon fraction. If True, a new property will
        be added to the tree, 'vel.circ.max.fbcorr'.

    Returns
    -------
    tree_dict : dictionary
        Dictionary with key:value pairs corresponding to sim_name:merger_tree
        objects loaded in by the halo_analysis package. merger_tree may be modified
        if prop_subset is specified.
    """
    if assign_species is True:
        species_ = ['star']
    else:
        species_ = None
        snapshot_ind = None
    tree_dict = {}
    for directory, name in zip(sim_directories, sim_names):
        tree = halo.io.IO.read_tree(simulation_directory=directory,
                                    host_number=host_number,
                                    species=species_,
                                    species_snapshot_indices=snapshot_ind)

        if dmo is True:
            fb_correction = 1 - tree.Cosmology['baryon.fraction']
            tree['vel.circ.max.fbcorr'] = tree['vel.circ.max']*np.sqrt(fb_correction)
            tree['mass.fbcorr'] = tree['mass']*fb_correction
            tree['mass.bound.fbcorr'] = tree['mass.bound']*fb_correction

        if prop_subset is not None:
            prop_to_exclude = []
            for key in tree.keys():
                if key not in prop_subset:
                    prop_to_exclude.append(key)

            for key in prop_to_exclude:
                del tree[key]

        tree_dict[name] = tree

    return tree_dict


####################################
### mask halo catalogs and trees ###
####################################

# check if this is used anywhere and remove if not
def multiple_catalog_masks(sat, mask_names, dmo=False):
    catalog_mask_dict = {}
    for mask in mask_names:
        #catalog_mask = loop_sat(sat, mask_hosts, **{'mask_key':mask, 'dmo':dmo})
        #label_names = sat.hal_label[mask]
        #catalog_mask_dict[mask] = {'mask':catalog_mask, 'label':label_names}
        catalog_mask_dict[mask] = loop_sat(sat, mask_hosts, **{'mask_key':mask, 'dmo':dmo})
    return catalog_mask_dict

def mask_hosts(sat, hal_name, redshift_index, mask_key, dmo=False):
    '''
    Generate a mask for a single snapshot adhering to conditions given in mask_cuts_*.
    '''
    hal = sat.hal_catalog[hal_name][redshift_index]

    if dmo == False:
        snap_mask = mask_hal_baryonic(hal, sat, mask_keys=mask_key)

    elif dmo == True:
        snap_mask = mask_hal_dmo(hal, sat, hal_name, redshift_index, mask_keys=mask_key)

    return snap_mask

def mask_catalogs(sat, mask_keys, dmo=False):
    '''
    Generate a mask for each host at each snapshot.
    '''
    sat_mask_dict = {}
    for host_key in sat.hal_catalog.keys():
        host_masks = []
        for redshift_index in range(len(sat.redshift)):
            hal = sat.hal_catalog[host_key][redshift_index]
            if dmo is True:
                snap_mask = mask_hal_dmo(hal=hal, sat=sat, hal_name=host_key, 
                    redshift_index=redshift_index, mask_keys=mask_keys)
            else:
                snap_mask = mask_hal_baryonic(hal=hal, sat=sat, mask_keys=mask_keys, 
                    hal_name=host_key, redshift_index=redshift_index)
            host_masks.append(snap_mask)
        sat_mask_dict[host_key] = host_masks

    return sat_mask_dict

def mask_hosts_lite(sat_params, sat_data, mask_keys, dmo=False):
    '''
    Generate a mask for a single snapshot adhering to conditions given in mask_cuts_*.
    '''
    sat_mask_dict = {}
    for host_key in sat_data.keys():
        host_masks = []
        for redshift_index in range(len(sat_params.redshift)):
            hal = sat_data[host_key][redshift_index]
            if dmo is True:
                snap_mask = mask_hal_dmo(hal=hal, sat=sat_params, hal_name=host_key, redshift_index=redshift_index, mask_keys=mask_keys)
            else:
                snap_mask = mask_hal_baryonic(hal=hal, sat=sat_params, mask_keys=mask_keys)
            host_masks.append(snap_mask)
        sat_mask_dict[host_key] = host_masks

    return sat_mask_dict

def mask_hal_baryonic(hal, sat, mask_keys, hal_name=None, redshift_index=None):
    '''
    Create masks on different properties for the baryonic halo catalogs.
    '''
    # systematically exclude the host/main halo
    host_mask = np.ones(len(hal.prop('host.index')), dtype=bool)
    host_mask[hal.prop('host.index')[0]] = False

    mask_lowres_fraction = hal.prop('lowres.mass.frac') <= sat.low_res

    # exclude halos with bound fractions less than 0.4 (default)
    bound_mask = hal.prop('mass.bound/mass') > sat.bound_frac

    mask_distance = ((hal.prop('host.distance.total') >= sat.r_range[0]) & 
                    (hal.prop('host.distance.total') <= sat.r_range[1]))

    mask_star_density = hal.prop('star.density.50') >= sat.star_density

    snapshot_mask_dict = {}
    for mask_key in mask_keys:
        if mask_key == 'star.mass':
            mask_star_mass_low = hal.prop('star.mass') >= sat.star_mass[0]
            mask_star_mass_high = hal.prop('star.mass') <= sat.star_mass[1]
            combined_mask = (host_mask & mask_star_mass_low & mask_star_mass_high & 
                            mask_star_density & mask_lowres_fraction & mask_distance & bound_mask)

        elif mask_key == 'star.number':
            mask_star_number = hal.prop('star.number') >= sat.star_number
            combined_mask = (host_mask & mask_star_number & mask_star_density & 
                            mask_lowres_fraction & mask_distance & bound_mask)

        elif mask_key == 'most.star.mass':
            #choosing top N most massive sats
            nan_mask = ~np.isnan(hal['star.mass'])
            base_mask = host_mask & mask_lowres_fraction & mask_distance & bound_mask & mask_star_density & nan_mask
            base_ind = np.where(base_mask)[0]
            base_sm = hal['star.mass'][base_ind]
            top_n_base_sm_ind = np.argsort(base_sm)[-sat.abs_number:]
            top_n_ind = base_ind[top_n_base_sm_ind]
            combined_mask = np.zeros(len(hal['star.mass']), dtype=bool)
            combined_mask[top_n_ind] = True

        elif mask_key == 'disk.star.mass':
            mask_star_mass_low = hal.prop('star.mass') >= sat.star_mass[0]
            mask_star_mass_high = hal.prop('star.mass') <= sat.star_mass[1]
            disk_mask = select_out_of_disk(hal['host.distance'], sat.host_disk_axes, hal_name, 
                sat.snapshot[redshift_index], disk_mask_angle=sat.disk_mask_angle)
            combined_mask = (host_mask & mask_star_mass_low & mask_star_mass_high & 
                mask_star_density & mask_lowres_fraction & mask_distance & bound_mask &
                disk_mask)

        elif mask_key == 'disk.most.star.mass':
            #choosing top N most massive sats
            nan_mask = ~np.isnan(hal['star.mass'])
            disk_mask = select_out_of_disk(hal['host.distance'], sat.host_disk_axes, hal_name, 
                sat.snapshot[redshift_index], disk_mask_angle=sat.disk_mask_angle)
            base_mask = (host_mask & mask_lowres_fraction & mask_distance & 
                bound_mask & mask_star_density & disk_mask & nan_mask)
            base_ind = np.where(base_mask)[0]
            base_sm = hal['star.mass'][base_ind]
            top_n_base_sm_ind = np.argsort(base_sm)[-sat.abs_number:]
            top_n_ind = base_ind[top_n_base_sm_ind]
            combined_mask = np.zeros(len(hal['star.mass']), dtype=bool)
            combined_mask[top_n_ind] = True

        elif mask_key == 'most.massive':
            #choosing top N most massive sats
            nan_mask = ~np.isnan(hal['mass'])
            base_mask = host_mask & mask_lowres_fraction & mask_distance & bound_mask & nan_mask
            base_ind = np.where(base_mask)[0]
            base_sm = hal['mass'][base_ind]
            top_n_base_sm_ind = np.argsort(base_sm)[-sat.abs_number:]
            top_n_ind = base_ind[top_n_base_sm_ind]
            combined_mask = np.zeros(len(hal['mass']), dtype=bool)
            combined_mask[top_n_ind] = True

        elif mask_key == 'mass.peak':
            base_mask = host_mask & mask_lowres_fraction & mask_distance & bound_mask
            base_ind = np.where(base_mask)[0]
            m_peak_mask = hal['mass.peak'][base_ind] > sat.mass_peak
            m_peak_ind = base_ind[m_peak_mask]
            combined_mask = np.zeros(len(hal['mass']), dtype=bool)
            combined_mask[m_peak_ind] = True

        elif mask_key == 'most.mass.peak':
            nan_mask = ~np.isnan(hal['mass.peak'])
            base_mask = host_mask & mask_lowres_fraction & mask_distance & bound_mask & nan_mask
            base_ind = np.where(base_mask)[0]
            base_sm = hal['mass.peak'][base_ind]
            top_n_base_sm_ind = np.argsort(base_sm)[-sat.abs_number:]
            top_n_ind = base_ind[top_n_base_sm_ind]
            combined_mask = np.zeros(len(hal['mass.peak']), dtype=bool)
            combined_mask[top_n_ind] = True

        elif mask_key == 'mass.bound':
            base_mask = host_mask & mask_distance & mask_lowres_fraction & bound_mask
            bound_mask_ = hal.prop('mass.bound') > sat.m_bound
            combined_mask = base_mask & bound_mask_

        elif mask_key == 'vel.circ.max':
            mask_v_circ = hal.prop('vel.circ.max') >= sat.vel_circ_max
            combined_mask = (host_mask & mask_lowres_fraction & mask_distance &
                            mask_v_circ & bound_mask)

        elif mask_key == 'most.vel.circ.max':
            nan_mask = ~np.isnan(hal['vel.circ.max'])
            base_mask = host_mask & mask_lowres_fraction & mask_distance & bound_mask & nan_mask
            base_ind = np.where(base_mask)[0]
            base_sm = hal['vel.circ.max'][base_ind]
            top_n_base_sm_ind = np.argsort(base_sm)[-sat.abs_number:]
            top_n_ind = base_ind[top_n_base_sm_ind]
            combined_mask = np.zeros(len(hal['vel.circ.max']), dtype=bool)
            combined_mask[top_n_ind] = True

        elif mask_key == 'v.peak':
            base_mask = host_mask & mask_lowres_fraction & mask_distance & bound_mask
            v_peak_mask = hal['vel.circ.peak'] > sat.v_peak
            combined_mask = base_mask & v_peak_mask

        elif mask_key == 'most.v.peak':
            nan_mask = ~np.isnan(hal['vel.circ.peak'])
            base_mask = host_mask & mask_lowres_fraction & mask_distance & bound_mask & nan_mask
            base_ind = np.where(base_mask)[0]
            base_sm = hal['vel.circ.peak'][base_ind]
            top_n_base_sm_ind = np.argsort(base_sm)[-sat.abs_number:]
            top_n_ind = base_ind[top_n_base_sm_ind]
            combined_mask = np.zeros(len(hal['vel.circ.peak']), dtype=bool)
            combined_mask[top_n_ind] = True

        else:
            raise ValueError('Halo catalog mask key ({}) not recognized'.format(mask_key))
        
        snapshot_mask_dict[mask_key] = combined_mask

    return snapshot_mask_dict

def mask_hal_dmo(hal, sat, hal_name, redshift_index, mask_keys):
    '''
    Create masks on different properties for the DMO halo catalogs.
    '''
    # systematically exclude the host/main halo
    host_mask = np.ones(len(hal.prop('host.index')), dtype=bool)
    host_mask[hal.prop('host.index')[0]] = False

    mask_distance = ((hal.prop('host.distance.total') >= sat.r_range[0]) & 
                    (hal.prop('host.distance.total') <= sat.r_range[1]))

    #mask_lowres_fraction = hal.prop('lowres.dark.mass.frac') <= sat.low_res
    mask_lowres_fraction = hal['mass.lowres']/hal['mass'] <= sat.low_res

    # exclude halos with bound fractions less than 0.4
    bound_mask = hal.prop('mass.bound/mass') > 0.4

    snapshot_mask_dict = {}
    for mask_key in mask_keys:
        if mask_key == 'number.sats':
            # number of satellites from baryonic run to be matched
            n_sats = sat.num_sats[hal_name][redshift_index]

            # basic mask excluding host, within radius, and below 2% of mass in low resolution DM
            base_mask = host_mask & mask_distance & mask_lowres_fraction & bound_mask
            base_ind = np.where(base_mask)[0]

            # select 'mass.fbcorr' according to base mask
            base_mmax = hal['mass.fbcorr'][base_ind]

            # choose n_sat highest mass.fbcorr from those satisfying base_mask
            top_n_base_mmax_ind = np.argsort(base_mmax)[-n_sats:]#indices in base_mmax
            top_n_ind = base_ind[top_n_base_mmax_ind]#indices in halo catalog

            # make new mask selecting halos satisfying all criteria 
            combined_mask = np.zeros(len(hal['mass.fbcorr']), dtype=bool)
            combined_mask[top_n_ind] = True

        elif mask_key == 'median.vel.circ.max':
            v_mask_med = hal['vel.circ.max.fbcorr'] >= sat.med_v_circ[redshift_index]
            combined_mask = host_mask & mask_distance & mask_lowres_fraction & v_mask_med & bound_mask

        elif mask_key == 'vel.circ.max':
            v_mask_const = hal['vel.circ.max.fbcorr'] >= sat.vel_circ_max
            combined_mask = host_mask & mask_distance & mask_lowres_fraction & v_mask_const & bound_mask

        elif mask_key == 'most.vel.circ.max':
            nan_mask = ~np.isnan(hal['vel.circ.max.fbcorr'])
            base_mask = host_mask & mask_distance & mask_lowres_fraction & bound_mask & nan_mask
            base_ind = np.where(base_mask)[0]
            base_vmax = hal['vel.circ.max.fbcorr'][base_ind]
            top_n_base_vmax_ind = np.argsort(base_vmax)[-sat.abs_number:]
            top_n_ind = base_ind[top_n_base_vmax_ind]
            combined_mask = np.zeros(len(hal['vel.circ.max.fbcorr']), dtype=bool)
            combined_mask[top_n_ind] = True

        elif mask_key == 'v.peak':
            base_mask = host_mask & mask_distance & mask_lowres_fraction & bound_mask
            v_peak_mask = hal['vel.circ.max.fbcorr'] > sat.v_peak
            combined_mask = base_mask & v_peak_mask

        elif mask_key == 'most.v.peak':
            nan_mask = ~np.isnan(hal['vel.circ.max.fbcorr'])
            base_mask = host_mask & mask_distance & mask_lowres_fraction & bound_mask & nan_mask
            base_ind = np.where(base_mask)[0]
            base_vmax = hal['vel.circ.peak.fbcorr'][base_ind]
            top_n_base_vmax_ind = np.argsort(base_vmax)[-sat.abs_number:]
            top_n_ind = base_ind[top_n_base_vmax_ind]
            combined_mask = np.zeros(len(hal['vel.circ.peak.fbcorr']), dtype=bool)
            combined_mask[top_n_ind] = True

        elif mask_key == 'most.massive':
            nan_mask = ~np.isnan(hal['mass.fbcorr'])
            base_mask = host_mask & mask_distance & mask_lowres_fraction & bound_mask & nan_mask
            base_ind = np.where(base_mask)[0]
            base_vmax = hal['mass.fbcorr'][base_ind]
            top_n_base_vmax_ind = np.argsort(base_vmax)[-sat.abs_number:]
            top_n_ind = base_ind[top_n_base_vmax_ind]
            combined_mask = np.zeros(len(hal['mass.fbcorr']), dtype=bool)
            combined_mask[top_n_ind] = True

        elif mask_key == 'mass.peak':
            base_mask = host_mask & mask_distance & mask_lowres_fraction & bound_mask
            m_peak_mask = hal['mass.peak.fbcorr'] > sat.mass_peak
            combined_mask = base_mask & m_peak_mask

        elif mask_key == 'most.mass.peak':
            nan_mask = ~np.isnan(hal['mass.peak.fbcorr'])
            base_mask = host_mask & mask_distance & mask_lowres_fraction & bound_mask & nan_mask
            base_ind = np.where(base_mask)[0]
            base_vmax = hal['mass.peak.fbcorr'][base_ind]
            top_n_base_vmax_ind = np.argsort(base_vmax)[-sat.abs_number:]
            top_n_ind = base_ind[top_n_base_vmax_ind]
            combined_mask = np.zeros(len(hal['mass.peak.fbcorr']), dtype=bool)
            combined_mask[top_n_ind] = True

        elif mask_key == 'mass.bound':
            base_mask = host_mask & mask_distance & mask_lowres_fraction & bound_mask
            bound_mask_ = hal.prop('mass.bound.fbcorr') > sat.m_bound
            combined_mask = base_mask & bound_mask_

        else:
            raise ValueError('Halo catalog mask key ({}) not recognized'.format(mask_key))

        snapshot_mask_dict[mask_key] = combined_mask

    return snapshot_mask_dict

def mask_lg_dmo_cat(sat):
    """
    Applies masks on dmo halo properties, values for masking are set in the
    SatParam class in halo_reader.py.
    """
    
    pair_mask_dict = {}
    for pair_name in sat.hal_catalog.keys():
        host_mask_dict = defaultdict(list)
        host_list = ['host', 'host2']
        for host_name, host_str in zip(sat.hal_name[pair_name], host_list):
            for redshift_index, hal in enumerate(sat.hal_catalog[pair_name]):
                # choose only satellites of a single host galaxy
                host_ind = np.unique(hal[host_str+'.index'])[0]
                host_system_mask = hal[host_str+'.index'] == host_ind

                # systematically exclude the host/main halo
                host_mask = ~(hal[host_str+'.index'] == np.arange(len(hal[host_str+'.index'])))

                # exclude halos contaminated by low resolution dark matter
                lowres_mask = hal['mass.lowres']/hal['mass'] <= sat.low_res

                # exclude halos with bound fractions less than 0.4
                bound_mask = hal.prop('mass.bound/mass') > 0.4

                # select subhalos within a certain distance to be satellites
                dist_3d = hal[host_str+'.distance']
                total_distance = np.sqrt(dist_3d[:,0]**2 + dist_3d[:,1]**2 + dist_3d[:,2]**2)
                distance_mask = ((total_distance >= sat.r_range[0]) & 
                                (total_distance <= sat.r_range[1]))
                
                # basic mask excluding host, within radius, and below 2% of mass in low resolution DM
                base_mask = host_system_mask & distance_mask & lowres_mask & bound_mask
                base_ind = np.where(base_mask)[0]

                redshift_mask_dict = {}
                for dmo_key in sat.mask_names:
                    if dmo_key == 'number.sats':
                        # number of satellites from baryonic run to be matched
                        n_sats = sat.num_sats[host_name][redshift_index]

                        # select 'mass.fbcorr' according to base mask
                        base_mmax = hal['mass.fbcorr'][base_ind]

                        # choose n_sat highest vel.circ.max.fbcorr from those satisfying base_mask
                        top_n_base_mmax_ind = np.argsort(base_mmax)[-n_sats:]#indices in base_mmax
                        top_n_ind = base_ind[top_n_base_mmax_ind]#indices in halo catalog

                        # make new mask selecting halos satisfying all criteria 
                        combined_mask = np.zeros(len(hal['mass.fbcorr']), dtype=bool)
                        combined_mask[top_n_ind] = True

                    elif dmo_key == 'median.vel.circ.max':
                        v_mask_med = hal['vel.circ.max.fbcorr'] >= sat.med_v_circ[redshift_index]
                        combined_mask = base_mask & v_mask_med

                    elif dmo_key == 'vel.circ.max':
                        v_mask_const = hal['vel.circ.max.fbcorr'] >= sat.vel_circ_max
                        combined_mask = base_mask & v_mask_const

                    elif dmo_key == 'most.vel.circ.max':
                        nan_mask = ~np.isnan(hal['vel.circ.max.fbcorr'])
                        base_mask = base_mask & nan_mask
                        base_ind = np.where(base_mask)[0]
                        base_vmax = hal['vel.circ.max.fbcorr'][base_ind]
                        top_n_base_vmax_ind = np.argsort(base_vmax)[-sat.abs_number:]
                        top_n_ind = base_ind[top_n_base_vmax_ind]
                        combined_mask = np.zeros(len(hal['vel.circ.max.fbcorr']), dtype=bool)
                        combined_mask[top_n_ind] = True

                    elif dmo_key == 'v.peak':
                        nan_mask = ~np.isnan(hal['vel.circ.peak.fbcorr'])
                        base_mask = base_mask & nan_mask
                        v_peak_mask = hal['vel.circ.peak.fbcorr'] > sat.v_peak
                        combined_mask = base_mask & v_peak_mask

                    elif dmo_key == 'most.v.peak':
                        nan_mask = ~np.isnan(hal['vel.circ.peak.fbcorr'])
                        base_mask = base_mask & nan_mask
                        base_ind = np.where(base_mask)[0]
                        base_vmax = hal['vel.circ.peak.fbcorr'][base_ind]
                        top_n_base_vmax_ind = np.argsort(base_vmax)[-sat.abs_number:]
                        top_n_ind = base_ind[top_n_base_vmax_ind]
                        combined_mask = np.zeros(len(hal['vel.circ.peak.fbcorr']), dtype=bool)
                        combined_mask[top_n_ind] = True

                    elif dmo_key == 'most.massive':
                        nan_mask = ~np.isnan(hal['mass.fbcorr'])
                        base_mask = base_mask & nan_mask
                        base_ind = np.where(base_mask)[0]
                        base_vmax = hal['mass.fbcorr'][base_ind]
                        top_n_base_vmax_ind = np.argsort(base_vmax)[-sat.abs_number:]
                        top_n_ind = base_ind[top_n_base_vmax_ind]
                        combined_mask = np.zeros(len(hal['mass.fbcorr']), dtype=bool)
                        combined_mask[top_n_ind] = True

                    elif dmo_key == 'mass.peak':
                        m_peak_mask = hal['mass.peak.fbcorr'][base_ind] > sat.mass_peak
                        m_peak_ind = base_ind[m_peak_mask]
                        combined_mask = np.zeros(len(hal['mass.peak.fbcorr']), dtype=bool)
                        combined_mask[m_peak_ind] = True

                    elif dmo_key == 'most.mass.peak':
                        nan_mask = ~np.isnan(hal['mass.peak.fbcorr'])
                        base_mask = base_mask & nan_mask
                        base_ind = np.where(base_mask)[0]
                        base_vmax = hal['mass.peak.fbcorr'][base_ind]
                        top_n_base_vmax_ind = np.argsort(base_vmax)[-sat.abs_number:]
                        top_n_ind = base_ind[top_n_base_vmax_ind]
                        combined_mask = np.zeros(len(hal['mass.peak.fbcorr']), dtype=bool)
                        combined_mask[top_n_ind] = True

                    redshift_mask_dict[dmo_key] = combined_mask
                host_mask_dict[host_name].append(redshift_mask_dict)

        pair_mask_dict[pair_name] = host_mask_dict

    return pair_mask_dict

def mask_lg_baryon_cat(sat):
    """
    Applies masks on baryonic halo properties, values for masking are set in the
    SatParam class in halo_reader.py.

    Returns
    -------
    tree_mask_dict : dictionary
        Dictionary of dictionaries of tree masks where key:value pairs
        correspond to sim_name:{mask_type:mask}.
    """
    
    pair_mask_dict = {}
    for pair_name in sat.hal_catalog.keys():
        host_mask_dict = defaultdict(list)
        host_list = ['host', 'host2']
        for host_name, host_str in zip(sat.hal_name[pair_name], host_list):
            for i,hal in enumerate(sat.hal_catalog[pair_name]):
                # choose only satellites of a single host galaxy
                host_ind = np.unique(hal[host_str+'.index'])[0]
                host_system_mask = hal[host_str+'.index'] == host_ind

                # systematically exclude the host/main halo
                host_mask = ~(hal[host_str+'.index'] == np.arange(len(hal[host_str+'.index'])))

                # exclude halos contaminated by low resolution dark matter
                lowres_mask = hal['mass.lowres']/hal['mass'] <= sat.low_res

                # exclude halos with bound fractions less than 0.4
                bound_mask = hal.prop('mass.bound/mass') > 0.4

                # select subhalos within a certain distance to be satellites
                dist_3d = hal[host_str+'.distance']
                total_distance = np.sqrt(dist_3d[:,0]**2 + dist_3d[:,1]**2 + dist_3d[:,2]**2)
                distance_mask = ((total_distance >= sat.r_range[0]) & 
                                (total_distance <= sat.r_range[1]))
                
                mask_star_density = hal.prop('star.density.50') >= sat.star_density

                redshift_mask_dict = {}

                for mask_key in sat.mask_names:
                    if mask_key == 'star.mass':
                        base_mask = host_mask & mask_star_density & lowres_mask & distance_mask & bound_mask
                        base_ind = np.where(base_mask)[0]
                        mask_star_mass_low = hal.prop('star.mass') >= sat.star_mass[0]
                        mask_star_mass_high = hal.prop('star.mass') <= sat.star_mass[1]
                        combined_mask = base_mask & mask_star_mass_low & mask_star_mass_high

                    elif mask_key == 'most.star.mass':
                        base_mask = host_mask & mask_star_density & lowres_mask & distance_mask & bound_mask
                        base_ind = np.where(base_mask)[0]
                        base_vmax = hal['star.mass'][base_ind]
                        top_n_base_vmax_ind = np.argsort(base_vmax)[-sat.abs_number:]
                        top_n_ind = base_ind[top_n_base_vmax_ind]
                        combined_mask = np.zeros(len(hal['star.mass']), dtype=bool)
                        combined_mask[top_n_ind] = True

                    elif mask_key == 'disk.star.mass':
                        mask_star_mass_low = hal.prop('star.mass') >= sat.star_mass[0]
                        mask_star_mass_high = hal.prop('star.mass') <= sat.star_mass[1]
                        disk_mask = select_out_of_disk(hal[host_str+'.distance'], sat.host_disk_axes, host_name, 
                            sat.snapshot[i], disk_mask_angle=sat.disk_mask_angle)
                        combined_mask = (host_mask & mask_star_mass_low & mask_star_mass_high & 
                            mask_star_density & lowres_mask & distance_mask & bound_mask &
                            disk_mask)

                    elif mask_key == 'disk.most.star.mass':
                        #choosing top N most massive sats
                        disk_mask = select_out_of_disk(hal[host_str+'.distance'], sat.host_disk_axes, host_name, 
                            sat.snapshot[i], disk_mask_angle=sat.disk_mask_angle)
                        base_mask = (host_mask & lowres_mask & distance_mask & 
                            bound_mask & mask_star_density & disk_mask)
                        base_ind = np.where(base_mask)[0]
                        base_sm = hal['star.mass'][base_ind]
                        top_n_base_sm_ind = np.argsort(base_sm)[-sat.abs_number:]
                        top_n_ind = base_ind[top_n_base_sm_ind]
                        combined_mask = np.zeros(len(hal['star.mass']), dtype=bool)
                        combined_mask[top_n_ind] = True

                    elif mask_key == 'most.massive':
                        nan_mask = ~np.isnan(hal['mass'])
                        base_mask = host_mask & lowres_mask & distance_mask & bound_mask & nan_mask
                        base_ind = np.where(base_mask)[0]
                        base_vmax = hal['mass'][base_ind]
                        top_n_base_vmax_ind = np.argsort(base_vmax)[-sat.abs_number:]
                        top_n_ind = base_ind[top_n_base_vmax_ind]
                        combined_mask = np.zeros(len(hal['mass']), dtype=bool)
                        combined_mask[top_n_ind] = True

                    elif mask_key == 'mass.peak':
                        base_mask = host_mask & lowres_mask & distance_mask & bound_mask
                        base_ind = np.where(base_mask)[0]
                        m_peak_mask = hal['mass.peak'][base_ind] > sat.mass_peak
                        m_peak_ind = base_ind[m_peak_mask]
                        combined_mask = np.zeros(len(hal['mass.peak']), dtype=bool)
                        combined_mask[m_peak_ind] = True

                    elif mask_key == 'most.mass.peak':
                        nan_mask = ~np.isnan(hal['mass.peak'])
                        base_mask = host_mask & lowres_mask & distance_mask & bound_mask & nan_mask
                        base_ind = np.where(base_mask)[0]
                        base_vmax = hal['mass.peak'][base_ind]
                        top_n_base_vmax_ind = np.argsort(base_vmax)[-sat.abs_number:]
                        top_n_ind = base_ind[top_n_base_vmax_ind]
                        combined_mask = np.zeros(len(hal['mass.peak']), dtype=bool)
                        combined_mask[top_n_ind] = True

                    elif mask_key == 'vel.circ.max':
                        base_mask = host_mask & lowres_mask & distance_mask & bound_mask
                        base_ind = np.where(base_mask)[0]
                        v_mask_const = hal['vel.circ.max'] >= sat.vel_circ_max
                        combined_mask = base_mask & v_mask_const

                    elif mask_key == 'most.vel.circ.max':
                        nan_mask = ~np.isnan(hal['vel.circ.max'])
                        base_mask = host_mask & lowres_mask & distance_mask & bound_mask & nan_mask
                        base_ind = np.where(base_mask)[0]
                        base_vmax = hal['vel.circ.max'][base_ind]
                        top_n_base_vmax_ind = np.argsort(base_vmax)[-sat.abs_number:]
                        top_n_ind = base_ind[top_n_base_vmax_ind]
                        combined_mask = np.zeros(len(hal['vel.circ.max']), dtype=bool)
                        combined_mask[top_n_ind] = True

                    elif mask_key == 'most.v.peak':
                        nan_mask = ~np.isnan(hal['vel.circ.peak'])
                        base_mask = host_mask & lowres_mask & distance_mask & bound_mask & nan_mask
                        base_ind = np.where(base_mask)[0]
                        base_vmax = hal['vel.circ.peak'][base_ind]
                        top_n_base_vmax_ind = np.argsort(base_vmax)[-sat.abs_number:]
                        top_n_ind = base_ind[top_n_base_vmax_ind]
                        combined_mask = np.zeros(len(hal['vel.circ.peak']), dtype=bool)
                        combined_mask[top_n_ind] = True

                    else:
                        print('no mask available')

                    redshift_mask_dict[mask_key] = combined_mask
                host_mask_dict[host_name].append(redshift_mask_dict)

        pair_mask_dict[pair_name] = host_mask_dict

    return pair_mask_dict

def mask_tree(sat):
    """
    Applies masks on baryonic halo properties, values for masking are set in the
    SatParam class in halo_reader.py.

    Returns
    -------
    tree_mask_dict : dictionary
        Dictionary of dictionaries of tree masks where key:value pairs
        correspond to sim_name:{mask_type:mask}.
    """
    tree_mask_dict = {}
    for host_key in sat.tree.keys():
        # check for iterable, if not make iterable
        # (load hals needs a non-iterable, need to fix this inconsistency in a better way)
        # also exits in loop hal
        snapshot_ = sat.snapshots_to_mask
        if type(snapshot_) is int:
            snapshot_ = [snapshot_]
        for snap in snapshot_:
            # look at a single snapshot/redshift
            #redshift_snap_id = sat.tree[host_key].Snapshot.get_snapshot_indices('redshift', redshift)
            redshift_mask = sat.tree[host_key]['snapshot'] == snap

            # systematically exclude the host/main halo
            host_mask = ~(sat.tree[host_key]['host.index'] == np.arange(len(sat.tree[host_key]['host.index'])))

            # exclude phantom halos generated by rockstar halo finder
            phantom_mask = ~(sat.tree[host_key]['am.phantom'] == 1)

            # exclude halos contaminated by low resolution dark matter
            lowres_mask = sat.tree[host_key]['mass.lowres']/sat.tree[host_key]['mass'] <= sat.low_res

            # exclude halos with bound fractions less than 0.4
            bound_mask = sat.tree[host_key].prop('mass.bound/mass') > 0.4

            # select subhalos within a certain distance to be satellites
            total_distance = sat.tree[host_key].prop('host.distance.total')
            distance_mask = ((total_distance >= sat.r_range[0]) & 
                            (total_distance <= sat.r_range[1]))

            redshift_mask_dict = {}
            for star_key in sat.mask_names:
                if star_key == 'star.mass':
                    # for star particle data make a cut on half stellar density
                    star_density_mask = sat.tree[host_key].prop('star.density.50') >= sat.star_density
                    star_mass_low_mask = sat.tree[host_key]['star.mass'] >= sat.star_mass[0]
                    star_mass_high_mask = sat.tree[host_key]['star.mass'] <= sat.star_mass[1]
                    combined_mask = (redshift_mask & host_mask & star_mass_low_mask
                                    & star_mass_high_mask & star_density_mask & lowres_mask
                                    & distance_mask & phantom_mask & bound_mask)

                elif star_key == 'most.star.mass':
                    # for star particle data make a cut on half stellar density
                    star_density_mask = sat.tree[host_key].prop('star.density.50') >= sat.star_density
                    #choosing top 11 most massive sats
                    base_mask = (redshift_mask & host_mask & star_density_mask 
                                    & lowres_mask & distance_mask & phantom_mask & bound_mask)
                    base_ind = np.where(base_mask)[0]
                    base_sm = sat.tree[host_key]['star.mass'][base_ind]
                    top_n_base_sm_ind = np.argsort(base_sm)[-sat.abs_number:]
                    top_n_ind = base_ind[top_n_base_sm_ind]
                    combined_mask = np.zeros(len(sat.tree[host_key]['star.mass']), dtype=bool)
                    combined_mask[top_n_ind] = True

                elif star_key == 'star.number':
                    # for star particle data make a cut on half stellar density
                    #star_density_mask = sat.tree[host_key].prop('star.density.50') >= sat.star_density
                    star_number_mask = sat.tree[host_key]['star.number'] >= sat.star_number
                    combined_mask = (redshift_mask & host_mask & star_number_mask &#star_density_mask &  
                                    lowres_mask & distance_mask& phantom_mask & bound_mask)

                elif star_key == 'lmc.star.number':
                    # for star particle data make a cut on half stellar density
                    star_number_mask = sat.tree[host_key]['star.number'] >= sat.star_number
                    base_mask = (redshift_mask & host_mask & star_number_mask & 
                                lowres_mask & distance_mask & phantom_mask & bound_mask)
                    base_ind = np.where(base_mask)[0]
                    # code from Megan Barry 9/2021
                    progenitor_indices = sat.tree[host_key].prop('progenitor.main.indices',base_ind)
                    progenitor_indices[progenitor_indices < 0] = -1
                    central_indices = sat.tree[host_key]['central.index'][progenitor_indices]
                    LMC_indices = sat.tree[host_key].prop('progenitor.main.indices', sat.lmc_index)
                    LMC_tracker = np.full_like(progenitor_indices,0)
                    dummy_tracker = np.full_like(progenitor_indices,1)
                    for LMC_index in LMC_indices:
                        #returns a 1 if the subhalo in central_indices was ever a satellite of an LMC, 0 if not
                        LMC_tracker = np.where(central_indices == LMC_index,dummy_tracker,LMC_tracker)
                    result = np.sum(LMC_tracker,axis=1)
                    LMC_mask = result > 0

                    lmc_hosted_ind = base_ind[LMC_mask]
                    combined_mask = np.zeros(len(sat.tree[host_key]['mass']), dtype=bool)
                    combined_mask[lmc_hosted_ind] = True

                elif star_key == 'no.lmc.star.number':
                    # for star particle data make a cut on half stellar density
                    star_number_mask = sat.tree[host_key]['star.number'] >= sat.star_number
                    base_mask = (redshift_mask & host_mask & star_number_mask & 
                                lowres_mask & distance_mask & phantom_mask & bound_mask)
                    base_ind = np.where(base_mask)[0]
                    # code from Megan Barry 9/2021
                    progenitor_indices = sat.tree[host_key].prop('progenitor.main.indices',base_ind)
                    progenitor_indices[progenitor_indices < 0] = -1
                    central_indices = sat.tree[host_key]['central.index'][progenitor_indices]
                    LMC_indices = sat.tree[host_key].prop('progenitor.main.indices', sat.lmc_index)
                    LMC_tracker = np.full_like(progenitor_indices,0)
                    dummy_tracker = np.full_like(progenitor_indices,1)
                    for LMC_index in LMC_indices:
                        #returns a 1 if the subhalo in central_indices was ever a satellite of an LMC, 0 if not
                        LMC_tracker = np.where(central_indices == LMC_index,dummy_tracker,LMC_tracker)
                    result = np.sum(LMC_tracker,axis=1)
                    LMC_mask = result > 0

                    lmc_hosted_ind = base_ind[~LMC_mask]
                    combined_mask = np.zeros(len(sat.tree[host_key]['mass']), dtype=bool)
                    combined_mask[lmc_hosted_ind] = True

                elif star_key == 'vel.circ.max':
                    v_circ_mask = sat.tree[host_key]['vel.circ.max'] >= sat.vel_circ_max
                    combined_mask = (redshift_mask & host_mask & lowres_mask &
                                    distance_mask & v_circ_mask & phantom_mask & bound_mask)

                elif star_key == 'v.peak':
                    base_mask = (redshift_mask & host_mask & 
                                 lowres_mask & distance_mask & phantom_mask & 
                                 bound_mask)
                    base_ind = np.where(base_mask)[0]
                    v_peak_arr = kin.v_peak(sat.tree[host_key], hal_mask=base_mask)
                    v_peak_mask = v_peak_arr > sat.v_peak
                    v_peak_ind = base_ind[v_peak_mask]
                    combined_mask = np.zeros(len(sat.tree[host_key]['vel.circ.max']), dtype=bool)
                    combined_mask[v_peak_ind] = True

                elif star_key == 'mass.peak':
                    base_mask = (redshift_mask & host_mask & 
                                 lowres_mask & distance_mask & phantom_mask & 
                                 bound_mask)
                    base_ind = np.where(base_mask)[0]
                    m_peak_arr = kin.v_peak(sat.tree[host_key], hal_mask=base_mask, hal_property='mass')
                    m_peak_mask = m_peak_arr > sat.mass_peak
                    m_peak_ind = base_ind[m_peak_mask]
                    combined_mask = np.zeros(len(sat.tree[host_key]['mass']), dtype=bool)
                    combined_mask[m_peak_ind] = True

                elif star_key == 'mass.bound':
                    bound_mask_ = sat.tree[host_key].prop('mass.bound') > sat.m_bound
                    combined_mask = (redshift_mask & host_mask & 
                                 lowres_mask & distance_mask & phantom_mask & 
                                 bound_mask & bound_mask_)

                elif star_key == 'completeness':
                    star_density_mask = sat.tree[host_key].prop('star.density.50') >= sat.star_density
                    total_distance = sat.tree[host_key].prop('host.distance.total')
                    star_mass_high_mask = sat.tree[host_key]['star.mass'] <= sat.star_mass[1]

                    d_1_mask = ((total_distance >= sat.r_range[0]) & 
                                    (total_distance <= sat.d_complete[0])) 
                    sm_low_mask1 = sat.tree[host_key]['star.mass'] >= sat.sm_complete[0]
                    combined_mask1 = (redshift_mask & host_mask & sm_low_mask1
                                    & star_mass_high_mask & star_density_mask & lowres_mask
                                    & d_1_mask & phantom_mask & bound_mask)

                    sm_low_mask2 = sat.tree[host_key]['star.mass'] >= sat.sm_complete[1]
                    d_2_mask = ((total_distance >= sat.r_range[0]) & 
                                    (total_distance <= sat.d_complete[1])) 
                    combined_mask2 = (redshift_mask & host_mask & sm_low_mask2
                                    & star_mass_high_mask & star_density_mask & lowres_mask
                                    & d_2_mask & phantom_mask & bound_mask)

                    combined_mask = combined_mask1 | combined_mask2

                elif star_key == 'lmc.mass.peak':
                    # select those halos that were previously hosted by an LMC analogue
                    base_mask = (redshift_mask & host_mask & 
                                 lowres_mask & distance_mask & phantom_mask & 
                                 bound_mask)
                    base_ind = np.where(base_mask)[0]
                    m_peak_arr = kin.v_peak(sat.tree[host_key], hal_mask=base_mask, hal_property='mass')
                    m_peak_mask = m_peak_arr > sat.mass_peak
                    m_peak_ind = base_ind[m_peak_mask]

                    # code from Megan Barry 9/2021
                    progenitor_indices = sat.tree[host_key].prop('progenitor.main.indices',m_peak_ind)
                    progenitor_indices[progenitor_indices < 0] = -1
                    central_indices = sat.tree[host_key]['central.index'][progenitor_indices]
                    LMC_indices = sat.tree[host_key].prop('progenitor.main.indices', sat.lmc_index)
                    LMC_tracker = np.full_like(progenitor_indices,0)
                    dummy_tracker = np.full_like(progenitor_indices,1)
                    for LMC_index in LMC_indices:
                        #returns a 1 if the subhalo in central_indices was ever a satellite of an LMC, 0 if not
                        LMC_tracker = np.where(central_indices == LMC_index,dummy_tracker,LMC_tracker)
                    result = np.sum(LMC_tracker,axis=1)
                    LMC_mask = result > 0

                    lmc_hosted_ind = m_peak_ind[LMC_mask]
                    combined_mask = np.zeros(len(sat.tree[host_key]['mass']), dtype=bool)
                    combined_mask[lmc_hosted_ind] = True

                elif star_key == 'no.lmc.mass.peak':
                    # select those halos that were NOT previously hosted by an LMC analogue
                    base_mask = (redshift_mask & host_mask & 
                                lowres_mask & distance_mask & phantom_mask & 
                                bound_mask)
                    base_ind = np.where(base_mask)[0]
                    m_peak_arr = kin.v_peak(sat.tree[host_key], hal_mask=base_mask, hal_property='mass')
                    m_peak_mask = m_peak_arr > sat.mass_peak
                    m_peak_ind = base_ind[m_peak_mask]

                    # code from Megan Barry 9/2021
                    progenitor_indices = sat.tree[host_key].prop('progenitor.main.indices',m_peak_ind)
                    progenitor_indices[progenitor_indices < 0] = -1
                    central_indices = sat.tree[host_key]['central.index'][progenitor_indices]
                    LMC_indices = sat.tree[host_key].prop('progenitor.main.indices', sat.lmc_index)
                    LMC_tracker = np.full_like(progenitor_indices,0)
                    dummy_tracker = np.full_like(progenitor_indices,1)
                    for LMC_index in LMC_indices:
                        #returns a 1 if the subhalo in central_indices was ever a satellite of an LMC, 0 if not
                        LMC_tracker = np.where(central_indices == LMC_index,dummy_tracker,LMC_tracker)
                    result = np.sum(LMC_tracker,axis=1)
                    LMC_mask = result > 0

                    lmc_hosted_ind = m_peak_ind[~LMC_mask]
                    combined_mask = np.zeros(len(sat.tree[host_key]['mass']), dtype=bool)
                    combined_mask[lmc_hosted_ind] = True

                else:
                    raise ValueError('Tree mask key ({}) not recognized'.format(star_key))

                redshift_mask_dict[star_key] = combined_mask

            if host_key not in tree_mask_dict.keys():
                tree_mask_dict[host_key] = [redshift_mask_dict]
            else:
                tree_mask_dict[host_key].append(redshift_mask_dict)

    return tree_mask_dict

def mask_tree_lg(sat):
    """
    Applies masks on baryonic halo properties, values for masking are set in the
    SatParam class in halo_reader.py.

    Returns
    -------
    tree_mask_dict : dictionary
        Dictionary of dictionaries of tree masks where key:value pairs
        correspond to sim_name:{mask_type:mask}.
    """
    
    pair_mask_dict = {}
    for pair_name in sat.tree.keys():
        host_mask_dict = {}
        host_list = ['host', 'host2']
        for host_name, host_str in zip(sat.hal_name[pair_name], host_list):
            # check for iterable, if not make iterable
            # (load hals needs a non-iterable, need to fix this inconsistency in a better way)
            # also exits in loop hal
            snapshot_ = sat.snapshots_to_mask
            if type(snapshot_) is int:
                snapshot_ = [snapshot_]
            for snap in snapshot_:
                # look at a single snapshot/redshift
                #redshift_snap_id = sat.tree[pair_name].Snapshot.get_snapshot_indices('redshift', redshift)
                redshift_mask = sat.tree[pair_name]['snapshot'] == snap

                # choose only satellites of a single host galaxy
                host_ind = np.unique(sat.tree[pair_name][host_str+'.index'][redshift_mask])[0]
                host_system_mask = sat.tree[pair_name][host_str+'.index'] == host_ind

                # systematically exclude the host/main halo
                host_mask = ~(sat.tree[pair_name][host_str+'.index'] == np.arange(len(sat.tree[pair_name][host_str+'.index'])))

                # exclude phantom halos generated by rockstar halo finder
                phantom_mask = ~(sat.tree[pair_name]['am.phantom'] == 1)

                # exclude halos contaminated by low resolution dark matter
                lowres_mask = sat.tree[pair_name]['mass.lowres']/sat.tree[pair_name]['mass'] <= sat.low_res

                # exclude halos with bound fractions less than 0.4
                bound_mask = sat.tree[pair_name].prop('mass.bound/mass') > 0.4

                # for star particle data make a cut on half stellar density
                star_density_mask = sat.tree[pair_name].prop('star.density.50') >= sat.star_density

                # select subhalos within a certain distance to be satellites
                #dist_3d = sat.tree[pair_name][host_str+'.distance']
                #total_distance = np.sqrt(dist_3d[:,0]**2 + dist_3d[:,1]**2 + dist_3d[:,2]**2)
                total_distance = sat.tree[pair_name].prop(host_str+'.distance.total')
                distance_mask = ((total_distance >= sat.r_range[0]) & 
                                (total_distance <= sat.r_range[1]))

                # OR select nonsatellites/subhalos outside of a given distance (300 kpc) and within some larger distance
                # this will return identical masks for both LG hosts in the simulation
                # nonsats must be within the larger distance limit of at least 1 host, but may be further from the other
                # also exclude both hosts
                host_mask1 = ~(sat.tree[pair_name]['host.index'] == np.arange(len(sat.tree[pair_name]['host.index'])))
                host_mask2 = ~(sat.tree[pair_name]['host2.index'] == np.arange(len(sat.tree[pair_name]['host2.index'])))
                nonsat_mask1 = sat.tree[pair_name].prop('host.distance.total') > 300
                nonsat_mask2 = sat.tree[pair_name].prop('host2.distance.total') > 300
                outer_distance_mask = ((sat.tree[pair_name].prop('host.distance.total') <= sat.r_range[1]) | 
                    (sat.tree[pair_name].prop('host2.distance.total') <= sat.r_range[1]))
                nonsatellite_mask = nonsat_mask1 & nonsat_mask2 & outer_distance_mask & host_mask1 & host_mask2

                redshift_mask_dict = {}
                for star_key in sat.mask_names:
                    if star_key == 'star.mass':
                        star_mass_low_mask = sat.tree[pair_name]['star.mass'] >= sat.star_mass[0]
                        star_mass_high_mask = sat.tree[pair_name]['star.mass'] <= sat.star_mass[1]
                        combined_mask = (redshift_mask & host_mask & star_mass_low_mask
                                        & star_mass_high_mask & star_density_mask & lowres_mask
                                        & distance_mask & phantom_mask & host_system_mask & bound_mask)

                    elif star_key == 'most.star.mass':
                        #choose top 11 most massive satellites
                        base_mask = (redshift_mask & host_mask & star_density_mask 
                                        & lowres_mask & distance_mask & phantom_mask & host_system_mask & bound_mask)
                        base_ind = np.where(base_mask)[0]
                        base_sm = sat.tree[pair_name]['star.mass'][base_ind]
                        top_n_base_sm_ind = np.argsort(base_sm)[-sat.abs_number:]
                        top_n_ind = base_ind[top_n_base_sm_ind]
                        combined_mask = np.zeros(len(sat.tree[pair_name]['star.mass']), dtype=bool)
                        combined_mask[top_n_ind] = True

                    elif star_key == 'star.number':
                        star_number_mask = sat.tree[pair_name]['star.number'] >= sat.star_number
                        combined_mask = (redshift_mask & host_mask & star_number_mask & 
                                        star_density_mask & lowres_mask & distance_mask
                                        & phantom_mask & host_system_mask & bound_mask)

                    elif star_key == 'star.mass.nonsatellites':
                        star_mass_low_mask = sat.tree[pair_name]['star.mass'] >= sat.star_mass[0]
                        star_mass_high_mask = sat.tree[pair_name]['star.mass'] <= sat.star_mass[1]
                        combined_mask = (redshift_mask & star_mass_low_mask & star_mass_high_mask & 
                                        star_density_mask & lowres_mask & phantom_mask & bound_mask &
                                        nonsatellite_mask)

                    elif star_key == 'vel.circ.max':
                        v_circ_mask = sat.tree[pair_name]['vel.circ.max'] >= sat.vel_circ_max
                        combined_mask = (redshift_mask & host_mask & lowres_mask &
                                        distance_mask & v_circ_mask & phantom_mask & host_system_mask & bound_mask)

                    elif star_key == 'v.peak':
                        base_mask = (redshift_mask & host_mask &
                                    lowres_mask & distance_mask & phantom_mask & 
                                    bound_mask)
                        base_ind = np.where(base_mask)[0]
                        v_peak_arr = kin.v_peak(sat.tree[pair_name], hal_mask=base_mask)
                        v_peak_mask = v_peak_arr > sat.v_peak
                        v_peak_ind = base_ind[v_peak_mask]
                        combined_mask = np.zeros(len(sat.tree[pair_name]['vel.circ.max']), dtype=bool)
                        combined_mask[v_peak_ind] = True

                    elif star_key == 'mass.peak':
                        base_mask = (redshift_mask & host_mask &
                                    lowres_mask & distance_mask & phantom_mask & 
                                    bound_mask)
                        base_ind = np.where(base_mask)[0]
                        m_peak_arr = kin.v_peak(sat.tree[pair_name], hal_mask=base_mask, hal_property='mass')
                        m_peak_mask = m_peak_arr > sat.mass_peak
                        m_peak_ind = base_ind[m_peak_mask]
                        combined_mask = np.zeros(len(sat.tree[pair_name]['mass']), dtype=bool)
                        combined_mask[m_peak_ind] = True

                    elif star_key == 'mass.bound':
                        bound_mask_ = sat.tree[pair_name].prop('mass.bound') > sat.m_bound
                        combined_mask = (redshift_mask & host_mask & 
                                    lowres_mask & distance_mask & phantom_mask & 
                                    bound_mask & bound_mask_)

                    elif star_key == 'completeness':
                        star_density_mask = sat.tree[pair_name].prop('star.density.50') >= sat.star_density
                        total_distance = sat.tree[pair_name].prop(host_str+'.distance.total')
                        star_mass_high_mask = sat.tree[pair_name]['star.mass'] <= sat.star_mass[1]

                        d_1_mask = ((total_distance >= sat.r_range[0]) & 
                                        (total_distance <= sat.d_complete[0])) 
                        sm_low_mask1 = sat.tree[pair_name]['star.mass'] >= sat.sm_complete[0]
                        combined_mask1 = (redshift_mask & host_mask & sm_low_mask1
                                        & star_mass_high_mask & star_density_mask & lowres_mask
                                        & d_1_mask & phantom_mask & bound_mask)

                        sm_low_mask2 = sat.tree[pair_name]['star.mass'] >= sat.sm_complete[1]
                        d_2_mask = ((total_distance >= sat.r_range[0]) & 
                                        (total_distance <= sat.d_complete[1])) 
                        combined_mask2 = (redshift_mask & host_mask & sm_low_mask2
                                        & star_mass_high_mask & star_density_mask & lowres_mask
                                        & d_2_mask & phantom_mask & bound_mask)

                        combined_mask = combined_mask1 | combined_mask2

                    else:
                        raise ValueError('Tree mask key ({}) not recognized'.format(star_key))

                    redshift_mask_dict[star_key] = combined_mask

                if host_name not in host_mask_dict.keys():
                    host_mask_dict[host_name] = [redshift_mask_dict]
                else:
                    host_mask_dict[host_name].append(redshift_mask_dict)

        pair_mask_dict[pair_name] = host_mask_dict
    '''

    ##########################################################################
    # try getting a mask at redshift 0 and tracking those halos back in time #
    ##########################################################################

    pair_mask_dict = {}
    for pair_name in sat.tree.keys():
        host_mask_dict = {}
        host_list = ['host', 'host2']
        for host_name, host_str in zip(sat.hal_name[pair_name], host_list):
            # look at a single snapshot/redshift
            redshift_snap_id = sat.tree[pair_name].Snapshot.get_snapshot_indices('redshift', np.min(sat.redshift))
            redshift_mask = sat.tree[pair_name]['snapshot'] == redshift_snap_id

            # choose only satellites of a single host galaxy
            host_ind = np.unique(sat.tree[pair_name][host_str+'.index'][redshift_mask])[0]
            host_system_mask = sat.tree[pair_name][host_str+'.index'] == host_ind

            # systematically exclude the host/main halo
            host_mask = ~(sat.tree[pair_name][host_str+'.index'] == np.arange(len(sat.tree[pair_name][host_str+'.index'])))

            # exclude phantom halos generated by rockstar halo finder
            phantom_mask = ~(sat.tree[pair_name]['am.phantom'] == 1)

            # exclude halos contaminated by low resolution dark matter
            lowres_mask = sat.tree[pair_name]['mass.lowres']/sat.tree[pair_name]['mass'] <= sat.low_res

            # select subhalos within a certain distance to be satellites
            dist_3d = sat.tree[pair_name][host_str+'.distance']
            total_distance = np.sqrt(dist_3d[:,0]**2 + dist_3d[:,1]**2 + dist_3d[:,2]**2)
            distance_mask = ((total_distance >= sat.r_range[0]) & 
                            (total_distance <= sat.r_range[1]))

            # for star particle data make a cut on half stellar density
            star_density_mask = sat.tree[pair_name].prop('star.density.50') >= sat.star_density

            star_key_mask_dict = {}
            for star_key in sat.mask_names:
                if star_key == 'star.mass':
                    # choose satellites in a given stellar mass range
                    star_mass_low_mask = sat.tree[pair_name]['star.mass'] >= sat.star_mass[0]
                    star_mass_high_mask = sat.tree[pair_name]['star.mass'] <= sat.star_mass[1]
                    combined_mask = (redshift_mask & host_mask & star_mass_low_mask
                                    & star_mass_high_mask & star_density_mask & lowres_mask
                                    & distance_mask & phantom_mask & host_system_mask)

                elif star_key == 'most.massive':
                    # choose a number (sat.abs_number) of the most massive
                    # satellites
                    base_mask = (redshift_mask & host_mask & star_density_mask 
                                    & lowres_mask & distance_mask & phantom_mask & host_system_mask)
                    base_ind = np.where(base_mask)[0]
                    base_sm = sat.tree[pair_name]['star.mass'][base_ind]
                    top_n_base_sm_ind = np.argsort(base_sm)[-sat.abs_number:]
                    top_n_ind = base_ind[top_n_base_sm_ind]
                    combined_mask = np.zeros(len(sat.tree[pair_name]['star.mass']), dtype=bool)
                    combined_mask[top_n_ind] = True

                elif star_key == 'star.number':
                    # choose satellites with at least sat.star_number number of
                    # star particles
                    star_number_mask = sat.tree[pair_name]['star.number'] >= sat.star_number
                    combined_mask = (redshift_mask & host_mask & star_number_mask & 
                                    star_density_mask & lowres_mask & distance_mask
                                    & phantom_mask & host_system_mask)

                elif star_key == 'vel.circ.max':
                    # choose satellites above a maximum circular velocity
                    v_circ_mask = sat.tree[pair_name]['vel.circ.max'] >= sat.vel_circ_max
                    combined_mask = (redshift_mask & host_mask & lowres_mask &
                                    distance_mask & v_circ_mask & phantom_mask & host_system_mask)

                star_key_indices = np.where(combined_mask)[0]
                star_key_tracker = halo_track(sat.tree[pair_name], star_key_indices, sat.redshift)
                star_key_mask_dict[star_key] = star_key_tracker

            host_mask_dict[host_name] = star_key_mask_dict

        pair_mask_dict[pair_name] = host_mask_dict
    '''

    return pair_mask_dict

def mask_tree_dmo(sat):
    """
    Applies masks on dark matter only halo properties, values for masking are
    set in the SatParam class in halo_reader.py.

    Returns
    -------
    tree_mask_dict : dictionary
        Dictionary of dictionaries of tree masks where key:value pairs
        correspond to sim_name:{mask_type:mask}.
    """
    tree_mask_dict = {}
    for host_key in sat.tree.keys():
        #for i, redshift in enumerate(sat.redshift):
        snapshot_ = sat.snapshots_to_mask
        if type(snapshot_) is int:
            snapshot_ = [snapshot_]
        for i,snap in enumerate(snapshot_):
            # look at a single snapshot/redshift
            #redshift_snap_id = sat.tree[host_key].Snapshot.get_snapshot_indices('redshift', redshift)
            redshift_mask = sat.tree[host_key]['snapshot'] == snap

            # systematically exclude the host/main halo
            host_mask = ~(sat.tree[host_key]['host.index'] == np.arange(len(sat.tree[host_key]['host.index'])))

            # exclude phantom halos generated by rockstar halo finder
            phantom_mask = ~(sat.tree[host_key]['am.phantom'] == 1)

            # exclude halos contaminated by low resolution dark matter
            lowres_mask = sat.tree[host_key]['mass.lowres']/sat.tree[host_key]['mass'] <= sat.low_res

            # exclude halos with bound fractions less than 0.4
            bound_mask = sat.tree[host_key].prop('mass.bound/mass') > 0.4

            # select subhalos within a certain distance to be satellites
            total_distance = sat.tree[host_key].prop('host.distance.total')
            distance_mask = ((total_distance >= sat.r_range[0]) & 
                            (total_distance <= sat.r_range[1]))

            redshift_mask_dict = {}
            for dmo_key in sat.mask_names:
                if dmo_key == 'number.sats':
                    n_sats = sat.num_sats[host_key][i]
                    base_mask = host_mask & redshift_mask & phantom_mask & distance_mask & lowres_mask & bound_mask
                    base_ind = np.where(base_mask)[0]
                    base_vmax = sat.tree[host_key]['vel.circ.max.fbcorr'][base_ind]

                    top_n_base_vmax_ind = np.argsort(base_vmax)[-n_sats:]
                    top_n_ind = base_ind[top_n_base_vmax_ind]
                    combined_mask = np.zeros(len(sat.tree[host_key]['vel.circ.max.fbcorr']), dtype=bool)
                    combined_mask[top_n_ind] = True

                elif dmo_key == 'most.massive':
                    base_mask = host_mask & redshift_mask & phantom_mask & distance_mask & lowres_mask & bound_mask
                    base_ind = np.where(base_mask)[0]
                    base_vmax = sat.tree[host_key]['vel.circ.max.fbcorr'][base_ind]

                    top_n_base_vmax_ind = np.argsort(base_vmax)[-sat.abs_number:]
                    top_n_ind = base_ind[top_n_base_vmax_ind]
                    combined_mask = np.zeros(len(sat.tree[host_key]['vel.circ.max.fbcorr']), dtype=bool)
                    combined_mask[top_n_ind] = True

                elif dmo_key == 'median.vel.circ.max':
                    v_mask_med = sat.tree[host_key]['vel.circ.max.fbcorr'] >= sat.med_v_circ[i]
                    combined_mask = host_mask & redshift_mask & phantom_mask & distance_mask & lowres_mask & v_mask_med & bound_mask

                elif dmo_key == 'vel.circ.max':
                    v_mask_const = sat.tree[host_key]['vel.circ.max.fbcorr'] >= sat.vel_circ_max
                    combined_mask = host_mask & redshift_mask & phantom_mask & distance_mask & lowres_mask & v_mask_const & bound_mask

                elif dmo_key == 'v.peak':
                    base_mask = host_mask & redshift_mask & phantom_mask & distance_mask & lowres_mask & bound_mask
                    base_ind = np.where(base_mask)[0]
                    v_peak_arr = kin.v_peak(sat.tree[host_key], hal_mask=base_mask, hal_property='vel.circ.max.fbcorr')
                    v_peak_mask = v_peak_arr > sat.v_peak
                    v_peak_ind = base_ind[v_peak_mask]
                    combined_mask = np.zeros(len(sat.tree[host_key]['vel.circ.max.fbcorr']), dtype=bool)
                    combined_mask[v_peak_ind] = True

                elif dmo_key == 'mass.peak':
                    base_mask = host_mask & redshift_mask & phantom_mask & distance_mask & lowres_mask & bound_mask
                    base_ind = np.where(base_mask)[0]
                    m_peak_arr = kin.v_peak(sat.tree[host_key], hal_mask=base_mask, hal_property='mass.fbcorr')
                    m_peak_mask = m_peak_arr > sat.mass_peak
                    m_peak_ind = base_ind[m_peak_mask]
                    combined_mask = np.zeros(len(sat.tree[host_key]['mass.fbcorr']), dtype=bool)
                    combined_mask[m_peak_ind] = True

                elif dmo_key == 'mass.bound':
                    bound_mask_ = sat.tree[host_key].prop('mass.bound.fbcorr') > sat.m_bound
                    combined_mask = (redshift_mask & host_mask & 
                                 lowres_mask & distance_mask & phantom_mask & 
                                 bound_mask & bound_mask_)

                else:
                    raise ValueError('Tree mask key ({}) not recognized'.format(dmo_key))

                redshift_mask_dict[dmo_key] = combined_mask

            if host_key not in tree_mask_dict.keys():
                tree_mask_dict[host_key] = [redshift_mask_dict]
            else:
                tree_mask_dict[host_key].append(redshift_mask_dict)

    return tree_mask_dict


#################################################################
### make isotropic versions of halo catalogs and merger trees ###
#################################################################

def hal_iso(sat, mask_names, sat_data=None):
    iso_mask_dict = defaultdict(list)
    if sat.sat_type == 'hal':
        for host_key in sat.hal_catalog.keys():
            for redshift_index in range(len(sat.redshift)):
                snap_mask_dict = {}
                for mask in mask_names:
                    hal = sat.hal_catalog[host_key][redshift_index]
                    hal_mask = sat.catalog_mask[host_key][redshift_index][mask]
                    snap_mask_dict[mask] = iso.iso_distribution(hal, hal_mask, n_iter=sat.n_iter)
                iso_mask_dict[host_key].append(snap_mask_dict)

    elif sat.sat_type == 'hal.lg':
        for pair_name in sat.hal_catalog.keys():
            for host_name, host_str in zip(sat.hal_name[pair_name], ['host.', 'host2.']):
                for redshift_index in range(len(sat.redshift)):
                    snap_mask_dict = {}
                    for mask in mask_names:
                        hal = sat.hal_catalog[pair_name][redshift_index]
                        hal_mask = sat.catalog_mask[pair_name][host_name][redshift_index][mask]
                        snap_mask_dict[mask] = iso.iso_distribution(hal, hal_mask, host_str=host_str, n_iter=sat.n_iter)
                    iso_mask_dict[host_name].append(snap_mask_dict)
    else:
        print('Satellite object type not recognized')

    return iso_mask_dict

def hal_iso_lite(sat_data, sat_params, mask_names):
    iso_mask_dict = defaultdict(list)
    if sat_params.sat_type == 'hal.lite':
        for host_key in sat_data.keys():
            for redshift_index in range(len(sat_params.redshift)):
                snap_mask_dict = {}
                for mask in mask_names:
                    hal = sat_data[host_key][redshift_index]
                    hal_mask = sat_params.catalog_mask[host_key][redshift_index][mask]
                    snap_mask_dict[mask] = iso.iso_distribution(hal, hal_mask, n_iter=sat_params.n_iter)
                iso_mask_dict[host_key].append(snap_mask_dict)
    else:
        print('Satellite object type not recognized')

    return iso_mask_dict

def tree_iso(sat, mask_names):
    """
    Creates isotropic distributions for each host and each mask on the merger
    trees contained in sat object.

    Parameters
    ----------
    sat : SatelliteTree object
        Class defined in halo_reader.py.
    mask_names : list
        List of strings corresponding to types of masks in sat object.
    """
    iso_mask_dict = defaultdict(list)
    if sat.sat_type == 'tree':
        for host_key in sat.tree.keys():
            for redshift_index in range(len(sat.redshift)):
                snap_mask_dict = {}
                for mask in mask_names:
                    hal = sat.tree[host_key]
                    hal_mask = sat.tree_mask[host_key][redshift_index][mask]
                    snap_mask_dict[mask] = iso.iso_distribution(hal, hal_mask, n_iter=sat.n_iter)
                iso_mask_dict[host_key].append(snap_mask_dict)
    elif sat.sat_type == 'tree.lg':
        for pair_name in sat.tree.keys():
            for host_name, host_str in zip(sat.hal_name[pair_name], ['host.', 'host2.']):
                for redshift_index in range(len(sat.redshift)):
                    snap_mask_dict = {}
                    for mask in mask_names:
                        tree = sat.tree[pair_name]
                        tree_mask = sat.tree_mask[pair_name][host_name][redshift_index][mask]
                        snap_mask_dict[mask] = iso.iso_distribution(tree, tree_mask, host_str=host_str, n_iter=sat.n_iter)
                    iso_mask_dict[host_name].append(snap_mask_dict)

    return iso_mask_dict

# below needs baryon fraction (f_b) correction updates (at least)

##########################################################
### loop functions over halo catalogs and merger trees ###
##########################################################

def loop_sat(sat, exec_func, **kwargs):
    '''
    Loop a function over a time series of hals for each host in a Satellite object.
    Used in creation of halo catalog masks and isotropic distributions in halo_reader.py.
    '''
    loop_dict = {}
    for hal_name in sat.hal_catalog.keys():
        for redshift_index in range(len(sat.hal_catalog[hal_name])):

            if hal_name not in loop_dict.keys():
                loop_dict[hal_name] = [exec_func(sat, hal_name, redshift_index, **kwargs)]

            else:
                loop_dict[hal_name].append(exec_func(sat, hal_name, redshift_index, **kwargs))

    return loop_dict

def loop_hal(sat, mask_key, exec_func, **kwargs):
    '''
    Loop a function over a series of halo catalogs for each simulation in a sat 
    object.

    Parameters
    ----------
    sat : Satellite object
        class created in halo_reader.py
    mask_key : str
        keyword specifying the type of halo property to cut/mask halos on
    exec_func : function
        function to call repeatedly/pass sat to
    kwargs : dictionary
        dictionary of key value pairs where keys are the keywords of exec_func

    Returns
    -------
    loop_dict : dictionary
        stores the output of exec_func for each simulation and snapshot in sat
    '''
    loop_dict = defaultdict(list)
    if sat.snapshots_to_mask is None:
        snap_list = sat.snapshot
    else:
        snap_list = sat.snapshots_to_mask
    if type(snap_list) == int:
        snap_list = [snap_list]

    if sat.sat_type == 'tree':
        for host_name in sat.tree.keys():
            for redshift_index in range(len(snap_list)):
                tree = sat.tree[host_name]
                tree_mask = sat.tree_mask[host_name][redshift_index][mask_key]
                loop_dict[host_name].append(exec_func(hal=tree, hal_mask=tree_mask, **kwargs))
    elif sat.sat_type == 'tree.lg':
        for pair_name in sat.tree.keys():
            for host_name, host_str in zip(sat.hal_name[pair_name], ['host.', 'host2.']):
                for redshift_index in range(len(snap_list)):
                    tree = sat.tree[pair_name]
                    tree_mask = sat.tree_mask[pair_name][host_name][redshift_index][mask_key]
                    loop_dict[host_name].append(exec_func(hal=tree, hal_mask=tree_mask, host_str=host_str, **kwargs))
    elif sat.sat_type == 'hal':
        for hal_name in sat.hal_name:
            for redshift_index in range(len(snap_list)):
                #the line below implicitly loops over the snapshots/redshifts that have been loaded
                #for hal, hal_mask in zip(sat.hal_catalog[hal_name], sat.catalog_mask[mask_key][hal_name]):
                hal = sat.hal_catalog[hal_name][redshift_index]
                hal_mask = sat.catalog_mask[hal_name][redshift_index][mask_key]
                loop_dict[hal_name].append(exec_func(hal=hal, hal_mask=hal_mask, **kwargs))
    elif sat.sat_type == 'hal.lg':
        for pair_name in sat.hal_catalog.keys():
            for host_name, host_str in zip(sat.hal_name[pair_name], ['host.', 'host2.']):
                for redshift_index in range(len(snap_list)):
                    cat = sat.hal_catalog[pair_name][redshift_index]
                    cat_mask = sat.catalog_mask[pair_name][host_name][redshift_index][mask_key]
                    loop_dict[host_name].append(exec_func(hal=cat, hal_mask=cat_mask, host_str=host_str, **kwargs))
    else:
        print('sat type not recognized')

    return loop_dict

def loop_hal_lite(sat_data, sat_params, mask_key, exec_func, **kwargs):
    '''
    Loop a function over a series of halo catalogs for each simulation in a sat 
    object.

    Parameters
    ----------
    sat : Satellite object
        class created in halo_reader.py
    mask_key : str
        keyword specifying the type of halo property to cut/mask halos on
    exec_func : function
        function to call repeatedly/pass sat to
    kwargs : dictionary
        dictionary of key value pairs where keys are the keywords of exec_func

    Returns
    -------
    loop_dict : dictionary
        stores the output of exec_func for each simulation and snapshot in sat
    '''
    loop_dict = defaultdict(list)
    if sat_params.sat_type == 'hal.lite':
        for host_key in sat_params.hal_name:
            for redshift_index in range(len(sat_params.redshift)):
                hal = sat_data[host_key][redshift_index]
                hal_mask = sat_params.catalog_mask[host_key][redshift_index][mask_key]
                loop_dict[host_key].append(exec_func(hal=hal, hal_mask=hal_mask, **kwargs))
    elif sat_params.sat_type == 'hal.lg.lite':
        for pair_name in sat_data.keys():
            for host_name, host_str in zip(sat_params.hal_name[pair_name], ['host.', 'host2.']):
                for redshift_index in range(len(sat_params.redshift)):
                    cat = sat_data[pair_name][redshift_index]
                    cat_mask = sat_params.catalog_mask[pair_name][host_name][redshift_index][mask_key]
                    loop_dict[host_name].append(exec_func(hal=cat, hal_mask=cat_mask, host_str=host_str, **kwargs))
    else:
        print('sat params type not recognized')

    return loop_dict

def loop_iso(sat, mask_key, exec_func, **kwargs):
    '''
    Loop over a time series of isotropic satellite distributions.
    '''
    loop_dict = defaultdict(list)
    if 'tree' in sat.sat_type:
        for host_key in sat.isotropic.keys():
            for redshift_index in range(len(sat.redshift)):
                iso_hal = sat.isotropic[host_key][redshift_index][mask_key]
                loop_dict[host_key].append(exec_func(iso_hal, **kwargs))
    elif 'hal' in sat.sat_type:
        for host_name in sat.isotropic.keys():
            for redshift_index in range(len(sat.redshift)):
                iso_hal = sat.isotropic[host_name][redshift_index][mask_key]
                loop_dict[host_name].append(exec_func(iso_hal, **kwargs))
    else:
        print('unknown Satelite object type')

    return loop_dict

def loop_iso_lite(sat_params, mask_key, exec_func, **kwargs):
    '''
    Loop over a time series of isotropic satellite distributions.
    '''
    loop_dict = defaultdict(list)
    if sat_params.sat_type == 'hal.lite':
        for host_key in sat_params.isotropic.keys():
            for redshift_index in range(len(sat_params.redshift)):
                iso_hal = sat_params.isotropic[host_key][redshift_index][mask_key]
                loop_dict[host_key].append(exec_func(iso_hal, **kwargs))
    else:
        print('unknown Satelite object type')

    return loop_dict

# doesn't work
def loop_iso_parallel(sat, mask_key, exec_func, kwargs):
    '''
    Loop over a time series of isotropic satellite distributions.
    '''
    import multiprocessing as mp
    pool = mp.Pool(mp.cpu_count())

    loop_dict = defaultdict(list)
    if 'tree' in sat.sat_type:
        for host_key in sat.isotropic.keys():
            for redshift_index in range(len(sat.redshift)):
                iso_hal = sat.isotropic[host_key][redshift_index][mask_key]
                loop_dict[host_key].append(exec_func(iso_hal, **kwargs))
    elif 'hal' in sat.sat_type:
        for host_name in sat.isotropic.keys():
            for redshift_index in range(len(sat.redshift)):
                iso_hal = sat.isotropic[host_name][redshift_index][mask_key]
                loop_dict[host_name].append(exec_func(iso_hal, **kwargs))
    else:
        print('unknown Satelite object type')

    return loop_dict

def iso_snapshot_loop(exec_func, tree, host_masks, mask_key, snapshot_index, kwargs):
    px = exec_func(
        hal=tree, 
        hal_mask=host_masks[snapshot_index][mask_key], 
        **kwargs)
    return px

# doesn't work
def loop_hal_parallel(sat, mask_key, exec_func, kwargs):
    import multiprocessing as mp
    pool = mp.Pool(32)

    loop_dict = defaultdict(list)
    if sat.sat_type == 'tree':
        for host_name in sat.tree.keys():
            loop_dict[host_name] = [pool.apply(snapshot_loop, args=(exec_func, sat.tree[host_name], sat.tree_mask[host_name], mask_key, snapshot_ind, kwargs)) for snapshot_ind in range(len(sat.redshift))]
    elif sat.sat_type == 'tree.lg':
        for pair_name in sat.tree.keys():
            for host_name, host_str in zip(sat.hal_name[pair_name], ['host.', 'host2.']):
                loop_dict[host_name] = [pool.apply(snapshot_loop_lg, args=(exec_func, sat, host_name, mask_key, pair_name, host_str, snapshot_ind), **kwargs) for snapshot_ind in range(len(sat.redshift))]
    elif sat.sat_type == 'hal':
        for hal_name in sat.hal_name:
            loop_dict[hal_name] = [pool.apply(snapshot_loop_hal, args=(exec_func, sat.hal_catalog[hal_name][snapshot_ind], sat.catalog_mask[mask_key]['mask'][hal_name][snapshot_ind], kwargs)) for snapshot_ind in range(len(sat.redshift))]
    elif sat.sat_type == 'hal.lg':
        for pair_name in sat.hal_catalog.keys():
            for host_name, host_str in zip(sat.hal_name[pair_name], ['host.', 'host2.']):
                for redshift_index in range(len(sat.redshift)):
                    cat = sat.hal_catalog[pair_name][redshift_index]
                    cat_mask = sat.catalog_mask[pair_name][host_name][redshift_index][mask_key]
                    loop_dict[host_name].append(exec_func(hal=cat, hal_mask=cat_mask, host_str=host_str, **kwargs))
    pool.close()
    pool.join()
    return loop_dict

def snapshot_loop(exec_func, tree, host_masks, mask_key, snapshot_index, kwargs):
    px = exec_func(
        hal=tree, 
        hal_mask=host_masks[snapshot_index][mask_key], 
        **kwargs)
    return px

def snapshot_loop_hal(exec_func, hal, host_masks, kwargs):
    px = exec_func(
        hal=hal, 
        hal_mask=host_masks, 
        **kwargs)
    return px

def snapshot_loop_lg(exec_func, sat, host_name, mask_key, pair_name, host_str, snapshot_index, kwargs):
    px = exec_func(
        hal=sat.tree[host_name], 
        hal_mask=sat.tree_mask[pair_name][host_name][snapshot_index][mask_key], 
        host_str=host_str,
        **kwargs)
    return px


########################################
### halo tracking & helper functions ###
########################################

def single_property_dict(sat_dict, dict_key):
    '''
    Take a dictionary containing multiple properties and return a dictionary of
    a single property (dict_key) for all host halos in sat_dict.
    '''
    single_dict = {}
    for hal_name in sat_dict.keys():
        redshifts = len(sat_dict[hal_name])
        prop_list = [sat_dict[hal_name][j][dict_key] for j in range(redshifts)]
        single_dict[hal_name] = prop_list

    return single_dict

def hal_label_names(sat):
    '''
    Makes plotting labels for each mask for informative legends.
    '''
    star_mass_names = []
    star_number_names = []
    vcirc_names = []
    most_massive_names = []

    name_iter = hal_name_iter(sat)

    for name in name_iter:
        sm_mag = math.floor(np.log10(sat.star_mass[0]))
        sm_coeff = sat.star_mass[0]/10**(sm_mag)
        if sm_coeff == 1.0:
            star_mass_names.append(name+r' ($M_{\star} \geq $'+r'$10^{}$'.format(sm_mag)+r'$ M_{\odot}$)')
        else:
            star_mass_names.append(name+r' ($M_{\star} \geq $'+r'${} \times 10^{}$'.format(sm_coeff, sm_mag)+r'$ M_{\odot}$)')
        star_number_names.append(name+r' ($N_{SP} \geq $'+'{}'.format(int(sat.star_number)))
        vcirc_names.append(name+r' ($v_{circ} \geq $'+'{}'.format(sat.vel_circ_max)+'$km/s$')
        most_massive_names.append(r'{} ({} most massive)'.format(name, sat.abs_number))

    label_names_dict = {'star.mass':star_mass_names, 'star.number':star_number_names, 'vel.circ.max':vcirc_names, 'most.massive':most_massive_names}

    return label_names_dict

def default_mask(hal, hal_mask):
    if hal_mask is None:
        hal_mask = np.ones(len(hal['id']), dtype=bool)
    else:
        pass

    return hal_mask

def hal_name_iter(sat):
    if sat.sat_type == 'tree.lg':
        name_iter = []
        for pair_name in sat.hal_name.keys():
            name_iter = name_iter + sat.hal_name[pair_name]
    elif sat.sat_type == 'tree':
        name_iter = list(sat.hal_name)
    elif sat.sat_type == 'hal':
        name_iter = list(sat.hal_name)
    elif sat.sat_type == 'hal.lg':
        name_iter = []
        for pair_name in sat.hal_name.keys():
            name_iter = name_iter + sat.hal_name[pair_name]

    return name_iter

def sort_by_key_coadd(
    sat, calc_sat_dict, name_iter=None, obs_sat_dict=None, obs_sys='MW', 
    obs_sample=None):
    mean = []
    median = []
    percentile = []

    if name_iter is None:
        name_iter = hal_name_iter(sat)
    for i,sm in enumerate(calc_sat_dict.keys()):
        sm_ratio_list = []
        for hal_name in name_iter:
            hal_array = calc_sat_dict[sm][hal_name]
            if obs_sample:
                hal_med = np.nanmedian(hal_array)
                sm_ratio_list.append([hal_med/obs_i for obs_i in obs_sample[i]])
            else:
                for ha in hal_array:
                    if obs_sat_dict:
                        sm_ratio_list.append(ha[0]/obs_sat_dict[obs_sys][i])
                    else:
                        sm_ratio_list.append(ha[0])

        mean.append(np.nanmean(sm_ratio_list))
        median.append(np.nanmedian(sm_ratio_list))
        percentile.append(np.nanpercentile(sm_ratio_list, [16, 84, 2.5, 97.5]))

    sorted_coadd = {'mean':mean, 'median':median, 'percentile':percentile}

    return sorted_coadd

def reset_sm_limits(sat, lower_lim, upper_lim=1e10, obs=False):
    sat.star_mass = [lower_lim, upper_lim]
    if sat.sat_type == 'tree':
        sat.tree_mask = mask_tree(sat)
    elif sat.sat_type == 'tree.lg':
        sat.tree_mask = mask_tree_lg(sat)
    elif sat.sat_type == 'hal':
        sat.catalog_mask = multiple_catalog_masks(sat, mask_names=['star.mass'])
    elif sat.sat_type == 'hal.lg':
        sat.catalog_mask = mask_lg_baryon_cat(sat)
    sat.hal_label = hal_label_names(sat)
    if obs:
        sat.observation_mask = obs.obs_mask(sat.observation, star_mass=sat.star_mass, r_range=sat.r_range)

    return sat

def iso_probability(true_property, iso_property):
    prob_over_time = {}
    for host in true_property.keys():
        host_time_prob = []
        for i in range(len(true_property[host])):
            host_time = true_property[host][i]
            iso_host_time = iso_property[host][i]
            host_prob = np.sum(np.array(iso_host_time) <= host_time)/len(iso_host_time)
            host_time_prob.append(host_prob)
        prob_over_time[host] = host_time_prob

    return prob_over_time

def halo_track(
    tree, initial_tree_indices, redshift_list=None, snapshot_list=None):
    """
    Track halos in a merger tree at specific times given by redshift or
    given by snapshot. Must supply either a redshift list or a snapshot list
    that ends with the redshift/snapshot at which initial_tree_indices are
    specified.
 
    Parameters
    ----------
    tree : dictionary
        Merger tree loaded in using halo_analysis package.
    initial_tree_indices : list or array
        1D array of tree indices of subhalos at any individual snapshot.
    redshift_list : list or array
        List of floats specifying which redshifts to track subhalos at and save.
    snapshot_list : list or array
        List of ints specifying which snapshots to track subhalos at and save.

    Returns
    -------
    hal_tracker : array
        Array of size (601, # of tracked subhalos) with the tracked tree indices
        of each subhalo at each snapshot. The index along the first dimension of
        this array corresponds directly to the snapshot number. Untracked snapshots
        are left with default values of -1, and when a subhalo is lost (going
        back in time) its last known tree index is repeated through snapshot 0.
    """
    if redshift_list:
        snapshot_ids = tree.Snapshot.get_snapshot_indices('redshift', redshift_list)
    else:
        snapshot_ids = snapshot_list

    hal_tracker = np.full((601, len(initial_tree_indices)), -1, dtype='int32')
    i = np.max(snapshot_ids)
    hal_tracker[i] = initial_tree_indices

    tracking_ids = initial_tree_indices
    while i >= np.min(snapshot_ids):
        i -= 1
        progenitor_indices = tree['progenitor.main.index'][tracking_ids]
        if i in snapshot_ids:
            tracking_ids = optim_track(tracking_ids, progenitor_indices)
            hal_tracker[i] = tracking_ids

    return hal_tracker

@jit(nopython=True)
def optim_track(track_ids, progenitor_inds):
    negative_ids = np.where(progenitor_inds < 0)[0]
    progenitor_inds[negative_ids] = track_ids[negative_ids]

    return progenitor_inds

def get_galaxies_history(
    hal, hal_mask, host_str='host.', snapshot_limits=[1,600]):
    # hal_mask should be for a selection of satellites at a single snapshot
    initial_snapshot = hal['snapshot'][hal_mask][0]
    all_snapshots = np.arange(snapshot_limits[0], initial_snapshot+1, 1)
    initial_ids = np.where(hal_mask)[0]# these are indices NOT tree id's

    # track the selection of satellites from z=0 all the way back to z=99, and
    # return their tree indices at all snapshots
    tracked_ids = halo_track(hal, initial_ids, snapshot_list=all_snapshots)

    # also track the main MW-like host galaxy and return its tree indices at all
    # snapshots
    host_ind = np.array([hal[host_str+'index'][hal_mask][0]])
    host_track_ = halo_track(hal, host_ind, snapshot_list=all_snapshots)
    host_track = np.array([track[0] for track in host_track_], dtype='int32')

    # get the central index for each tracked satellite at all snapshots
    central_mask = np.array(hal['central.index'][tracked_ids]) >= 0
    central_ind = np.where(central_mask, np.array(hal['central.index'][tracked_ids]), np.full(np.array(hal['central.index'][tracked_ids]).shape, -1))
    
    # find the distance of the central from the main MW-like host galaxy, where
    # the central is lost/doesn't exist, use a nan
    nan_central = np.full(np.array(tracked_ids).shape, np.nan)
    central_mass = np.where(central_mask, hal['mass'][central_ind], nan_central)
    central_star_mass = np.where(central_mask, hal['star.mass'][central_ind], nan_central)
    central_host_distance_total = np.where(central_mask, hal.prop(host_str+'distance.total')[central_ind], nan_central)

    # same as above, but for 3D distance
    central_mask_3d = np.reshape(np.dstack([central_mask, central_mask, central_mask]), np.array(tracked_ids).shape + (3,))
    nan_central_3d = np.full(np.array(tracked_ids).shape + (3,), np.nan)
    central_position = np.where(central_mask_3d, hal['position'][central_ind], nan_central_3d)
    central_velocity = np.where(central_mask_3d, hal['velocity'][central_ind], nan_central_3d)
    central_host_distance = np.where(central_mask_3d, hal[host_str+'distance'][central_ind], nan_central_3d)

    # get the total number of subhalos in each group, cutting on sm and hm
    central_mass_mask = hal['mass'] > 1e7
    central_group_size = np.full(np.array(tracked_ids).size, np.nan)
    hci = hal['central.index']
    central_ind_ = np.where(central_mask, np.array(hal['central.index'][tracked_ids]), np.full(np.array(hal['central.index'][tracked_ids]).shape, np.nan))
    for i,tracked_central_ind in enumerate(np.ndarray.flatten(central_ind_)):
        central_group_size[i] = sum_central(hci, tracked_central_ind, central_mass_mask)
    central_group_size = np.reshape(central_group_size, np.array(tracked_ids).shape)

    # organize into a dictionary for output
    galaxies_history = {
        'snapshot':np.arange(0,snapshot_limits[1]+1,1),
        'tree.index':tracked_ids,
        'mass':np.array(hal['mass'][tracked_ids]),
        'position':hal['position'][tracked_ids],
        'velocity':hal['velocity'][tracked_ids],
        'radius':hal['radius'][tracked_ids],
        'vel.circ.max':hal.prop('vel.circ.max')[tracked_ids],
        'vel.std':hal.prop('vel.std')[tracked_ids],
        'star.mass':np.array(hal['star.mass'][tracked_ids]),
        'star.radius.50':np.array(hal['star.radius.50'][tracked_ids]),
        'star.vel.std.50':np.array(hal['star.vel.std.50'][tracked_ids]),
        'central.index':np.array(hal['central.index'][tracked_ids]),
        'central.mass':central_mass,
        'central.star.mass':central_star_mass,
        'central.position':central_position,
        'central.velocity':central_velocity,
        'central.host.distance':central_host_distance,
        'central.host.distance.total':central_host_distance_total,
        'central.group.size':central_group_size,
        'host.distance':np.array(hal[host_str+'distance'][tracked_ids]),
        'host.distance.total':np.array(hal.prop(host_str+'distance.total')[tracked_ids]),
        'host.velocity':np.array(hal[host_str+'velocity'][tracked_ids]),
        'main.host.index':host_track,
        'main.host.mass':np.array(hal['mass'][host_track]),
        'main.host.radius':np.array(hal['radius'][host_track]),
        'main.host.scale.radius':np.array(hal['scale.radius'][host_track]),
        'main.host.star.mass':np.array(hal['star.mass'][host_track])
    }

    return galaxies_history

@jit(nopython=True)
def sum_central(hal_central_index, y, central_mass_mask_):
    return np.sum((hal_central_index == y) & central_mass_mask_)

def group_assoc(hal, hal_mask, host_str='host.'):
    # see if subhalos have past group associations
    all_snapshots = np.arange(1,601,1)
    initial_ids = np.where(hal_mask)[0]
    tracked_ids = halo_track(hal, initial_ids, snapshot_list=all_snapshots)
    tracked_central = np.array([hal['central.index'][ids.astype(int)] for ids in tracked_ids])
    #tracked_central_local = np.array([hal['central.local.index'][ids.astype(int)] for ids in tracked_ids])

    return tracked_central

def select_out_of_disk(
    sat_coords, host_axes_dict, host_name, snapshot_index, disk_mask_angle=12.0):
    if 'm12' in host_name:
        disk_axes = host_axes_dict[host_name][0][snapshot_index]
    else:
        disk_axes = host_axes_dict[host_name][snapshot_index]
    # cut out satellites that lie within +- disk_mask_angle degrees of the simulated MW disk
    sat_prime_coords = ut.basic.coordinate.get_coordinates_rotated(sat_coords, rotation_tensor=disk_axes)
    tangent_of_open_angle = sat_prime_coords[:,2]/np.sqrt(sat_prime_coords[:,0]**2 + sat_prime_coords[:,1]**2)
    disk_mask = np.abs(np.degrees(np.arctan(tangent_of_open_angle))) > disk_mask_angle

    return disk_mask

def convert_snapshot_to_time(
    snapshot, time_kind='redshift', 
    snap_file_path='/Users/jsamuel/Desktop/Latte/tables/',
    pre_convert_type='snapshot'):
    
    time_table = pd.read_csv(snap_file_path+'snapshot_times.txt', sep=' ')
    
    if type(snapshot) == int:
        snapshot_mask = time_table[pre_convert_type] == snapshot
        converted_time = time_table[time_kind][snapshot_mask].values[0]
        return converted_time
    elif (type(snapshot) == list )|(type(snapshot) == np.ndarray):
        converted_time = []
        for snap in snapshot:
            snapshot_mask = time_table[pre_convert_type] == snap
            converted_time.append(time_table[time_kind][snapshot_mask].values[0])
        return np.array(converted_time)
    else:
        raise ValueError('Data type of input snapshot not recognized. Must be int or list/array.')

def color_cycle(cycle_length=14, cmap_name='plasma', low=0, high=1):
    cmap=plt.get_cmap(cmap_name)
    colors = cmap(np.linspace(low, high, cycle_length))

    return colors