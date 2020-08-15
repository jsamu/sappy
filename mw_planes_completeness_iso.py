import numpy as np
import pandas as pd
import utilities as ut
from numba import jit
from collections import defaultdict
from satellite_analysis import satellite_io as sio
from satellite_analysis import rand_axes as ra
from satellite_analysis import isotropic as iso


def loop_iso(sat, mask_key, exec_func, **kwargs):
    '''
    Loop over a time series of isotropic satellite distributions.
    '''
    loop_dict = defaultdict(list)
    if 'hal' in sat.sat_type:
        for host_name in sat.isotropic.keys():
            for redshift_index in range(len(sat.redshift)):
                iso_hal = sat.isotropic[host_name][redshift_index][mask_key]

                current_snapshot = sat.snapshot[redshift_index]

                loop_dict[host_name].append(
                    exec_func(iso_hal, host_name=host_name, snapshot_index=current_snapshot, **kwargs))
    else:
        print('unknown first input (Satellite object) type')

    return loop_dict

def select_out_of_disk(
    sat_coords, host_axes_dict=None, host_name=None, snapshot_index=None, 
    disk_mask_angle=12.0, return_mask=False):
    if 'm12' in host_name:
        disk_axes = host_axes_dict[host_name][0][snapshot_index]
    else:
        disk_axes = host_axes_dict[host_name][snapshot_index]
    # cut out satellites that lie within +- disk_mask_angle degrees of the simulated MW disk
    # mask is True where satellites are unobscured
    sat_prime_coords = ut.basic.coordinate.get_coordinates_rotated(sat_coords, rotation_tensor=disk_axes)
    tangent_of_open_angle = sat_prime_coords[:,2]/np.sqrt(sat_prime_coords[:,0]**2 + sat_prime_coords[:,1]**2)
    disk_mask = np.abs(np.degrees(np.arctan(tangent_of_open_angle))) > disk_mask_angle

    if return_mask:
        return sat_coords[disk_mask], disk_mask
    else:
        return sat_coords[disk_mask]

def get_iso_disk_mask(
    iso_sat_coords, host_axes_dict, host_name, snapshot_index, disk_mask_angle):
    # get a disk mask of dimension (n_iter, n_sat)
    n_iter, n_sat, dims = iso_sat_coords.shape
    iso_disk_mask_k = np.zeros((n_iter, n_sat), dtype='bool')
    for i, iso_sat_coords_i in enumerate(iso_sat_coords):
        # apply disk mask
        iso_masked_sat_coords, iso_disk_mask = select_out_of_disk(iso_sat_coords_i, 
                                                                host_axes_dict=host_axes_dict, 
                                                                host_name=host_name, 
                                                                snapshot_index=snapshot_index,
                                                                disk_mask_angle=disk_mask_angle, 
                                                                return_mask=True)
        # store the disk mask for each isotropic iteration
        iso_disk_mask_k[i] = iso_disk_mask

    return iso_disk_mask_k

def rand_iso_rms_min(
    iso_hal, r_bins=None, n_iter=None, distribution=True, host_name=None, 
    snapshot_index=None, host_axes_dict=None, disk_mask_angle=12.0):
    rot_vecs, rot_mats = ra.rand_rot_vec(n_iter)
    rms_minor_k = np.zeros((n_iter, n_iter))
    #rms_major_k = np.zeros((n_iter, n_iter))

    # variables defined for optimization
    n_sat = len(iso_hal['iso_coords'][0])
    all_iso_coords = np.reshape(iso_hal['iso_coords'], (n_iter*n_sat, 3))

    # get a disk mask of dimension (n_iter, n_sat)
    iso_disk_mask_k = get_iso_disk_mask(iso_hal['iso_coords'], 
                                        host_axes_dict=host_axes_dict, 
                                        host_name=host_name, 
                                        snapshot_index=snapshot_index,
                                        disk_mask_angle=disk_mask_angle)

    for k,axes in enumerate(rot_vecs):
        # rotate sat coords to a random plane to "fit" to
        sat_prime_coords = ut.basic.coordinate.get_coordinates_rotated(all_iso_coords, rotation_tensor=axes)
        sat_prime_coords = np.reshape(sat_prime_coords, (n_iter, n_sat, 3))

        # get rms height masking out those satellites obscured by the host disk
        # stack 3 copies of iso disk mask for all 3 coordinates
        iso_disk_mask_k_stack = np.dstack([iso_disk_mask_k, iso_disk_mask_k, iso_disk_mask_k])

        # create a copy of the prime coords array with the disk masked entries replaced by nan's
        sat_prime_coords_masked = np.where(iso_disk_mask_k_stack, sat_prime_coords, np.nan)
        rms_minor_k[k] = np.sqrt(np.nanmean((sat_prime_coords_masked[:,:,2])**2, axis=1))
        #rms_major_k[k] = np.sqrt(np.nanmean((sat_prime_coords_masked[:,:,0])**2, axis=1))

    rms_minor_n = np.nanmin(rms_minor_k, axis=0)

    if distribution:
        return rms_minor_n
    else:
        return np.nanmean(rms_minor_n)

def iso_rand_angle_width(
    iso_hal, threshold_fraction=1.0, n_iter=None, angle_range=None,
    angle_width=None, host_name=None, snapshot_index=None, host_axes_dict=None, 
    disk_mask_angle=12.0):
    rand_axes, rand_matrices = ra.rand_rot_vec(n_iter)

    # variables defined for optimization
    n_sat = len(iso_hal['iso_coords'][0])
    iso_coords = iso_hal['iso_coords']
    all_snap_coords = np.reshape(iso_coords, (n_iter*n_sat, 3))
    frac_enclosed_range = np.zeros((len(angle_range), n_iter))
    nan_array = np.full(frac_enclosed_range.shape, np.nan)
    phi_width_k = np.zeros((n_iter, n_iter))
    angle_array = np.zeros(frac_enclosed_range.shape)
    for j,opening_angle in enumerate(angle_range):
        angle_array[j] = np.full(n_iter, opening_angle)

    # get a disk mask of dimension (n_iter, n_sat)
    iso_disk_mask_k = get_iso_disk_mask(iso_hal['iso_coords'], 
                                        host_axes_dict=host_axes_dict, 
                                        host_name=host_name, 
                                        snapshot_index=snapshot_index,
                                        disk_mask_angle=disk_mask_angle)
    n_sat_masked = [np.sum(iso_dm) for iso_dm in iso_disk_mask_k]
    
    for k,axes in enumerate(rand_axes):
        snap_prime_coords = ut.basic.coordinate.get_coordinates_rotated(all_snap_coords, rotation_tensor=axes)
        tangent_of_opening_angle = snap_prime_coords[:,2]/np.sqrt(snap_prime_coords[:,0]**2 + snap_prime_coords[:,1]**2)
        snap_angles_k = np.degrees(np.arctan(tangent_of_opening_angle))
        snap_angles_n = np.reshape(snap_angles_k, (n_iter, n_sat))

        # create a copy of the angles array with the disk masked entries replaced by nan's
        snap_angles_n_masked = np.where(iso_disk_mask_k, snap_angles_n, np.nan)

        phi_width_k[k] = iso.optim_open_angle(snap_angles_n_masked, angle_range, 
            threshold_fraction, n_sat_masked[k], frac_enclosed_range, nan_array, angle_array)

    phi_width_n = np.nanmin(phi_width_k, axis=0)

    return phi_width_n

def iso_axis_ratio(
    iso_hal, distribution=True, verbose=False, host_name=None, 
    snapshot_index=None, host_axes_dict=None, disk_mask_angle=12.0):
    '''
    Calculate the minor to major axis ratio using the MOI tensor at each redshift
    for each host halo's isotropic satellite coordinates.
    '''
    # get a disk mask of dimension (n_iter, n_sat)
    iso_disk_mask = get_iso_disk_mask(iso_hal['iso_coords'], 
                                        host_axes_dict=host_axes_dict, 
                                        host_name=host_name, 
                                        snapshot_index=snapshot_index,
                                        disk_mask_angle=disk_mask_angle)

    iter_ratios = np.zeros(len(iso_hal['iso_coords']))

    for n in range(len(iso_hal['iso_coords'])):
        coords = iso_hal['iso_coords'][n][iso_disk_mask[n]]
        sat_axes = ut.coordinate.get_principal_axes(coords, verbose=verbose)
        iter_ratios[n] = sat_axes[2][0]

    if distribution:
        return iter_ratios
    else:
        return np.nanmean(iter_ratios)

def iso_orbital_pole_dispersion(iso_hal, n_iter=None, host_name=None, 
    snapshot_index=None, host_axes_dict=None, disk_mask_angle=12.0):
    '''
    Calculate the angular dispersion [deg] of satellite orbital poles around
    their mean orbital pole for isotropic unit velocities."
    '''
    # get a disk mask of dimension (n_iter, n_sat)
    iso_disk_mask = get_iso_disk_mask(iso_hal['iso_coords'], 
                                        host_axes_dict=host_axes_dict, 
                                        host_name=host_name, 
                                        snapshot_index=snapshot_index,
                                        disk_mask_angle=disk_mask_angle)

    pole_disp_n = np.zeros(n_iter)
    for n in range(n_iter):
        iso_vels = iso_hal['iso_vels'][n][iso_disk_mask[n]]
        iso_coords = iso_hal['iso_coords'][n][iso_disk_mask[n]]
        if len(iso_vels) == 0:
            pole_disp_n[n] = np.nan
        else:
            j_vec = np.array([np.cross(x,v)/np.linalg.norm(np.cross(x,v)) for x, v in zip(iso_coords, iso_vels)])
            avg_j_vec = np.mean(j_vec, axis=0, dtype=np.float64)/np.linalg.norm(np.mean(j_vec, axis=0))
            avg_j_dot_j = np.array([np.dot(avg_j_vec, j_vec_i) for j_vec_i in j_vec]) 
            pole_disp = np.sqrt(np.mean(np.arccos(avg_j_dot_j)**2, dtype=np.float64))
            pole_disp_n[n] = np.degrees(pole_disp)

    return pole_disp_n
