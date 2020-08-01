import numpy as np
import pandas as pd
import utilities as ut
from numba import jit
from collections import defaultdict
from satellite_analysis import satellite_io as sio
from satellite_analysis import rand_axes as ra
from satellite_analysis import spatial as spa


def loop_hal(sat, mask_key, exec_func, **kwargs):
    '''
    Loop a function over a series of halo catalogs for each simulation in a sat 
    object. Also explicitly pass host names and redshifts to exec_func.
    '''
    loop_dict = defaultdict(list)
    if sat.sat_type == 'hal':
        for hal_name in sat.hal_name:
            for z_index, z in enumerate(sat.redshift):
                hal = sat.hal_catalog[hal_name][z_index]
                hal_mask = sat.catalog_mask[hal_name][z_index][mask_key]

                current_snapshot = sat.snapshot[z_index]

                loop_dict[hal_name].append(
                    exec_func(hal=hal, hal_mask=hal_mask, host_name=hal_name, 
                    snapshot_index=current_snapshot, **kwargs))

    elif sat.sat_type == 'hal.lg':
        for pair_name in sat.hal_catalog.keys():
            for host_name, host_str in zip(sat.hal_name[pair_name], ['host.', 'host2.']):
                for z_index, z in enumerate(sat.redshift):
                    cat = sat.hal_catalog[pair_name][z_index]
                    cat_mask = sat.catalog_mask[pair_name][host_name][z_index][mask_key]

                    current_snapshot = sat.snapshot[z_index]

                    loop_dict[host_name].append(
                        exec_func(hal=cat, hal_mask=cat_mask, host_str=host_str, 
                        host_name=host_name, snapshot_index=current_snapshot, **kwargs))
    else:
        print('sat type not recognized')

    return loop_dict

def select_out_of_disk(
    sat_coords, host_axes_dict, host_name, snapshot_index, disk_mask_angle=12.0,
    return_mask=False):
    if 'm12' in host_name:
        disk_axes = host_axes_dict[host_name][0][snapshot_index]
    else:
        disk_axes = host_axes_dict[host_name][snapshot_index]
    # cut out satellites that lie within +- disk_mask_angle degrees of the simulated MW disk
    sat_prime_coords = ut.basic.coordinate.get_coordinates_rotated(sat_coords, rotation_tensor=disk_axes)
    tangent_of_open_angle = sat_prime_coords[:,2]/np.sqrt(sat_prime_coords[:,0]**2 + sat_prime_coords[:,1]**2)
    disk_mask = np.abs(np.degrees(np.arctan(tangent_of_open_angle))) > disk_mask_angle

    if return_mask:
        return sat_coords[disk_mask], disk_mask
    else:
        return sat_coords[disk_mask]

@jit
def rand_rms_min(
    hal, hal_mask=None, host_str='host.', host_name=None, snapshot_index=None,
    n_iter=None, r_frac=None, radius_bins=None, return_ax=False, 
    return_parallel=False, host_axes_dict=None, disk_mask_angle=12.0):
    '''
    Rotates the 3D positions of the satellites randomly and uniformly for sat.n_iter
    realizations. Finds the rms along the z-axis that encloses the specified
    fraction of satellites, and then gets the minimum of the rms values across all
    random realizations.
    '''
    if r_frac is None:
        sat_coords = hal.prop(host_str+'distance')[hal_mask]
    else:
        rad = spa.r_fraction(hal, hal_mask=hal_mask, host_str=host_str, 
            frac=r_frac, radius_bins=radius_bins)
        radial_mask = hal.prop(host_str+'distance.total') <= rad
        sat_coords = hal.prop(host_str+'distance')[hal_mask & radial_mask]
    rot_vecs, rot_mats = ra.rand_rot_vec(n_iter)
    rms_minor_n = np.zeros(n_iter)
    rms_major_n = np.zeros(n_iter)

    # apply disk mask
    sat_coords = select_out_of_disk(sat_coords, host_axes_dict, host_name, snapshot_index,
                                    disk_mask_angle=disk_mask_angle)

    for n, rot_vec in enumerate(rot_vecs):
        sat_prime_coords = ut.basic.coordinate.get_coordinates_rotated(sat_coords, rotation_tensor=rot_vec)
        rms_minor_n[n] = np.sqrt(np.nanmean(sat_prime_coords[:,2]**2))
        rms_major_n[n] = np.sqrt(np.nanmean(sat_prime_coords[:,0]**2))

    min_rms_minor = np.nanmin(rms_minor_n)
    if return_ax is True:
        try:
            rms_major = rms_major_n[min_index]
            # return just the vector normal to the plane
            min_ax = rot_vecs[min_index][2]
        except:
            rms_major = np.nan
            min_ax = np.array([np.nan, np.nan, np.nan])
        return {'rms_minor':min_rms_minor, 'rms_major':rms_major, 'ax':min_ax}
    elif return_parallel is True:
        try:
            rms_major = rms_major_n[min_index]
        except:
            rms_major = np.nan
        return {'rms_minor':min_rms_minor, 'rms_major':rms_major}
    else:
        return min_rms_minor

def rand_angle(
    hal, hal_mask=None, host_str='host.', host_name=None, snapshot_index=None, 
    n_iter=1000, host_axes_dict=None, disk_mask_angle=12.0):
    '''
    Calculates opening angles off of a set of randomly/isotropicallly generated
    axes (for sat.n_iter realizations).
    Returns opening angles, the vectors that the original axes are rotated by,
    and the rotation matrix that performs the rotations.
    '''
    sat_coords = hal.prop(host_str+'distance')[hal_mask]
    rot_vecs, rot_mats = ra.rand_rot_vec(n_iter)
    open_angle_n = []

    # apply disk mask
    sat_coords = select_out_of_disk(sat_coords, host_axes_dict, host_name, snapshot_index,
                                    disk_mask_angle=disk_mask_angle)

    for n in range(n_iter):
        sat_prime_coords = ut.basic.coordinate.get_coordinates_rotated(sat_coords, rotation_tensor=rot_vecs[n])
        tangent_of_open_angle = sat_prime_coords[:,2]/np.sqrt(sat_prime_coords[:,0]**2 + sat_prime_coords[:,1]**2)
        open_angle_n.append(np.degrees(np.arctan(tangent_of_open_angle)))

    return np.array(open_angle_n), rot_vecs, rot_mats

@jit
def rand_angle_width(
    hal, hal_mask=None, host_str='host.', host_name=None, snapshot_index=None, 
    n_iter=1000, fraction=1.0, angle_range=None, return_ax=False,
    host_axes_dict=None, disk_mask_angle=12.0):
    '''
    Rotates the 3D positions of the satellites randomly and uniformly for sat.n_iter
    realizations. Finds the angle off of the simulation x-axis that encloses a given
    fraction of the satellites, and then gets the minimum of this angle across all
    random realizations for each host halo at each redshift.
    '''
    rand_angles, rand_axes, rand_mats = rand_angle(hal, hal_mask=hal_mask, 
        host_str=host_str, host_name=host_name, snapshot_index=snapshot_index,
        n_iter=n_iter, host_axes_dict=host_axes_dict, disk_mask_angle=disk_mask_angle)
    phi_width_n = np.zeros(n_iter)

    for n, snap_angles_n in enumerate(rand_angles):
        if snap_angles_n.size == 0:
            phi_width_n[n] = np.nan
        else:
            phi_width_n = ra.optim_open_angle(snap_angles_n, angle_range, fraction, phi_width_n, n)

    phi_width = np.nanmin(phi_width_n)
    min_index = np.where(phi_width_n == np.nanmin(phi_width_n))[0][0]

    # return just the vector normal to the plane
    min_ax = rand_axes[min_index][2]

    if return_ax is True:
        return {'angle':phi_width, 'ax':min_ax}
    else:
        return phi_width

@jit
def axis_ratio(
    hal, hal_mask=None, host_str='host.', host_name=None, snapshot_index=None, 
    host_axes_dict=None, disk_mask_angle=12.0, verbose=False):
    '''
    Get the axis ratio (minor/major) for the disk-masked distribution of 
    satellites.
    '''
    sat_coords = hal.prop(host_str+'distance')[hal_mask]
    sat_masked_coords = select_out_of_disk(sat_coords, host_axes_dict, host_name, snapshot_index, disk_mask_angle=disk_mask_angle)
    sat_axes = ut.coordinate.get_principal_axes(sat_masked_coords, verbose=verbose)

    return sat_axes[2][0]

def orbital_ang_momentum(
    hal, hal_mask=None, host_str='host.', host_name=None, snapshot_index=None,
    norm=False, host_axes_dict=None, disk_mask_angle=12.0):
    '''
    Compute mass-agnostic orbital angular momentum as L=(v)x(r) where the
    vectors are defined with respect to the central host halo. Returned value
    has units of kpc^2/s.
    '''
    sat_coords = hal.prop(host_str+'distance')[hal_mask]#*ut.basic.constant.km_per_kpc
    position_masked, disk_mask = select_out_of_disk(sat_coords, host_axes_dict, 
        host_name, snapshot_index, return_mask=True, disk_mask_angle=disk_mask_angle)
    velocity = hal.prop(host_str+'velocity')[hal_mask]*ut.basic.constant.kpc_per_km
    velocity_masked = velocity[disk_mask]
    if norm is True:
        ang_momentum = np.array([np.cross(x,v)/np.linalg.norm(np.cross(x,v)) for x, v in zip(position_masked, velocity_masked)])
    elif norm is False:
        ang_momentum = np.array([np.cross(x,v) for x, v in zip(position_masked, velocity_masked)])

    return ang_momentum

def orbital_pole_dispersion(
    hal, hal_mask=None, host_str='host.', host_name=None, snapshot_index=None,
    host_axes_dict=None, disk_mask_angle=12.0):
    '''
    Calculate the angular dispersion [deg] of satellite orbital poles around
    their mean orbital pole.
    '''
    if np.sum(hal_mask) == 0:
        pole_disp = np.nan
        avg_j_vec = np.array([np.nan, np.nan, np.nan])
    else:
        j_vec = orbital_ang_momentum(hal, hal_mask, host_str, host_name, 
            snapshot_index, norm=True, host_axes_dict=host_axes_dict, disk_mask_angle=disk_mask_angle)
        avg_j_vec = np.mean(j_vec, axis=0, dtype=np.float64)/np.linalg.norm(np.mean(j_vec, axis=0))
        avg_j_dot_j = np.array([np.dot(avg_j_vec, j_vec_i) for j_vec_i in j_vec]) 
        pole_disp = np.sqrt(np.mean(np.arccos(avg_j_dot_j)**2, dtype=np.float64))
        pole_disp = np.degrees(pole_disp)

    return {'orbital.pole.dispersion':pole_disp, 'average.orbital.pole':avg_j_vec}