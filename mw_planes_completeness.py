import numpy as np
import utilities as ut
from numba import jit
from satellite_analysis import satellite_io as sio
from satellite_analysis import rand_axes as ra
from satellite_analysis import spatial as spa


def select_out_of_disk(
    sat_coords, disk_axes=None, disk_mask_angle=12.0, return_mask=False):
    # cut out satellites that lie within +- 12 degrees of the simulated MW disk
    sat_prime_coords = ut.basic.coordinate.get_coordinates_rotated(sat_coords, rotation_tensor=disk_axes)
    tangent_of_open_angle = sat_prime_coords[:,2]/np.sqrt(sat_prime_coords[:,0]**2 + sat_prime_coords[:,1]**2)
    disk_mask = np.abs(np.degrees(np.arctan(tangent_of_open_angle))) < disk_mask_angle

    if return_mask:
        return sat_coords[disk_mask], disk_mask
    else:
        return sat_coords[disk_mask]

@jit
def rand_rms_min(
    hal, hal_mask=None, host_str='host.', n_iter=None, r_frac=None, 
    radius_bins=None, return_ax=False, return_parallel=False, disk_axes=None):
    '''
    Rotates the 3D positions of the satellites randomly and uniformly for sat.n_iter
    realizations. Finds the rms along the z-axis that encloses the specified
    fraction of satellites, and then gets the minimum of the rms values across all
    random realizations.
    '''
    if r_frac is None:
        sat_coords = hal.prop(host_str+'distance')[hal_mask]
    else:
        rad = spa.r_fraction(hal, hal_mask=hal_mask, host_str=host_str, frac=r_frac, radius_bins=radius_bins)
        radial_mask = hal.prop(host_str+'distance.total') <= rad
        sat_coords = hal.prop(host_str+'distance')[hal_mask & radial_mask]
    rot_vecs, rot_mats = ra.rand_rot_vec(n_iter)
    rms_minor_n = np.zeros(n_iter)
    rms_major_n = np.zeros(n_iter)

    sat_coords = select_out_of_disk(sat_coords, disk_axes=disk_axes)

    for n, rot_vec in enumerate(rot_vecs):
        sat_prime_coords = ut.basic.coordinate.get_coordinates_rotated(sat_coords, rotation_tensor=rot_vec)
        rms_minor_n[n] = np.sqrt(np.nanmean(sat_prime_coords[:,2]**2))
        rms_major_n[n] = np.sqrt(np.nanmean(sat_prime_coords[:,0]**2))

    min_rms_minor = np.min(rms_minor_n)
    min_index = np.where(rms_minor_n == np.min(rms_minor_n))[0][0]
    if return_ax is True:
        rms_major = rms_major_n[min_index]
        # return just the vector normal to the plane
        min_ax = rot_vecs[min_index][2]
        return {'rms_minor':min_rms_minor, 'rms_major':rms_major, 'ax':min_ax}
    elif return_parallel is True:
        rms_major = rms_major_n[min_index]
        return {'rms_minor':min_rms_minor, 'rms_major':rms_major}
    else:
        return min_rms_minor

def rand_angle(hal, hal_mask=None, host_str='host.', n_iter=1000, disk_axes=None):
    '''
    Calculates opening angles off of a set of randomly/isotropicallly generated
    axes (for sat.n_iter realizations).
    Returns opening angles, the vectors that the original axes are rotated by,
    and the rotation matrix that performs the rotations.
    '''
    sat_coords = hal.prop(host_str+'distance')[hal_mask]
    rot_vecs, rot_mats = ra.rand_rot_vec(n_iter)
    open_angle_n = []

    sat_coords = select_out_of_disk(sat_coords, disk_axes=disk_axes)

    for n in range(n_iter):
        sat_prime_coords = ut.basic.coordinate.get_coordinates_rotated(sat_coords, rotation_tensor=rot_vecs[n])
        tangent_of_open_angle = sat_prime_coords[:,2]/np.sqrt(sat_prime_coords[:,0]**2 + sat_prime_coords[:,1]**2)
        open_angle_n.append(np.degrees(np.arctan(tangent_of_open_angle)))

    return np.array(open_angle_n), rot_vecs, rot_mats

@jit
def rand_angle_width(
    hal, hal_mask=None, host_str='host.', n_iter=1000, fraction=1.0, 
    angle_range=None, return_ax=False, disk_axes=None):
    '''
    Rotates the 3D positions of the satellites randomly and uniformly for sat.n_iter
    realizations. Finds the angle off of the simulation x-axis that encloses a given
    fraction of the satellites, and then gets the minimum of this angle across all
    random realizations for each host halo at each redshift.
    '''
    rand_angles, rand_axes, rand_mats = rand_angle(hal, hal_mask=hal_mask, 
        host_str=host_str, n_iter=n_iter, disk_axes=disk_axes)
    phi_width_n = np.zeros(n_iter)

    for n, snap_angles_n in enumerate(rand_angles):
        if snap_angles_n.size == 0:
            phi_width_n[n] = np.nan
        else:
            phi_width_n = ra.optim_open_angle(snap_angles_n, angle_range, fraction, phi_width_n, n)

    phi_width = np.min(phi_width_n)
    min_index = np.where(phi_width_n == np.min(phi_width_n))[0][0]

    # return just the vector normal to the plane
    min_ax = rand_axes[min_index][2]

    if return_ax is True:
        return {'angle':phi_width, 'ax':min_ax}
    else:
        return phi_width

@jit
def axis_ratio(
    hal, hal_mask=None, host_str='host.', return_ax=False, disk_axes=None):
    '''
    Get the axis ratio (minor/major) for the disk-masked distribution of 
    satellites.
    '''
    sat_coords = hal.prop(host_str+'distance')[hal_mask]
    sat_masked_coords = select_out_of_disk(sat_coords, disk_axes=disk_axes)
    sat_axes = ut.coordinate.get_principal_axes(sat_masked_coords)

    return sat_axes[2][0]
