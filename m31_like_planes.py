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
from satellite_analysis import spatial as spa


def select_in_2d_projection(
    sat_coords_rot, sat_stellar_mass, rlim2d=150, n_sat=None, return_mask=False):
    # choose y axis to be along the line of sight
    # in keeping with rms height convention
    proj_2d = np.sqrt(sat_coords_rot[:,0]**2 + sat_coords_rot[:,2]**2)
    proj_mask = proj_2d <= rlim2d

    if n_sat is not None:
        if np.sum(proj_mask) > n_sat:
            base_ind = np.where(proj_mask)[0]
            base_sm = sat_stellar_mass[base_ind]
            top_n_base_sm_ind = np.argsort(base_sm)[-n_sat:]
            top_n_ind = base_ind[top_n_base_sm_ind]
            proj_mask = np.zeros(len(sat_stellar_mass), dtype=bool)
            proj_mask[top_n_ind] = True
        elif np.sum(proj_mask) < n_sat:
            proj_mask = np.zeros(len(sat_stellar_mass), dtype=bool)

    if return_mask:
        return sat_coords_rot[proj_mask], proj_mask
    else:
        return sat_coords_rot[proj_mask]

@jit
def rand_rms_min(
    hal, hal_mask=None, host_str='host.', n_iter=None, r_frac=None, 
    radius_bins=None, return_ax=False, return_parallel=False, projection=None,
    n_sat=None, return_dist=False):
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

    sat_star_mass = hal.prop('star.mass')[hal_mask]

    for n, rot_vec in enumerate(rot_vecs):
        sat_prime_coords = ut.basic.coordinate.get_coordinates_rotated(sat_coords, rotation_tensor=rot_vec)
        if projection is not None:
            sat_prime_coords = select_in_2d_projection(sat_prime_coords,
                                                        sat_star_mass, 
                                                        rlim2d=projection,
                                                        n_sat=n_sat)
            #sat_prime_coords = select_in_2d_projection(sat_prime_coords, rlim2d=projection)
        if len(sat_prime_coords) == 0:
            rms_minor_n[n] = np.nan
            rms_major_n[n] = np.nan
        else:
            rms_minor_n[n] = np.sqrt(np.nanmean(sat_prime_coords[:,2]**2))
            rms_major_n[n] = np.sqrt(np.nanmean(sat_prime_coords[:,0]**2))

    if return_ax:
        min_rms_minor = np.nanmin(rms_minor_n)
        min_index = np.where(rms_minor_n == np.nanmin(rms_minor_n))[0][0]
        rms_major = rms_major_n[min_index]
        # return just the vector normal to the plane
        min_ax = rot_vecs[min_index][2]
        return {'rms_minor':min_rms_minor, 'rms_major':rms_major, 'ax':min_ax}
    elif return_parallel:
        min_rms_minor = np.nanmin(rms_minor_n)
        min_index = np.where(rms_minor_n == np.nanmin(rms_minor_n))[0][0]
        rms_major = rms_major_n[min_index]
        return {'rms_minor':min_rms_minor, 'rms_major':rms_major}
    elif return_dist:
        return rms_minor_n
    else:
        min_rms_minor = np.nanmin(rms_minor_n)
        min_index = np.where(rms_minor_n == np.nanmin(rms_minor_n))[0][0]
        return min_rms_minor

def rand_angle(
    hal, hal_mask=None, host_str='host.', n_iter=1000, projection=None, n_sat=None):
    '''
    Calculates opening angles off of a set of randomly/isotropicallly generated
    axes (for sat.n_iter realizations).
    Returns opening angles, the vectors that the original axes are rotated by,
    and the rotation matrix that performs the rotations.
    '''
    sat_coords = hal.prop(host_str+'distance')[hal_mask]
    rot_vecs, rot_mats = ra.rand_rot_vec(n_iter)
    open_angle_n = []

    sat_star_mass = hal.prop('star.mass')[hal_mask]

    for n in range(n_iter):
        sat_prime_coords = ut.basic.coordinate.get_coordinates_rotated(sat_coords, rotation_tensor=rot_vecs[n])
        if projection is not None:
            sat_prime_coords = select_in_2d_projection(sat_prime_coords,
                                            sat_star_mass, 
                                            rlim2d=projection,
                                            n_sat=n_sat)
            #sat_prime_coords = select_in_2d_projection(sat_prime_coords, rlim2d=projection)
        tangent_of_open_angle = sat_prime_coords[:,2]/np.sqrt(sat_prime_coords[:,0]**2 + sat_prime_coords[:,1]**2)
        open_angle_n.append(np.degrees(np.arctan(tangent_of_open_angle)))

    return np.array(open_angle_n), rot_vecs, rot_mats

@jit
def rand_angle_width(
    hal, hal_mask=None, host_str='host.', n_iter=1000, fraction=1.0, 
    angle_range=None, return_ax=False, projection=None, n_sat=None):
    '''
    Rotates the 3D positions of the satellites randomly and uniformly for sat.n_iter
    realizations. Finds the angle off of the simulation x-axis that encloses a given
    fraction of the satellites, and then gets the minimum of this angle across all
    random realizations for each host halo at each redshift.
    '''
    rand_angles, rand_axes, rand_mats = rand_angle(hal, 
                                                   hal_mask=hal_mask, 
                                                   host_str=host_str, 
                                                   n_iter=n_iter, 
                                                   projection=projection,
                                                   n_sat=n_sat)
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
    hal, hal_mask=None, host_str='host.', return_ax=False, projection=None, 
    n_iter=1000, n_sat=None, return_dist=False):
    '''
    Get the axis ratio (minor/major) for the total distribution of satellites
    within the fiducial virial radius of the host halo.
    '''
    sat_coords = hal.prop(host_str+'distance')[hal_mask]
    rot_vecs, rot_mats = ra.rand_rot_vec(n_iter)
    axis_ratio_n = np.zeros(n_iter)

    sat_star_mass = hal.prop('star.mass')[hal_mask]

    for n in range(n_iter):
        sat_prime_coords = ut.basic.coordinate.get_coordinates_rotated(sat_coords, rotation_tensor=rot_vecs[n])
        if projection is not None:
            sat_prime_coords = select_in_2d_projection(sat_prime_coords,
                                            sat_star_mass, 
                                            rlim2d=projection,
                                            n_sat=n_sat)
            #sat_prime_coords = select_in_2d_projection(sat_prime_coords, rlim2d=projection)
        if len(sat_prime_coords) == 0:
            axis_ratio_n[n] = np.nan
        else:
            sat_axes = ut.coordinate.get_principal_axes(sat_prime_coords)
            axis_ratio_n[n] = sat_axes[1][0]
    if return_dist:
        return axis_ratio_n
    else:
        return np.min(axis_ratio_n)
"""
@jit
def get_satellite_principal_axes(
    hal, hal_mask=None, host_str='host.', mass_kind=None, projection=None):
    '''
    Find the inertia tensor and maj/med/min axes of the satellite distribution
    (hal) that meets the criteria of (hal_mask).
    '''
    distance_vectors = hal.prop(host_str+'distance')[hal_mask]
    if projection is not None:
        distance_vectors = select_in_2d_projection(distance_vectors, rlim2d=projection)

    moi_quantities = ut.coordinate.get_principal_axes(distance_vectors)

    return moi_quantities
"""
def rand_los_vel_coherence(
    hal, hal_mask=None, host_str='host.', n_iter=1000, projection=None, 
    n_sat=None, return_dist=False):
    """
    Find maximum fraction of satellites with correlated LOS velocities along
    n_iter different lines of sight and return maximum fraction.
    """
    sat_vels = hal.prop(host_str+'velocity')[hal_mask]
    sat_coords = hal.prop(host_str+'distance')[hal_mask]
    rot_vecs, rot_mats = ra.rand_rot_vec(n_iter)
    coherent_frac_n = np.zeros(n_iter)
    rms_minor_n = np.zeros(n_iter)

    sat_star_mass = hal.prop('star.mass')[hal_mask]
    num_of_sats = np.zeros(n_iter)
    min_sm_of_sats = np.zeros(n_iter)

    for n, rot_vec in enumerate(rot_vecs):
        # rotate positions and velocities
        sat_prime_coords = ut.basic.coordinate.get_coordinates_rotated(sat_coords, rotation_tensor=rot_vec)
        sat_prime_vels = ut.basic.coordinate.get_coordinates_rotated(sat_vels, rotation_tensor=rot_vec)
        if projection is not None:
            sat_prime_coords, proj_2d_mask = select_in_2d_projection(
                                            sat_prime_coords,
                                            sat_star_mass, 
                                            rlim2d=projection,
                                            return_mask=True,
                                            n_sat=n_sat)
            sat_prime_vels = sat_prime_vels[proj_2d_mask]
            #num_of_sats[n] = np.sum(proj_2d_mask)
            #min_sm_of_sats[n] = np.min(sat_star_mass[proj_2d_mask])

        # get fraction of satellites with coherence in LOS velocity and
        # rms height for each random set of axes
        if len(sat_prime_coords) == 0:
            coherent_frac_n[n] = np.nan
        else:
            coherent_frac_n, rms_minor_n = optim_los_vel_coherence(
                sat_prime_coords, sat_prime_vels, coherent_frac_n, rms_minor_n, n)

    # print statements to get minimum numbers and stellar masses of satellites
    # under the different M31-like selection functions
    #print('num of sats:', np.min(num_of_sats), np.median(num_of_sats), np.max(num_of_sats))
    #print('min stellar mass of sats:', np.min(min_sm_of_sats))

    # why am I selecting a fraction based on min rms height here? probably b/c
    # the observations are predicated on seeing M31's sat plane edge-on?
    #min_rms_minor = np.min(rms_minor_n)
    #min_rms_index = np.where(rms_minor_n == np.min(rms_minor_n))[0][0]

    #return coherent_frac_n[min_rms_index]
    if return_dist:
        return coherent_frac_n
    else:
        return np.max(coherent_frac_n)

@jit(nopython=True)
def optim_los_vel_coherence(sat_coords, sat_vels, coherent_frac, rms_minor, i):
    # find 2D co-rotating fraction at each iteration
    # y axis [1] is along LOS, positive or 0 y velocities are "approaching"
    nsat = sat_coords[:,0].size
    left_sats = sat_coords[:,0] >= 0
    right_sats = sat_coords[:,0] < 0
    approaching = sat_vels[:,1] >= 0
    receding = sat_vels[:,1] < 0
    fracs = np.zeros(2)
    fracs[0] = np.sum((left_sats & approaching)|(right_sats & receding))/nsat
    fracs[1] = np.sum((left_sats & receding)|(right_sats & approaching))/nsat
    assert np.sum(fracs) == 1.0
    coherent_frac[i] = np.max(fracs)
    rms_minor[i] = np.sqrt(np.mean(sat_coords[:,2]**2))

    return coherent_frac, rms_minor

# included as a test, average isotropic fraction is around 0.65-0.68 for
# 'most.massive' mask, and ~0.6 for 'mass.peak' mask (> 8e8 Msun), so as number of
# satellites increases, I think it's getting closer to the expected 0.5
# right around 0.5-0.55 for mpeak > 1e8, as expected
@jit
def iso_rand_los_vel_coherence(iso_hal, n_iter=1000, projection=None):
    """
    Find maximum co-rotating fraction of isotropic satellite velocities.
    """
    rot_vecs, rand_matrices = ra.rand_rot_vec(n_iter)
    coherent_frac_n = np.zeros(n_iter)
    rms_minor_n = np.zeros(n_iter)

    for n in range(n_iter):
        iso_coords = iso_hal['iso_coords'][n]
        iso_vels = iso_hal['iso_vels'][n]
        coherent_frac_k = np.zeros(n_iter)
        rms_minor_k = np.zeros(n_iter)

        for k, rot_vec in enumerate(rot_vecs):
            # rotate positions and velocities
            sat_prime_coords = ut.basic.coordinate.get_coordinates_rotated(iso_coords, rotation_tensor=rot_vec)
            sat_prime_vels = ut.basic.coordinate.get_coordinates_rotated(iso_vels, rotation_tensor=rot_vec)
            if projection is not None:
                sat_prime_coords, proj_2d_mask = select_in_2d_projection(sat_prime_coords, rlim2d=projection, return_mask=True)
                sat_prime_vels = sat_prime_vels[proj_2d_mask]

            # find 2D co-rotating fraction at each iteration
            coherent_frac_k, rms_minor_k = optim_los_vel_coherence(
                sat_prime_coords, sat_prime_vels, coherent_frac_k, rms_minor_k, k)

        rms_minor_n[n] = np.min(rms_minor_k)
        min_rms_index = np.where(rms_minor_k == np.min(rms_minor_k))[0][0]
        coherent_frac_n[n] = coherent_frac_k[min_rms_index]

    return {'coherent.fraction':np.mean(coherent_frac_n), 'rms':np.mean(rms_minor_n)}