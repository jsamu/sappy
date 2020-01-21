import numpy as np
from numba import jit
import utilities as ut
from satellite_analysis import spatial as spa
from satellite_analysis import angular as ang
from satellite_analysis import math_funcs as mf

@jit
def rand_rot_vec(n_iter):
    '''
    Generate three random rotation vectors for n_iter realizations.
    '''
    rot_vecs = np.zeros((n_iter,3,3))
    rot_mats = np.zeros((n_iter,3,3))
    xn_3d = np.random.rand(n_iter,3)

    for n in range(n_iter):
        rot = rand_rot_matrix(xn_3d[n])
        rot_mats[n] = rot
        rot_vecs[n] = rot.T

    return rot_vecs, rot_mats

@jit
def rand_rot_matrix(rand_3d_array):
    '''
    Generate a random and uniform rotation matrix. Cite:
    http://blog.lostinmyterminal.com/python/2015/05/12/random-rotation-matrix.html
    '''
    theta, phi, z = rand_3d_array
    
    theta = theta*2.0*np.pi  # Rotation about the pole (Z).
    phi = phi*2.0*np.pi  # For direction of pole deflection.
    z = z*2.0  # For magnitude of pole deflection.
    
    # Compute a vector V used for distributing points over the sphere
    # via the reflection I - V Transpose(V).  This formulation of V
    # will guarantee that if x[1] and x[2] are uniformly distributed,
    # the reflected points will be uniform on the sphere.  Note that V
    # has length sqrt(2) to eliminate the 2 in the Householder matrix.
    
    r = np.sqrt(z)
    Vx, Vy, Vz = V = (
        np.sin(phi) * r,
        np.cos(phi) * r,
        np.sqrt(2.0 - z)
        )
    
    st = np.sin(theta)
    ct = np.cos(theta)
    
    R = np.array(((ct, st, 0), (-st, ct, 0), (0, 0, 1)))
    
    # Construct the rotation matrix  ( V Transpose(V) - I ) R.
    M = (np.outer(V, V) - np.eye(3)).dot(R)

    return M

def rand_angle(hal, hal_mask=None, host_str='host.', n_iter=1000):
    '''
    Calculates opening angles off of a set of randomly/isotropicallly generated
    axes (for sat.n_iter realizations).
    Returns opening angles, the vectors that the original axes are rotated by,
    and the rotation matrix that performs the rotations.
    '''
    sat_coords = hal.prop(host_str+'distance')[hal_mask]
    rot_vecs, rot_mats = rand_rot_vec(n_iter)
    open_angle_n = []

    for n in range(n_iter):
        sat_prime_coords = ut.basic.coordinate.get_coordinates_rotated(sat_coords, rotation_tensor=rot_vecs[n])
        tangent_of_open_angle = sat_prime_coords[:,2]/np.sqrt(sat_prime_coords[:,0]**2 + sat_prime_coords[:,1]**2)
        open_angle_n.append(np.degrees(np.arctan(tangent_of_open_angle)))

    return np.array(open_angle_n), rot_vecs, rot_mats

@jit
def rand_angle_width(
    hal, hal_mask=None, host_str='host.', n_iter=1000, fraction=1.0, 
    angle_range=None, return_ax=False):
    '''
    Rotates the 3D positions of the satellites randomly and uniformly for sat.n_iter
    realizations. Finds the angle off of the simulation x-axis that encloses a given
    fraction of the satellites, and then gets the minimum of this angle across all
    random realizations for each host halo at each redshift.
    '''
    rand_angles, rand_axes, rand_mats = rand_angle(hal, hal_mask=hal_mask, host_str=host_str, n_iter=n_iter)
    phi_width_n = np.zeros(n_iter)

    for n, snap_angles_n in enumerate(rand_angles):
        if snap_angles_n.size == 0:
            phi_width_n[n] = np.nan
        else:
            phi_width_n = optim_open_angle(snap_angles_n, angle_range, fraction, phi_width_n, n)

    phi_width = np.min(phi_width_n)
    min_index = np.where(phi_width_n == np.min(phi_width_n))[0][0]

    # return just the vector normal to the plane
    min_ax = rand_axes[min_index][2]

    if return_ax is True:
        return {'angle':phi_width, 'ax':min_ax}
    else:
        return phi_width

@jit(nopython=True)
def optim_open_angle(snap_angles, angle_range, threshold_fraction, phi_width, i):
    for opening_angle in angle_range:
        angle_mask = (snap_angles <= opening_angle) & (snap_angles >= -opening_angle)
        frac_enclosed = np.sum(angle_mask)/snap_angles.size
        
        if frac_enclosed >= threshold_fraction:
            phi_width[i] = 2*opening_angle
            break
        else:
            pass

    return phi_width

@jit
def rand_rms_min(
    hal, hal_mask=None, host_str='host.', n_iter=None, r_frac=None, 
    radius_bins=None, return_ax=False, return_parallel=False):
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
    rot_vecs, rot_mats = rand_rot_vec(n_iter)
    rms_minor_n = np.zeros(n_iter)
    rms_major_n = np.zeros(n_iter)

    for n, rot_vec in enumerate(rot_vecs):
        sat_prime_coords = ut.basic.coordinate.get_coordinates_rotated(sat_coords, rotation_tensor=rot_vec)
        rms_minor_n[n] = np.sqrt(np.nanmean(sat_prime_coords[:,2]**2))
        rms_major_n[n] = np.sqrt(np.nanmean(sat_prime_coords[:,0]**2))

    min_rms_minor = np.min(rms_minor_n)
    #min_rms_major = np.min(rms_major_n)
    min_index = np.where(rms_minor_n == np.min(rms_minor_n))[0][0]
    if return_ax is True:
        rms_major = rms_major_n[min_index]
        # return just the vector normal to the plane
        min_ax = rot_vecs[min_index][2]
        #min_mat = rot_mats[min_index]
        return {'rms_minor':min_rms_minor, 'rms_major':rms_major, 'ax':min_ax}
    elif return_parallel is True:
        rms_major = rms_major_n[min_index]
        return {'rms_minor':min_rms_minor, 'rms_major':rms_major}
    else:
        return min_rms_minor

def rand_frac_open_angle(hal, hal_mask=None, angle_bins=None, n_iter=1000):
    '''
    Calculate the fraction of satellites enclosed at each angular bin in sat.a_bins
    for the isotropic distribution of each snapshot.
    '''
    sat_coords = hal.prop('host.distance')[hal_mask]
    rot_vecs, rot_mats = rand_rot_vec(n_iter)

    frac_enclosed_n = np.zeros((n_iter,len(angle_bins)))

    for n in range(n_iter):
        iter_frac_enclosed = []
        #iter_angles = iso_hal['iso_angles'][n]
        sat_prime_coords = ut.basic.coordinate.get_coordinates_rotated(sat_coords, rotation_tensor=rot_vecs[n])
        tangent_of_open_angle = sat_prime_coords[:,2]/np.sqrt(sat_prime_coords[:,0]**2 + sat_prime_coords[:,1]**2)
        #open_angle_n.append(np.degrees(np.arctan(tangent_of_open_angle)))
        iter_angles = np.degrees(np.arctan(tangent_of_open_angle))

        for open_angle in angle_bins:
            angle_mask = (iter_angles <= open_angle) & (iter_angles >= -open_angle)
            frac_enclosed = float(len(iter_angles[angle_mask]))/float(len(iter_angles))
            iter_frac_enclosed.append(frac_enclosed)

        frac_enclosed_n[n] = iter_frac_enclosed

    frac_enclosed_max = np.max(frac_enclosed_n, axis=0)

    return frac_enclosed_max

def old_get_ax_diff(sat):
    '''
    Checks the difference between the intertia tensor axes and the minimized
    opening angle axes by taking the difference of the Euler angles used to
    rotate to each frame from the simulation frame.
    '''
    rand_phi, rand_min_matrix, rand_min_axes = rand_angle_width(sat)
    open_angles, moi_rot_mat = ang.open_angle(sat, return_matrix=True)
    all_ang_diffs = []

    for host_moi, host_min in zip(moi_rot_mat, rand_min_matrix):
        host_ang_diffs = []
        for snap_moi, snap_min in zip(host_moi, host_min):
            moi_euler_angs = mf.rotationMatrixToEulerAngles(snap_moi)
            min_euler_angs = mf.rotationMatrixToEulerAngles(snap_min)
            diff_euler_angs = moi_euler_angs - min_euler_angs
            host_ang_diffs.append(diff_euler_angs)
        all_ang_diffs.append(host_ang_diffs)

    return all_ang_diffs
