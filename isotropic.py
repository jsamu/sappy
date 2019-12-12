import numpy as np
from numba import jit
import utilities as ut
from satellite_analysis import spatial as spa
from satellite_analysis import math_funcs as mf
from satellite_analysis import rand_axes as ra


def iso_distribution(hal, hal_mask=None, host_str='host.', n_iter=1000):
    '''
    Returns n_iter realizations of isotropic positions (keeping the true radial
    distribution/randomizing angles only), opening angles off of the MOI major axis,
    isotropic (unit) velocities, and the radial distribution of each snapshot.
    '''
    sat_coords = hal.prop(host_str+'distance')[hal_mask]

    # get radial distances
    radial_distance = np.array(hal.prop(host_str+'distance.total')[hal_mask])

    # generate isotropic (at fixed radial distribution) coordinates and unit velocities
    iso_coords, iso_vels = iso_iter(radial_distance, n_iter)

    # get opening angles wrt MOI axes
    sat_axes = spa.get_satellite_principal_axes(hal, hal_mask)
    rot_vec = sat_axes[0]
    snap_prime_coords = ut.basic.coordinate.get_coordinates_rotated(iso_coords, rotation_tensor=rot_vec)
    tangent_of_openning_angle = snap_prime_coords[:,:,2]/np.sqrt(snap_prime_coords[:,:,0]**2 + snap_prime_coords[:,:,1]**2)
    iso_open_angles = np.degrees(np.arctan(tangent_of_openning_angle))

    return {'iso_coords':iso_coords, 'iso_angles':iso_open_angles, 
            'iso_vels':iso_vels, 'iso_radii':radial_distance, 'true_coords':sat_coords}

@jit(nopython=True)
def iso_iter(radial_distance, n_iter):
    '''
    Generate n_iter realizations of isotropic coordinates (keeping the true radial
    distances/randomizing angles only) and isotropic (unit) velocities.
    '''
    r_size = radial_distance.size
    iter_iso_coord_list = np.zeros((n_iter, r_size, 3))
    iter_iso_vel_list = np.zeros((n_iter, r_size, 3))

    for n in range(n_iter):
        # generate random angular positions at fixed radial distribution
        rand_phi = np.random.uniform(0, 2*np.pi, size=r_size)
        rand_theta = np.arccos(np.random.uniform(-1, 1, size=r_size))
        iso_coord_list = np.zeros((r_size, 3))

        # generate random unit velocities
        phi_v = np.random.uniform(0, 2*np.pi, size=r_size)
        theta_v = np.arccos(np.random.uniform(-1, 1, size=r_size))
        iso_vel_list = np.zeros((r_size, 3))

        for k, r in enumerate(radial_distance):
            x = r * np.sin(rand_theta[k])*np.cos(rand_phi[k])
            y = r * np.sin(rand_theta[k])*np.sin(rand_phi[k])
            z = r * np.cos(rand_theta[k])
            iso_coord_list[k] = np.array([x, y, z])

            vx = np.sin(theta_v[k])*np.cos(phi_v[k])
            vy = np.sin(theta_v[k])*np.sin(phi_v[k])
            vz = np.cos(theta_v[k])
            iso_vel_list[k] = np.array([vx, vy, vz])

        iter_iso_coord_list[n] = iso_coord_list
        iter_iso_vel_list[n] = iso_vel_list

    return iter_iso_coord_list, iter_iso_vel_list

@jit
def iso_frac_open_angle(iso_hal, angle_bins=None):
    '''
    Calculate the fraction of satellites enclosed at each angular bin in sat.a_bins
    for the isotropic distribution of each snapshot.
    '''
    all_iters = np.zeros((len(iso_hal['iso_angles']), angle_bins.size))

    for n in range(len(iso_hal['iso_angles'])):
        iter_frac_enclosed = []
        iter_angles = iso_hal['iso_angles'][n]

        for open_angle in angle_bins:
            angle_mask = (iter_angles <= open_angle) & (iter_angles >= -open_angle)
            frac_enclosed = float(len(iter_angles[angle_mask]))/float(len(iter_angles))
            iter_frac_enclosed.append(frac_enclosed)

        all_iters[n] = iter_frac_enclosed
    
    snap_frac_enclosed = np.average(all_iters, axis=0)

    return snap_frac_enclosed

@jit
def iso_angle_width(iso_hal, threshold_fraction=0.68, angle_bins=None):
    '''
    Finds the angle that encloses a given fraction of the isotropic satellite
    opening angles at each redshift and for each host halo.
    '''
    iter_width = np.zeros(len(iso_hal['iso_angles']))

    for n in range(len(iso_hal['iso_angles'])):
        iter_angles = iso_hal['iso_angles'][n]

        for open_angle in angle_bins:
            angle_mask = (iter_angles <= open_angle) & (iter_angles >= -open_angle)
            frac_enclosed = float(len(iter_angles[angle_mask]))/float(len(iter_angles))
            
            if frac_enclosed >= threshold_fraction:
                iter_width[n] = 2*open_angle
                break
            else:
                pass

    #phi_width = np.average(iter_width)

    return iter_width

@jit
def iso_axis_ratio(iso_hal, distribution=True):
    '''
    Calculate the minor to major axis ratio using the MOI tensor at each redshift
    for each host halo's isotropic satellite coordinates.
    '''
    iter_ratios = np.zeros(len(iso_hal['iso_coords']))

    for n in range(len(iso_hal['iso_coords'])):
        coords = iso_hal['iso_coords'][n]
        sat_axes = ut.coordinate.get_principal_axes(coords)
        iter_ratios[n] = sat_axes[2][0]

    if distribution:
        return iter_ratios
    else:
        return np.average(iter_ratios)

@jit
def iso_r_fraction(iso_hal, frac=0.68, radius_bins=None):
    '''
    Find the radius (r) that encloses the given fraction of satellites (frac).
    Form a list of these radii for each redshift (host_r) for each host (r_all).
    '''
    radii = iso_hal['iso_radii']

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

@jit
def iso_rms_vs_r_frac(iso_hal, r_frac=None, radius_bins=None):
    rads = iso_r_fraction(iso_hal, frac=r_frac, radius_bins=radius_bins)
    rms_major = []
    rms_minor = []
    n_iter = len(iso_hal['iso_coords'])
    iter_rms_x = np.zeros(n_iter)
    iter_rms_z = np.zeros(n_iter)

    radial_mask = iso_hal['iso_radii'] <= rads
    for n in range(n_iter):
        sat_coords = iso_hal['iso_coords'][n][radial_mask]
        iter_rms_x[n] = spa.rms_maj(sat_coords)
        iter_rms_z[n] = spa.rms_min(sat_coords)

    rms_major = np.average(iter_rms_x)
    rms_minor = np.average(iter_rms_z)

    return {'rmsx':rms_major, 'rmsz':rms_minor, 'radii':rads}

# this will need to be run with loop_sat for now, change to loop_iso if possible later
@jit
def iso_rms_min_vs_r(sat, hal_name, redshift_index):
    '''
    Find the rms (height and 'radius') of isotropic satellites as a function of
    radius. 
    '''
    hal = sat.hal_catalog[hal_name][redshift_index]
    main_mask = sat.catalog_mask[hal_name][redshift_index]

    host_hals = sat.hal_catalog[hal_name]
    host_masks = sat.catalog_mask[hal_name]
    rad_min = max([np.sort(host_hals[k].prop('host.distance.total')[host_masks[k]])[2] for k in range(len(sat.redshift))])
    rad_min_round = mf.round_up(rad_min, base=25)
    rads = np.arange(rad_min_round, sat.r_range[1]+25, 25)

    iter_rms_z = np.zeros((sat.n_iter, rads.size))

    for n in range(sat.n_iter):
        r_rmsz = []
        iso_radii = sat.isotropic[hal_name][redshift_index]['iso_radii']
        iso_coords = sat.isotropic[hal_name][redshift_index]['iso_coords'][n]

        for r in rads:
            radial_mask = iso_radii <= r
            sat_coords = iso_coords[radial_mask]
            r_rmsz.append(spa.rms_min(sat_coords))

        iter_rms_z[n] = r_rmsz

    snap_rms_minor = np.average(iter_rms_z, axis=0)

    return {'rmsz':snap_rms_minor, 'radii':rads}

@jit
def iso_open_angle_v_r(iso_hal, threshold_fraction=0.5, radius_bins=None, angle_bins=None):
    '''
    Calculates the angle and radius that enclose a given fraction of the satellites.
    '''
    radii = iso_hal['iso_radii']
    open_angles = iso_hal['iso_angles']

    for k in radius_bins:
        radius_mask = radii <= k
        frac_enclosed = float(len(radii[radius_mask]))/float(len(radii))
        
        if frac_enclosed >= threshold_fraction:
            hal_rad = k
            iter_snap_angle = []

            for m in range(len(open_angles)):
                snap_angles = open_angles[m]
                r_mask_angles = snap_angles[radius_mask]

                for a in angle_bins:
                    angle_mask = (r_mask_angles <= a) & (r_mask_angles >= -a)
                    frac_enc = float(len(r_mask_angles[angle_mask]))/float(len(r_mask_angles))

                    if frac_enc == 1:
                        iter_snap_angle.append(2*a)
                        break

                    else:
                        pass
            
            hal_angle = np.mean(iter_snap_angle)
            break

        else:
            pass

    return {'radii':hal_rad, 'angles':hal_angle}

@jit
def iso_rand_angle_width_old(
    iso_hal, threshold_fraction=1.0, n_iter=None, angle_range=None,
    angle_width=None):
    '''
    Rotates the 3D positions of the satellites randomly and uniformly for sat.n_iter
    realizations. Finds the angle off of the simulation x-axis that encloses a given
    fraction of the satellites, and then gets the minimum of this angle across all
    random realizations for each host halo at each redshift for sat.n_iter realizations
    of isotropic satellite positions.
    '''
    rand_axes, rand_matrices = ra.rand_rot_vec(n_iter)
    phi_width_n = np.zeros(n_iter)

    for n in range(int(n_iter)):
        snap_coords = iso_hal['iso_coords'][n]
        if len(snap_coords) == 0:
            phi_width_n[n] = np.nan
        else:
            phi_width_k = np.zeros(n_iter)
            for k,axes in enumerate(rand_axes):
                snap_prime_coords = ut.basic.coordinate.get_coordinates_rotated(snap_coords, rotation_tensor=axes)
                tangent_of_openning_angle = snap_prime_coords[:,2]/np.sqrt(snap_prime_coords[:,0]**2 + snap_prime_coords[:,1]**2)
                snap_angles_n = np.degrees(np.arctan(tangent_of_openning_angle))
                phi_width_k = ra.optim_open_angle(snap_angles_n, angle_range, threshold_fraction, phi_width_k, k)
            phi_width_n[n] = np.min(phi_width_k)

    return phi_width_n

@jit
def iso_rand_angle_width(
    iso_hal, threshold_fraction=1.0, n_iter=None, angle_range=None,
    angle_width=None):
    '''
    Rotates the 3D positions of the satellites randomly and uniformly for sat.n_iter
    realizations. Finds the angle off of the simulation x-axis that encloses a given
    fraction of the satellites, and then gets the minimum of this angle across all
    random realizations for each host halo at each redshift for sat.n_iter realizations
    of isotropic satellite positions.
    '''
    rand_axes, rand_matrices = ra.rand_rot_vec(n_iter)

    # variables defined for optimization
    n_sat = len(iso_hal['iso_coords'][0])
    all_snap_coords = np.reshape(iso_hal['iso_coords'], (n_iter*n_sat, 3))
    frac_enclosed_range = np.zeros((len(angle_range), n_iter))
    nan_array = np.full(frac_enclosed_range.shape, np.nan)
    angle_array = np.zeros(frac_enclosed_range.shape)
    for j,opening_angle in enumerate(angle_range):
        angle_array[j] = np.full(n_iter, opening_angle)

    phi_width_k = np.zeros((n_iter, n_iter))
    for k,axes in enumerate(rand_axes):
        snap_prime_coords = ut.basic.coordinate.get_coordinates_rotated(all_snap_coords, rotation_tensor=axes)
        tangent_of_openning_angle = snap_prime_coords[:,2]/np.sqrt(snap_prime_coords[:,0]**2 + snap_prime_coords[:,1]**2)
        snap_angles_k = np.degrees(np.arctan(tangent_of_openning_angle))
        snap_angles_n = np.reshape(snap_angles_k, (n_iter, n_sat))
        phi_width_k[k] = optim_open_angle(snap_angles_n, angle_range, 
            threshold_fraction, n_sat, frac_enclosed_range, nan_array, angle_array)

    phi_width_n = np.min(phi_width_k, axis=0)

    return phi_width_n

@jit(nopython=True)
def optim_open_angle(
    snap_angles, angle_range, threshold_fraction, n_sat, frac_enclosed_range, nan_array, angle_array):
    for j,opening_angle in enumerate(angle_range):
        angle_mask = np.abs(snap_angles) <= opening_angle
        frac_enclosed = np.sum(angle_mask, axis=1)/n_sat
        frac_enclosed_range[j] = frac_enclosed

    min_encl_angles = np.where(frac_enclosed_range>=threshold_fraction, angle_array, nan_array)
    phi_width = 2*np.nanmin(min_encl_angles, axis=0)

    return phi_width

@jit
def optim_open_angle_old(
    snap_angles, angle_range, threshold_fraction, n_sat, frac_enclosed_range, nan_array, angle_array):
    frac_enclosed = np.array([0])
    for j,opening_angle in enumerate(angle_range):
        angle_mask = (snap_angles <= opening_angle) & (snap_angles >= -opening_angle)
        frac_enclosed = np.sum(angle_mask, axis=1)/n_sat
        frac_enclosed_range[j] = frac_enclosed

    min_encl_angles = np.where(frac_enclosed_range>=threshold_fraction, angle_array, nan_array)
    phi_width = 2*np.nanmin(min_encl_angles, axis=0)

    return phi_width

def orbital_pole_dispersion(iso_hal, n_iter=None):
    '''
    Calculate the angular dispersion [deg] of satellite orbital poles around
    their mean orbital pole for isotropic unit velocities."
    '''
    pole_disp_n = np.zeros(n_iter)
    for n in range(n_iter):
        iso_vels = iso_hal['iso_vels'][n]
        iso_coords = iso_hal['iso_coords'][n]
        if len(iso_vels) == 0:
            pole_disp_n[n] = np.nan
        else:
            j_vec = np.array([np.cross(x,v)/np.linalg.norm(np.cross(x,v)) for x, v in zip(iso_coords, iso_vels)])
            avg_j_vec = np.mean(j_vec, axis=0, dtype=np.float64)/np.linalg.norm(np.mean(j_vec, axis=0))
            avg_j_dot_j = np.array([np.dot(avg_j_vec, j_vec_i) for j_vec_i in j_vec]) 
            pole_disp = np.sqrt(np.mean(np.arccos(avg_j_dot_j)**2, dtype=np.float64))
            pole_disp_n[n] = np.degrees(pole_disp)

    return pole_disp_n

def orbital_pole_dispersion_old(iso_hal, n_iter=None):
    '''
    Calculate the angular dispersion [deg] of satellite orbital poles around
    their mean orbital pole for isotropic unit velocities."
    '''
    pole_disp_n = np.zeros(n_iter)
    #avg_pole_n = np.zeros((n_iter, 3))
    true_coords = iso_hal['true_coords']

    for n in range(n_iter):
        iso_vels = iso_hal['iso_vels'][n]
        if len(iso_vels) == 0:
            pole_disp_n[n] = np.nan
            #avg_pole_n[n] = np.array([np.nan, np.nan, np.nan])
        else:
            j_vec = np.array([np.cross(x,v)/np.linalg.norm(np.cross(x,v)) for x, v in zip(true_coords, iso_vels)])
            avg_j_vec = np.mean(j_vec, axis=0, dtype=np.float64)/np.linalg.norm(np.mean(j_vec, axis=0))
            avg_j_dot_j = np.array([np.dot(avg_j_vec, j_vec_i) for j_vec_i in j_vec]) 
            pole_disp = np.sqrt(np.mean(np.arccos(avg_j_dot_j)**2, dtype=np.float64))
            pole_disp_n[n] = np.degrees(pole_disp)
            #avg_pole_n[n] = avg_j_vec
    #return {'orbital.pole.dispersion':pole_disp_n, 'average.orbital.pole':avg_pole_n}
    return pole_disp_n

@jit
def iso_rand_los_vel_coherence(iso_hal, n_iter=None):
    """
    Find maximum co-rotating fraction of isotropic satellite velocities.
    """
    rot_vecs, rand_matrices = ra.rand_rot_vec(n_iter)
    coherent_frac_n = np.zeros(n_iter)
    rms_minor_n = np.zeros(n_iter)
    true_coords = iso_hal['true_coords']

    for n in range(n_iter):
        iso_vels = iso_hal['iso_vels'][n]
        coherent_frac_k = np.zeros(n_iter)
        rms_minor_k = np.zeros(n_iter)

        for k, rot_vec in enumerate(rot_vecs):
            # rotate positions and velocities
            sat_prime_coords = ut.basic.coordinate.get_coordinates_rotated(true_coords, rotation_tensor=rot_vec)
            sat_prime_vels = ut.basic.coordinate.get_coordinates_rotated(iso_vels, rotation_tensor=rot_vec)

            # find 2D co-rotating fraction at each iteration
            # x axis [0] is along LOS, y axis [1] determines left or right side
            coherent_frac_k, rms_minor_k = ra.optim_los_vel_coherence(
                sat_prime_coords, sat_prime_vels, coherent_frac_k, rms_minor_k, k)

        rms_minor_n[n] = np.min(rms_minor_k)
        min_rms_index = np.where(rms_minor_k == np.min(rms_minor_k))[0][0]
        coherent_frac_n[n] = coherent_frac_k[min_rms_index]

    return {'coherent.fraction':coherent_frac_n, 'rms':rms_minor_n}

@jit
def iso_rand_angle(iso_hal, hal_mask=None, n_iter=None):
    sat_coords = iso_hal['iso_coords']
    rot_vecs, rot_mats = ra.rand_rot_vec(n_iter)
    open_angle_k = []

    for k in range(n_iter):
        open_angle_n = []

        for n in range(n_iter):
            sat_prime_coords = ut.basic.coordinate.get_coordinates_rotated(sat_coords[k], rotation_tensor=rot_vecs[n])
            tangent_of_open_angle = sat_prime_coords[:,2]/np.sqrt(sat_prime_coords[:,0]**2 + sat_prime_coords[:,1]**2)
            open_angle_n.append(np.degrees(np.arctan(tangent_of_open_angle)))

        open_angle_k.append(np.min(open_angle_n))

    return open_angle_k, rot_vecs, rot_mats

@jit
def iso_rand_frac_open_angle(iso_hal, angle_bins=None, n_iter=1000):
    '''
    Calculate the fraction of satellites enclosed at each angular bin in sat.a_bins
    for the isotropic distribution of each snapshot.
    '''
    sat_coords = iso_hal['iso_coords']
    rot_vecs, rot_mats = ra.rand_rot_vec(n_iter)
    frac_enclosed_k = np.zeros((n_iter,len(angle_bins)))

    for k in range(n_iter):
        frac_enclosed_n = np.zeros((n_iter,len(angle_bins)))

        for n in range(n_iter):
            iter_frac_enclosed = []
            #iter_angles = iso_hal['iso_angles'][n]
            sat_prime_coords = ut.basic.coordinate.get_coordinates_rotated(sat_coords[k], rotation_tensor=rot_vecs[n])
            tangent_of_open_angle = sat_prime_coords[:,2]/np.sqrt(sat_prime_coords[:,0]**2 + sat_prime_coords[:,1]**2)
            #open_angle_n.append(np.degrees(np.arctan(tangent_of_open_angle)))
            iter_angles = np.degrees(np.arctan(tangent_of_open_angle))

            for open_angle in angle_bins:
                angle_mask = (iter_angles <= open_angle) & (iter_angles >= -open_angle)
                frac_enclosed = float(len(iter_angles[angle_mask]))/float(len(iter_angles))
                iter_frac_enclosed.append(frac_enclosed)

            frac_enclosed_n[n] = iter_frac_enclosed

        frac_enclosed_k[k] = np.max(frac_enclosed_n, axis=0)
    
    snap_frac_enclosed = np.average(frac_enclosed_k, axis=0)

    return snap_frac_enclosed

@jit
def rand_iso_rms_min(
    iso_hal, r_bins=None, n_iter=None, distribution=True, return_parallel=False):
    rot_vecs, rot_mats = ra.rand_rot_vec(n_iter)
    rms_minor_k = np.zeros((n_iter, n_iter))
    rms_major_k = np.zeros((n_iter, n_iter))

    # variables defined for optimization
    n_sat = len(iso_hal['iso_coords'][0])
    all_iso_coords = np.reshape(iso_hal['iso_coords'], (n_iter*n_sat, 3))

    for k,axes in enumerate(rot_vecs):
        sat_prime_coords = ut.basic.coordinate.get_coordinates_rotated(all_iso_coords, rotation_tensor=axes)
        sat_prime_coords = np.reshape(sat_prime_coords, (n_iter, n_sat, 3))
        rms_minor_k[k] = np.sqrt(np.nanmean(sat_prime_coords[:,:,2]**2, axis=1))
        rms_major_k[k] = np.sqrt(np.nanmean(sat_prime_coords[:,:,0]**2, axis=1))

    rms_minor_n = np.min(rms_minor_k, axis=0)
    rms_major_n = np.zeros(n_iter)
    for i in range(n_iter):
        rms_minor_i = np.min(rms_minor_k[i])
        min_index = np.where(rms_minor_k[i] == np.min(rms_minor_i))[0][0]
        rms_major_n[i] = rms_major_k[i][min_index]

    if return_parallel is True:
        if distribution:
            return {'rms_minor':rms_minor_n, 'rms_major':rms_major_n}
        else:
            return {'rms_minor':np.nanmean(rms_minor_n), 'rms_major':np.nanmean(rms_major_n)}
    else:
        if distribution:
            return rms_minor_n
        else:
            return np.nanmean(rms_minor_n)
