import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.patches import Polygon
from collections import defaultdict
from numba import jit
import astropy.coordinates as coord
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.coordinates import ICRS
import utilities as ut
import satellite_analysis as sa



def coord_transform_gc(pos_, vel_):
    # get galactocentric coords and velocity
    # len(pos_) = 3 for ra,dec,d_sun, similarly for vel_
    coords_hc = coord.ICRS(ra=pos_[0]*u.deg, dec=pos_[1]*u.deg, distance=pos_[2]*u.kpc, 
                           pm_ra_cosdec=vel_[0]*u.mas/u.yr, pm_dec=vel_[1]*u.mas/u.yr, 
                           radial_velocity=vel_[2]*u.km/u.s)
    coords_gc = coords_hc.transform_to(coord.Galactocentric)
    print(coords_gc)
    coords_gc_xyz = np.array([coords_gc.x.value, coords_gc.y.value, coords_gc.z.value])
    coords_gc_vel_xyz = np.array([coords_gc.velocity.d_x.value, coords_gc.velocity.d_y.value, 
                                coords_gc.velocity.d_z.value])
    
    return {'xyz.gc':coords_gc_xyz, 'v.xyz.gc':coords_gc_vel_xyz}

def coord_transform_gc_sample(pos_, pos_err, vel_, vel_err, name=None, n_iter=1000, return_dist=False):
    # sample uncertainties in (los) position and (full) velocity
    # len(pos_) = 3 for ra,dec,d_sun, similarly for vel_
    # operates on one satellite at a time
    r_helio_samp = np.random.normal(loc=pos_[2], scale=pos_err[2], size=n_iter)
    mu_ra_samp = np.random.normal(loc=vel_[0], scale=vel_err[0], size=n_iter)
    mu_dec_samp = np.random.normal(loc=vel_[1], scale=vel_err[1], size=n_iter)
    rv_helio_samp = np.random.normal(loc=vel_[2], scale=vel_err[2], size=n_iter)
    
    gc_position_samples = np.full((n_iter, 3), np.nan)
    gc_velocity_samples = np.full((n_iter, 3), np.nan)
    for i in range(n_iter):
        # set up heliocentric coords and velocity
        coords_hc = coord.ICRS(ra=pos_[0]*u.deg, dec=pos_[1]*u.deg, distance=r_helio_samp[i]*u.kpc, 
                               pm_ra_cosdec=mu_ra_samp[i]*u.mas/u.yr, pm_dec=mu_dec_samp[i]*u.mas/u.yr, 
                               radial_velocity=rv_helio_samp[i]*u.km/u.s)
        # convert to galactocentric
        coords_gc = coords_hc.transform_to(coord.Galactocentric)
        gc_position_samples[i] = np.array([coords_gc.x.value, coords_gc.y.value, coords_gc.z.value])
        gc_velocity_samples[i] = np.array([coords_gc.velocity.d_x.value, coords_gc.velocity.d_y.value, 
                                           coords_gc.velocity.d_z.value])
        
    if return_dist:
        return gc_position_samples, gc_velocity_samples
    else:
        xyz_gc_median = np.median(gc_position_samples, axis=0)
        xyz_gc_16 = np.percentile(gc_position_samples, 16, axis=0)
        xyz_gc_84 = np.percentile(gc_position_samples, 84, axis=0)
        xyz_gc_unc = np.maximum(xyz_gc_median-xyz_gc_16, xyz_gc_84-xyz_gc_median)
        
        v_xyz_gc_median = np.median(gc_velocity_samples, axis=0)
        v_xyz_gc_16 = np.percentile(gc_velocity_samples, 16, axis=0)
        v_xyz_gc_84 = np.percentile(gc_velocity_samples, 84, axis=0)
        v_xyz_gc_unc = np.maximum(v_xyz_gc_median-v_xyz_gc_16, v_xyz_gc_84-v_xyz_gc_median)
        return {'name':name,
                'x.gc':xyz_gc_median[0], 'y.gc':xyz_gc_median[1], 'z.gc':xyz_gc_median[2],
                'x.gc.unc':xyz_gc_unc[0], 'y.gc.unc':xyz_gc_unc[1], 'z.gc.unc':xyz_gc_unc[2],
                'v.x.gc':v_xyz_gc_median[0], 'v.y.gc':v_xyz_gc_median[1], 'v.z.gc':v_xyz_gc_median[2],
                'v.x.gc.unc':v_xyz_gc_unc[0], 'v.xyz.gc.unc':v_xyz_gc_unc[1], 'v.xyz.gc.unc':v_xyz_gc_unc[2]
               }

def plane_calc_obs_uncertainty(exec_func, mw_icrs_table, n_iter=1000, exclude=[]):
    # n_iter set in this function used for obs samples, plane calculation n_iters are left as default values
    
    # remove any satellites from loop that are excluded
    loop_sat_names = list(mw_icrs_table['name'].values)
    for sat_exc in exclude:
        loop_sat_names.remove(sat_exc)
    
    # set up an empty array for samples
    sat_pos_samples = np.full((n_iter, len(loop_sat_names), 3), np.nan)
    sat_vel_samples = np.full((n_iter, len(loop_sat_names), 3), np.nan)
    for i,sat in enumerate(loop_sat_names):
        msk = mw_icrs_table['name'] == sat
        sat_pos_samples[:,i,:], sat_vel_samples[:,i,:] = coord_transform_gc_sample(
                                                        [mw_icrs_table['ra'][msk].values[0], 
                                                       mw_icrs_table['dec'][msk].values[0], 
                                                       mw_icrs_table['d_sun'][msk].values[0]],
                                                      [0, 0, mw_icrs_table['d_sun_err'][msk].values[0]], 
                                                      [mw_icrs_table['mu_a'][msk].values[0], 
                                                       mw_icrs_table['mu_d'][msk].values[0], 
                                                       mw_icrs_table['v_sun'][msk].values[0]], 
                                                      [mw_icrs_table['mu_a_err'][msk].values[0], 
                                                       mw_icrs_table['mu_d_err'][msk].values[0], 
                                                       mw_icrs_table['v_sun_err'][msk].values[0]], 
                                                      name=sat, n_iter=n_iter, return_dist=True)

    samp_value = []
    for n in range(n_iter):
        samp_value.append(exec_func(sat_pos_samples[n], sat_vel_samples[n]))
    up68 = np.percentile(samp_value, 84) - np.median(samp_value)
    low68 = np.median(samp_value) - np.percentile(samp_value, 16)
    up95 = np.percentile(samp_value, 97.5) - np.median(samp_value)
    low95 = np.median(samp_value) - np.percentile(samp_value, 2.5)
    print('{:.0f} [+{:.0f} -{:.0f}] [{:.0f} {:.0f}] 68%'.format(np.median(samp_value), up68, low68, 
                                                            np.percentile(samp_value, 16),
                                                            np.percentile(samp_value, 84)))
    print('{:.2f} [+{:.2f} -{:.2f}] [{:.2f} {:.2f}] 68%'.format(np.median(samp_value), up68, low68, 
                                                            np.percentile(samp_value, 16),
                                                            np.percentile(samp_value, 84)))
    print('{:.0f} [+{:.0f} -{:.0f}] [{:.0f} {:.0f}] 95%'.format(np.median(samp_value), up95, low95, 
                                                            np.percentile(samp_value, 2.5),
                                                            np.percentile(samp_value, 97.5)))
    print('{:.2f} [+{:.2f} -{:.2f}] [{:.2f} {:.2f}] 95%'.format(np.median(samp_value), up95, low95, 
                                                            np.percentile(samp_value, 2.5),
                                                            np.percentile(samp_value, 97.5)))
    
def isotropic_fraction(exec_func, mw_icrs_table, n_iter=1000, true_metric_value=0, exclude=[]):
    # n_iter set in this function used for isotropic samples, plane calculation n_iters are default values
    
    # remove any satellites from loop that are excluded
    loop_sat_names = list(mw_icrs_table['name'].values)
    for sat_exc in exclude:
        loop_sat_names.remove(sat_exc)
    
    # set up an empty array for samples
    sat_r_gc = np.full(len(loop_sat_names), np.nan)
    for i,sat in enumerate(loop_sat_names):
        msk = mw_icrs_table['name'] == sat
        coords_hc = coord.ICRS(ra=mw_icrs_table['ra'][msk].values[0]*u.deg, 
                               dec=mw_icrs_table['dec'][msk].values[0]*u.deg, 
                               distance=mw_icrs_table['d_sun'][msk].values[0]*u.kpc)
        coords_gc = coords_hc.transform_to(coord.Galactocentric)
        sat_r_gc[i] = coords_gc.separation_3d(coord.Galactocentric(0*u.kpc,0*u.kpc,0*u.kpc)).kpc
            
    iso_coords, iso_vels = sa.isotropic.iso_iter(sat_r_gc, n_iter=n_iter)

    samp_value = []
    for n in range(n_iter):
        samp_value.append(exec_func(iso_coords[n], iso_vels[n]))
    
    return np.sum(np.array(samp_value) <= true_metric_value)/len(samp_value)

def orbital_pole_dispersion(position, velocity, norm=False):
    '''
    Calculate the angular dispersion [deg] of satellite orbital poles around
    their mean orbital pole.
    '''
    position = position*ut.basic.constant.km_per_kpc
    j_vec = np.array([np.cross(x_,v_)/np.linalg.norm(np.cross(x_,v_), keepdims=True) for x_, v_ in zip(position, velocity)])
    avg_j_vec = np.mean(j_vec, axis=0, dtype=np.float64)/np.linalg.norm(np.mean(j_vec, axis=0, dtype=np.float64), keepdims=True)
    avg_j_dot_j = np.array([np.dot(avg_j_vec, j_vec_i) for j_vec_i in j_vec]) 
    pole_disp = np.sqrt(np.mean(np.arccos(avg_j_dot_j, dtype=np.float64)**2, dtype=np.float64))
    pole_disp = np.degrees(pole_disp, dtype=np.float64)

    if norm:
        return avg_j_vec
    else:
        return pole_disp

def orbital_poles(position, velocity):
    '''
    Calculate the angular dispersion [deg] of satellite orbital poles around
    their mean orbital pole.
    '''
    position = position*ut.basic.constant.km_per_kpc
    j_vec = np.array([np.cross(x_,v_)/np.linalg.norm(np.cross(x_,v_), keepdims=True) for x_, v_ in zip(position, velocity)])
    avg_j_vec = np.mean(j_vec, axis=0, dtype=np.float64)/np.linalg.norm(np.mean(j_vec, axis=0, dtype=np.float64), keepdims=True)
    avg_j_dot_j = np.array([np.dot(avg_j_vec, j_vec_i) for j_vec_i in j_vec]) 
    pole_disp = np.sqrt(np.mean(np.arccos(avg_j_dot_j, dtype=np.float64)**2, dtype=np.float64))
    pole_disp = np.degrees(pole_disp, dtype=np.float64)

    return j_vec, avg_j_vec, pole_disp

def rand_rms_min(
    sat_coords, sat_vels, n_iter=5000, r_frac=1.0, norm=False, both=False):
    '''
    Rotates the 3D positions of the satellites randomly and uniformly for sat.n_iter
    realizations. Finds the angle off of the simulation x-axis that encloses a given
    fraction of the satellites, and then gets the minimum of this angle across all
    random realizations for each host halo at each redshift.
    '''
    rot_vecs, rot_mats = sa.rand_axes.rand_rot_vec(n_iter)
    rms_minor_n = np.zeros(n_iter)

    for n, rot_vec in enumerate(rot_vecs):
        sat_prime_coords = ut.basic.coordinate.get_coordinates_rotated(sat_coords, rotation_tensor=rot_vec)
        rms_minor_n[n] = np.sqrt(np.mean(sat_prime_coords[:,2]**2, dtype=np.float64), dtype=np.float64)

    norm_axes = rot_vecs[np.where(rms_minor_n == np.min(rms_minor_n))[0][0]]
    min_rms_minor = np.min(rms_minor_n)

    if norm is True:
        return norm_axes[2]
    elif both is True:
        return min_rms_minor, norm_axes
    else:
        return min_rms_minor

def rand_angle(sat_coords, n_iter=5000):
    '''
    Calculates opening angles off of a set of randomly/isotropicallly generated
    axes (for sat.n_iter realizations).
    Returns opening angles, the vectors that the original axes are rotated by,
    and the rotation matrix that performs the rotations.
    '''
    rot_vecs, rot_mats = sa.rand_axes.rand_rot_vec(n_iter)
    open_angle_n = []

    for n in range(n_iter):
        sat_prime_coords = ut.basic.coordinate.get_coordinates_rotated(sat_coords, rotation_tensor=rot_vecs[n])
        tangent_of_open_angle = sat_prime_coords[:,2]/np.sqrt(sat_prime_coords[:,0]**2 + sat_prime_coords[:,1]**2)
        open_angle_n.append(np.degrees(np.arctan(tangent_of_open_angle)))

    return np.array(open_angle_n), rot_vecs

@jit(nopython=True)
def optim_open_angle(snap_angles, angle_range, threshold_fraction, phi_width, i):
    for opening_angle in angle_range:
        #angle_mask = (snap_angles <= opening_angle) & (snap_angles >= -opening_angle)
        angle_mask = np.abs(snap_angles) <= opening_angle
        frac_enclosed = np.sum(angle_mask)/snap_angles.size
        
        if frac_enclosed >= threshold_fraction:
            phi_width[i] = 2*opening_angle
            break
        else:
            pass

    return phi_width

def rand_angle_width(
    sat_coords, sat_vels, n_iter=5000, fraction=1.0, angle_range=np.arange(0,91.5,1.5), norm=False):
    '''
    Rotates the 3D positions of the satellites randomly and uniformly for sat.n_iter
    realizations. Finds the angle off of the simulation x-axis that encloses a given
    fraction of the satellites, and then gets the minimum of this angle across all
    random realizations for each host halo at each redshift.
    '''
    rand_angles, rot_vecs_ = rand_angle(sat_coords, n_iter=n_iter)
    phi_width_n = np.zeros(n_iter)

    for n, snap_angles_n in enumerate(rand_angles):
        if snap_angles_n.size == 0:
            phi_width_n[n] = np.nan
        else:
            phi_width_n = optim_open_angle(snap_angles_n, angle_range, fraction, phi_width_n, n)

    phi_width = np.min(phi_width_n)
    #print(rot_vecs_[np.where(phi_width_n == np.min(phi_width_n))[0]])
    norm_axes = rot_vecs_[np.where(phi_width_n == np.min(phi_width_n))[0][0]]
    angs = rand_angles[np.where(phi_width_n == np.min(phi_width_n))[0][0]]

    if norm:
        return phi_width, norm_axes, angs
    else:
        return phi_width

def axis_ratio(position, velocity, norm=False):
    if norm:
        return ut.coordinate.get_principal_axes(position, verbose=False)[0][2]
    else:
        return ut.coordinate.get_principal_axes(position, verbose=False)[2][0]

def plane_metric_diagram(r_vec, v_vec, sat_names):
    # plot diagram of plane metrics
    fig, (ax1,ax2,ax3) = plt.subplots(1, 3, figsize=(12,4.5), sharex=True, sharey=True)
    fig.set_tight_layout(False)
    fig.subplots_adjust(left=0.1, right=0.99, top=0.99, bottom=0.15, hspace=0, wspace=0.03)
    colors = sa.population_planarity.color_cycle(cmap_name='rainbow')
    colors = colors[::-1]
    fig_scale_lim = 270

    # plot rms height
    rms_height, rms_axes = rand_rms_min(r_vec, v_vec, n_iter=10000, r_frac=1.0, both=True)
    rms_coords = ut.basic.coordinate.get_coordinates_rotated(r_vec, rotation_tensor=rms_axes)

    ax1.axhline(rms_height, c='k', linestyle=':', alpha=0.7)
    ax1.axhline(-rms_height, c='k', linestyle=':', alpha=0.7)
    for mac,c,sat_name in zip(rms_coords, colors, sat_names):
        ax1.plot(mac[0], mac[2], 'o', color=c, alpha=0.8, markeredgecolor='k', markeredgewidth=1, label=sat_name)
    print('rms height:', rms_height)
    fig.text(0.15, 0.25, r'RMS height = {:.0f} kpc'.format(rms_height), va='center', fontsize=18)
        
        
    # plot axis ratio
    pr_axes = ut.coordinate.get_principal_axes(r_vec, verbose=False)
    axis_coords = ut.basic.coordinate.get_coordinates_rotated(r_vec, rotation_tensor=pr_axes[0])
    axis_vels = ut.basic.coordinate.get_coordinates_rotated(v_vec, rotation_tensor=pr_axes[0])

    axis_ratio_0 = pr_axes[1][0]
    axis_ratio_1 = pr_axes[1][1]
    ell_maj_ax = 500
    axis_ratio_ell = Ellipse(xy=(0,0), width=ell_maj_ax, height=axis_ratio_0*ell_maj_ax, edgecolor='k', lw=2, 
                            facecolor='none', alpha=0.7, linestyle=':')
    ax2.add_artist(axis_ratio_ell)
    for mac,c,sat_name in zip(axis_coords, colors, sat_names):
        ax2.plot(mac[0], mac[2], 'o', color=c, alpha=0.8, markeredgecolor='k', markeredgewidth=1)
    print('axis ratio:', axis_ratio_0) 
    fig.text(0.47, 0.25, r'Axis ratio = {:.2f}'.format(axis_ratio_0), va='center', fontsize=18)



    #plot orbital poles
    orb_poles, avg_pole, orb_disp = orbital_poles(axis_coords, axis_vels)
    print('orbital dispersion:', orb_disp)
    q = ax3.quiver(axis_coords[:,0], axis_coords[:,2], orb_poles[:,0], orb_poles[:,2],
                    angles='xy', scale_units='xy', scale=0.014, alpha=0.6)
    q = ax3.quiver(0, 0, avg_pole[0], avg_pole[2], angles='xy', scale_units='xy', scale=0.0045, color='k')

    # orbital dispersion cone using polar angle coordinates/trig
    avg_pole_angle = -np.arctan(avg_pole[2]/avg_pole[0])
    cone_side_length = 2*fig_scale_lim*np.sqrt(avg_pole[0]**2 + avg_pole[2]**2)/np.cos(np.radians(orb_disp))
    cone_angle1 = np.radians(orb_disp) + avg_pole_angle
    cone_pt1 = cone_side_length*np.array([np.cos(cone_angle1), np.sin(cone_angle1)])

    cone_angle2 = avg_pole_angle - np.radians(orb_disp)
    cone_pt2 = cone_side_length*np.array([np.cos(cone_angle2), np.sin(cone_angle2)])

    orb_pts_ = np.array([[0,0], cone_pt1, cone_pt2])
    p = Polygon(orb_pts_, closed=True, fill=True, color='k', alpha=0.2)
    ax3.add_patch(p)

    # plot satellite positions
    for mac,c,sat_name in zip(axis_coords, colors, sat_names):
        ax3.plot(mac[0], mac[2], 'o', color=c, alpha=0.8, markeredgecolor='k', markeredgewidth=1)
        
    fig.text(0.74, 0.25, r'Orbital dispersion = {:.0f} $\degree$'.format(orb_disp), va='center', fontsize=18)

    #fig.text(0.415, 0.85, r'Milky Way Satellite Plane', va='center', fontsize=20)

    #titles = ('RMS height', 'Axis ratio', 'Opening angle', 'Orbital dispersion')
    for ax in (ax1,ax2,ax3):
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.axis('square')
        ax.set_xlim((-fig_scale_lim,fig_scale_lim))
        ax.set_ylim((-fig_scale_lim,fig_scale_lim))
        kpc_ticks = [-200, -100, 0, 100, 200]
        ax.set_xticks(kpc_ticks, minor=False)
        ax.set_yticks(kpc_ticks, minor=False)
        ax.set_xlabel('[kpc]', fontsize=20)
        #ax.set_title(title, fontsize=22)
    ax1.legend(fontsize=14, ncol=4, loc='upper center', columnspacing=0.55, borderaxespad=0.5)
    ax1.set_ylabel('[kpc]', fontsize=20)
    plt.show()

    return fig