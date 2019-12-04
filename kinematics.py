import numpy as np
from numba import jit
import utilities as ut
from satellite_analysis import satellite_io as sio
from satellite_analysis import spatial as spa
from satellite_analysis import rand_axes as ra


def v_peak(hal, hal_mask=None, host_str='host.', hal_property='vel.circ.max', return_snap=False):
    all_snapshots = np.arange(0,601,1)
    initial_ids = np.where(hal_mask)[0]
    tracked_ids = sio.halo_track(hal, initial_ids, snapshot_list=all_snapshots)
    tracked_vmax = np.array([hal[hal_property][ids.astype(int)] for ids in tracked_ids])
    peak_vmax = np.max(tracked_vmax, axis=0)
    
    if return_snap:
        peak_vmax_snapshot = []
        for i,peak in enumerate(peak_vmax):
            peak_vmax_snapshot.append(all_snapshots[-1+np.where(tracked_vmax[:,i]==peak)[0][0]])
        return peak_vmax, peak_vmax_snapshot
    else:
        return peak_vmax

def med_velcircmax_z0(sat, mask_key, hal_prop='vel.circ.max', single_value=False):
    '''
    Get the median of the minimum vel.circ.max of the (baryonic) satellites of
    the host halos in halo_catalog at redshift z=0. Used to mask DMO catalog.
    '''
    min_vels = sio.loop_hal(sat, mask_key, min_v_circ, **{'hal_prop':hal_prop})
    if single_value:
        median_vel_at_z = np.median([np.average(min_vels[host]) for host in min_vels.keys()])
    else:
        new_list = [min_vels[host] for host in min_vels.keys()]
        median_vel_at_z = np.median(new_list, axis=0)

    return median_vel_at_z

def min_v_circ(hal, hal_mask=None, host_str='', hal_prop='vel.circ.max'):
    '''
    Get the minimum vel.circ.max of the baryonic satellites for each host halo at
    each redshift in halo_catalog.
    '''
    try:
        min_circ_vel = np.min(hal[hal_prop][hal_mask])
    except:
        min_circ_vel = np.min(hal.prop(hal_prop)[hal_mask])

    return min_circ_vel

def internal_ang_momentum_dot_prod(hal, hal_mask=None, dot_vector=None):
    '''
    Get the dot product of the angular momentum of the satellites in a snapshot
    (host) with any vector (dot_vector).
    '''
    dot_product_list = []
    J_x = hal.prop('momentum.ang.x')[hal_mask]
    J_y = hal.prop('momentum.ang.y')[hal_mask]
    J_z = hal.prop('momentum.ang.z')[hal_mask]

    for i, j in enumerate(J_x):
        dot_product_list.append(np.sum([J_x[i]*dot_vector[0], J_y[i]*dot_vector[1], J_z[i]*dot_vector[2]]))

    return dot_product_list

def project_internal_ang_momentum(hal, hal_mask=None):
    '''
    Project the satellite angular momentum onto the MOI axes of the satellite
    distribution (should add functionality to project onto any input vector).
    '''
    sat_axes = spa.get_satellite_principal_axes(hal, hal_mask)
    J_x = hal.prop('momentum.ang.x')[hal_mask]
    J_y = hal.prop('momentum.ang.y')[hal_mask]
    J_z = hal.prop('momentum.ang.z')[hal_mask]
    J_vec = np.array([[J_x[i], J_y[i], J_z[i]] for i in range(len(J_x))])
    aligned_ang_momentum = ut.basic.coordinate.get_coordinates_rotated(J_vec, rotation_tensor=sat_axes[0])

    return aligned_ang_momentum

#@jit
def orbital_ang_momentum(hal, hal_mask=None, host_str='host.', norm=False):
    '''
    Compute mass-agnostic orbital angular momentum as L=(v)x(r) where the
    vectors are defined with respect to the central host halo. Returned value
    has units of kpc^2/s.
    '''
    position = hal.prop(host_str+'distance')[hal_mask]#*ut.basic.constant.km_per_kpc
    velocity = hal.prop(host_str+'velocity')[hal_mask]*ut.basic.constant.kpc_per_km
    if norm is True:
        ang_momentum = np.array([np.cross(x,v)/np.linalg.norm(np.cross(x,v)) for x, v in zip(position, velocity)])
    elif norm is False:
        ang_momentum = np.array([np.cross(x,v) for x, v in zip(position, velocity)])

    return ang_momentum

#@jit
def orbital_pole_dispersion(hal, hal_mask=None, host_str='host.'):
    '''
    Calculate the angular dispersion [deg] of satellite orbital poles around
    their mean orbital pole.
    '''
    if np.sum(hal_mask) == 0:
        pole_disp = np.nan
        avg_j_vec = np.array([np.nan, np.nan, np.nan])
    else:
        j_vec = orbital_ang_momentum(hal, hal_mask, host_str=host_str, norm=True)
        avg_j_vec = np.mean(j_vec, axis=0, dtype=np.float64)/np.linalg.norm(np.mean(j_vec, axis=0))
        avg_j_dot_j = np.array([np.dot(avg_j_vec, j_vec_i) for j_vec_i in j_vec]) 
        pole_disp = np.sqrt(np.mean(np.arccos(avg_j_dot_j)**2, dtype=np.float64))
        pole_disp = np.degrees(pole_disp)

    return {'orbital.pole.dispersion':pole_disp, 'average.orbital.pole':avg_j_vec}

def project_orbital_ang_momentum(hal, hal_mask=None, project_axes=None, norm=False):
    '''
    Project the satellite angular momentum onto the MOI axes of the satellite
    distribution (should add functionality to project onto any input vector).
    Default projection axes are the MOI tensor axes of all satellites in hal_mask.
    '''
    j_vec = orbital_ang_momentum(hal, hal_mask)
    if project_axes is None:
        project_axes = spa.get_satellite_principal_axes(hal, hal_mask)
        proj_ang_momentum = ut.basic.coordinate.get_coordinates_rotated(j_vec, rotation_tensor=project_axes[0])
    else:
        proj_ang_momentum = ut.basic.coordinate.get_coordinates_rotated(j_vec, rotation_tensor=project_axes)

    if norm is True:
        proj_ang_momentum = np.array([j/np.linalg.norm(j) for j in proj_ang_momentum])
    elif norm is False:
        pass

    return proj_ang_momentum

def project_orb_ang_mom_randminaxes(hal, hal_mask=None, fraction=1.0, angle_range=None, n_iter=1000, norm=False):
    '''
    Project the satellites' orbital angular momentum onto the random axes whose
    z component is normal to the plane that minimizes the enclosing angle for a
    given fraction of satellites. Enclosing angle is measured off of the x-y plane.
    '''
    rand_min_angle_dict = ra.rand_angle_width(hal, hal_mask, n_iter=n_iter, fraction=fraction, angle_range=angle_range)
    min_axes = rand_min_angle_dict['min_axes']
    projected_j = project_orbital_ang_momentum(hal, hal_mask, project_axes=min_axes, norm=norm)

    return projected_j

def ang_momentum_MOI_axes_dotprod(hal, hal_mask=None, norm=False):
    spatial_MOI = spa.get_satellite_principal_axes(hal, hal_mask)
    spatial_MOI_z_axis = spatial_MOI[0][2]

    j_vec = orbital_ang_momentum(hal, hal_mask, norm=norm)
    j_MOI = ut.coordinate.get_principal_axes(j_vec)
    j_MOI_z_axis = j_MOI[0][2]

    moi_dot_prod = abs(np.dot(spatial_MOI_z_axis, j_MOI_z_axis))

    return moi_dot_prod