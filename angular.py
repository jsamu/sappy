import numpy as np
import utilities as ut
from satellite_analysis import isotropic as iso
from satellite_analysis import spatial as spa
from satellite_analysis import math_funcs as mf
from satellite_analysis import rand_axes as ra


def open_angle(hal, hal_mask=None, return_matrix=False, return_vec=False):
    '''
    Define the opening angle of satellites off of the major axis of the inertia 
    tensor defined by the geometric distribution of all satellites within the
    fiducial virial radius (for all host halos in a sat object at each redshift.
    '''
    sat_axes = spa.get_satellite_principal_axes(hal, hal_mask)
    sat_coords = hal.prop('host.distance')[hal_mask]
    rot_vec = sat_axes[0]
    rot_mat = rot_vec.T
    sat_prime_coords = ut.basic.coordinate.get_coordinates_rotated(sat_coords, rotation_tensor=rot_vec)
    tangent_of_openning_angle = sat_prime_coords[:,2]/np.sqrt(sat_prime_coords[:,0]**2 + sat_prime_coords[:,1]**2)
    open_angles = np.degrees(np.arctan(tangent_of_openning_angle))

    if return_matrix == True:
        return {'angle':open_angles, 'matrix':rot_mat}

    elif return_vec == True:
        return {'angle':open_angles, 'axis':rot_vec}

    else:
        return open_angles

def angle_width(hal, hal_mask=None, threshold_fraction=0.68, angle_bins=None):
    '''
    Calculates the angle (centered on the major axis of the satellite distribution
    major axis) that encloses a given fraction of the total satellites.
    '''
    hal_angles = open_angle(hal, hal_mask=hal_mask, return_matrix=False, return_vec=False)

    for opening_angle in angle_bins:
        angle_mask = (hal_angles <= opening_angle) & (hal_angles >= -opening_angle)
        frac_enclosed = float(len(hal_angles[angle_mask]))/float(len(hal_angles))
        
        if frac_enclosed >= threshold_fraction:
            phi_width = 2*opening_angle
            break
        else:
            pass

    return phi_width

def fraction_open_angle(hal, hal_mask=None, angle_bins=None):
    '''
    Calculates the fraction of satellites enclosed for each angle in sat.a_bins.
    Returns the means, medians, and 68th and 95th percentiles of these values
    over the input redshift range.
    '''
    hal_angles = open_angle(hal, hal_mask=hal_mask, return_matrix=False, return_vec=False)
    hal_frac_enclosed = []
    for opening_angle in angle_bins:
        angle_mask = (hal_angles <= opening_angle) & (hal_angles >= -opening_angle)
        frac_enclosed = float(len(hal_angles[angle_mask]))/float(len(hal_angles))
        hal_frac_enclosed.append(frac_enclosed)

    return hal_frac_enclosed

# this is used when calculating rms vs fraction enclosed since it returns a mask over the angular bins
def open_angle_mask(hal, hal_mask=None, angle_bins=None):
    open_angles = open_angle(hal, hal_mask=hal_mask)
    hal_angle_masks = []

    for _angle in angle_bins:
        angle_mask = (open_angles <= _angle) & (open_angles >= -_angle)
        hal_angle_masks.append(angle_mask)

    angle_range = 2*angle_bins

    return angle_range, hal_angle_masks

def open_angle_v_r(hal, hal_mask=None, threshold_fraction=0.5, radius_bins=None, angle_bins=None):
    '''
    Calculates the angle and radius that enclose a given fraction of the satellites.
    '''
    radii = hal.prop('host.distance.total')[hal_mask]
    open_angles = open_angle(hal, hal_mask=hal_mask)

    for k in radius_bins:
        radius_mask = radii <= k
        frac_enclosed = float(len(radii[radius_mask]))/float(len(radii))
        
        if frac_enclosed >= threshold_fraction:
            hal_rad = k
            r_mask_angles = open_angles[radius_mask]
            for a in angle_bins:
                angle_mask = (r_mask_angles <= a) & (r_mask_angles >= -a)
                frac_enc = float(len(r_mask_angles[angle_mask]))/float(len(r_mask_angles))

                if frac_enc == 1:
                    hal_angle = 2*a
                    break

            break

        else:
            pass

    return {'radii':hal_rad, 'angles':hal_angle}
