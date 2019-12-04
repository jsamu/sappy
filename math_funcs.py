import numpy as np
import math


def round_up(x, base=5):
    '''
    Rounds the given value up to the next n*base value.
    '''
    x_round = int(base * round(float(x)/base))
    if x_round < x:
        x_round = x_round + base

    return x_round

def isRotationMatrix(R):
    '''
    Checks if a matrix is a valid rotation matrix. Cite: www.LearnOpenCV.com
    '''
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)

    return n < 1e-6

def rotationMatrixToEulerAngles(R):
    '''
    Calculates rotation matrix to euler angles
    The result is the same as MATLAB except the order
    of the euler angles (x and z are swapped).  Cite: www.LearnOpenCV.com
    '''
    assert(isRotationMatrix(R))
     
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
     
    singular = sy < 1e-6
 
    if  not singular:
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.degrees(np.array([x, y, z]))


########################################################
### moved here from basic file in an effort to split ###
### up functions more logically

def coadd_redshift(property_dict, dict_key=None, coadd_axis=0, all_sims=False):
    '''
    Get some basic satistics on a calculated property over time.
    '''
    hal_dict = {}

    # coadd over all sims and times
    if all_sims:
        redshift_prop = []
        for hal_name in property_dict.keys():
            redshifts = len(property_dict[hal_name])
            if type(property_dict[hal_name][0]) is list:
                if type(property_dict[hal_name][0][0]) is list:
                    for j in range(redshifts):
                        for i in range(len(property_dict[hal_name][j])):
                            redshift_prop.append(property_dict[hal_name][j][i])
                else:
                    redshift_prop = redshift_prop+[property_dict[hal_name][j] for j in range(redshifts)]
            
        hal_dict['mean'] = np.nanmean(redshift_prop, axis=coadd_axis)
        hal_dict['median'] = np.nanmedian(redshift_prop, axis=coadd_axis)
        hal_dict['percentile'] = np.nanpercentile(redshift_prop, [16, 84, 2.5, 97.5], axis=coadd_axis)

    else:
        for hal_name in property_dict.keys():
            coadd_dict = {}
            redshifts = len(property_dict[hal_name])
            if type(property_dict[hal_name][0]) is dict:
                redshift_prop = [property_dict[hal_name][j][dict_key] for j in range(redshifts)] 
            elif type(property_dict[hal_name][0]) is list:
                if type(property_dict[hal_name][0][0]) is list:
                    redshift_prop = []
                    for j in range(redshifts):
                        for i in range(len(property_dict[hal_name][j])):
                            redshift_prop.append(property_dict[hal_name][j][i])
                else:
                    redshift_prop = [property_dict[hal_name][j] for j in range(redshifts)]
            elif type(property_dict[hal_name][0]) is np.ndarray:
                redshift_prop = [property_dict[hal_name][j] for j in range(redshifts)]
            elif type(property_dict[hal_name][0]) is float:
                redshift_prop = property_dict[hal_name]
            coadd_dict['mean'] = np.nanmean(redshift_prop, axis=coadd_axis)
            coadd_dict['median'] = np.nanmedian(redshift_prop, axis=coadd_axis)
            coadd_dict['percentile'] = np.nanpercentile(redshift_prop, [16, 84, 2.5, 97.5], axis=coadd_axis)
            hal_dict[hal_name] = coadd_dict

    return hal_dict

def co_add_prop(hal_property_list, coadd_axis=0):
    '''
    Get some basic satistics on hal_property_list over time.
    '''
    means = []
    medians = []
    percentiles = []

    for hal_prop in hal_property_list:
        means.append(np.nanmean(hal_prop, axis=coadd_axis))
        medians.append(np.nanmedian(hal_prop, axis=coadd_axis))
        percentiles.append(np.nanpercentile(hal_prop, [16, 84, 2.5, 97.5], axis=coadd_axis, interpolation='nearest'))

    return means, medians, percentiles

def cumulative_prop_lg(
    hal, hal_mask=None, host_str='host.', hal_property='distance', bins=None,
    above=False, normalized=False):
    '''
    Get a cumulative distribution for hal_property given bins and hal_mask, of 
    a Local Group SatelliteTree object.
    '''
    snapshot_cumulative_dist = []
    dist_3d = hal[host_str+'distance']
    total_distance = np.sqrt(dist_3d[:,0]**2 + dist_3d[:,1]**2 + dist_3d[:,2]**2)
    masked_property = total_distance[hal_mask]

    for _bin in bins:
        if above == False:
            bin_mask = masked_property <= _bin
        elif above == True:
            bin_mask = masked_property >= _bin
        snapshot_cumulative_dist.append(len(masked_property[bin_mask]))

    if normalized == True:
        scd_max = np.max(snapshot_cumulative_dist)
        snapshot_cumulative_dist = snapshot_cumulative_dist/scd_max

    return snapshot_cumulative_dist

def cumulative_prop(
    hal, hal_mask=None, hal_property='distance.total', host_str='host.',
    bins=None, above=False, normalized=False):
    '''
    Generates a cumulative distribution for hal_property given bins and hal_mask.
    '''
    snapshot_cumulative_dist = []
    prop_str = '{}{}'.format(host_str, hal_property)
    masked_property = hal.prop(prop_str)[hal_mask]

    for _bin in bins:
        if above == False:
            bin_mask = masked_property <= _bin
        elif above == True:
            bin_mask = masked_property >= _bin
        snapshot_cumulative_dist.append(len(masked_property[bin_mask]))

    if normalized == True:
        scd_max = np.max(snapshot_cumulative_dist)
        snapshot_cumulative_dist = snapshot_cumulative_dist/scd_max
    return snapshot_cumulative_dist