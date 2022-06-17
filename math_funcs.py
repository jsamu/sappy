import numpy as np
import math
from scipy import stats


def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def qf_stats(qf, mstar_bins=np.array([1e5, 1e6, 1e7, 1e8, 1e9]), percentile_limits=[16,84], name=''):
    std_ = np.nanstd(qf, axis=0)
    # eliminate values of scatter that are plotted outside [0,1]
    std = np.where(
        np.nanmean(qf, axis=0)+std_ > 1, 1-np.nanmean(qf, axis=0), std_)
    std = np.where(
        np.nanmean(qf, axis=0)-std_ < 0, np.nanmean(qf, axis=0), std)
    return {
        'name':np.full(np.nanmean(qf, axis=0).shape, name),
        'left.star.mass.bin':mstar_bins,
        'mean':np.nanmean(qf, axis=0),
        'median':np.nanmedian(qf, axis=0),
        'min':np.nanmin(qf, axis=0),
        'max':np.nanmax(qf, axis=0),
        'std':std,
        '16_percentile':np.nanpercentile(qf, percentile_limits[0], axis=0),
        '84_percentile':np.nanpercentile(qf, percentile_limits[1], axis=0)
        }

def beta_error(numers, denoms, conf_inter=0.683):
    # taken from AW's LG QF code
    #conf_inter = 0.683  # 1 - sigma
    p_lo = numers / denoms - stats.distributions.beta.ppf(
        0.5 * (1 - conf_inter), numers + 1, denoms - numers + 1
    )
    p_hi = (
        stats.distributions.beta.ppf(
            1 - 0.5 * (1 - conf_inter), numers + 1, denoms - numers + 1
        )
        - numers / denoms
    )
    return np.array([p_lo, p_hi]).clip(0)

def round_up(x, base=5):
    '''
    Rounds the given value up to the next n*base value.
    '''
    x_round = int(base * round(float(x)/base))
    if x_round < x:
        x_round = x_round + base

    return x_round

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

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


def coadd_redshift(
    property_dict, dict_key=None, coadd_axis=0, all_sims=False,
    percentiles=[16,84,2.5,97.5]):
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
        hal_dict['percentile'] = np.nanpercentile(redshift_prop, percentiles, axis=coadd_axis)

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