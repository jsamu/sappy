import numpy as np
import utilities as ut
from scipy.integrate import quad
from satellite_analysis import math_funcs as mf


def integrate_kernel(
    h_i, int_lower_limit, weight_distance=None, weight_size=None, normalize=True,
    int_upper_limit=None):
    # need to make some sort of switch that decides btwn full or partial integral
    # depending on if center of gas cell is inside or outside the cylinder
    # weight should be the distance btwn a gas cell center and the center
    # of the ram pressure cylinder

    assert (int_lower_limit>=0) & (h_i>int_lower_limit), 'bad lower integration limit'

    def small_q_kernel(q, h_i):
        return (8 / np.pi * h_i ** 3)*(1 + 6 * q ** 2 * (q - 1))

    def large_q_kernel(q, h_i):
        return (8 / np.pi * h_i ** 3)*(2 * (1 - q) ** 3)

    # normalize radial coordinates
    q1 = int_lower_limit/h_i
    if int_upper_limit is not None:
        q2 = int_upper_limit/h_i
    else:
        q2 = 1.0

    # set defaults
    small_q_integral = 0
    large_q_integral = 0

    # perform the integration in two pieces for a cubic spline: 0<=q<0.5 & 0.5<=q<1
    if q1 < 0.5:
        small_q_integral, sqi_err = quad(small_q_kernel, q1, np.min([0.5,q2]), args=(h_i))
        if q2 > 0.5:
            large_q_integral, lqi_err = quad(large_q_kernel, 0.5, np.min([1.0,q2]), args=(h_i))
    elif q1 >= 0.5:
        large_q_integral, lqi_err = quad(large_q_kernel, q1, np.min([1.0,q2]), args=(h_i))
    else:
        print('unknown integration limits')
        return np.nan

    # solid angle correction
    solid_angle_prefactor = 4*np.pi
    if (weight_distance is not None) & (weight_size is not None):
        solid_angle_prefactor = 2*np.pi*(1-weight_distance/np.sqrt(weight_distance**2 + weight_size**2))
        #solid_angle_prefactor = 0.5*(1-weight_distance/np.sqrt(weight_distance**2 + weight_size**2))

        # or use effective area weighting from Hopkins 2018 SN paper?
        # see equation 2 for little omega_b
        #A_i = np.pi*h_i**2
        #solid_angle_prefactor *= 0.5*(1 - 1/np.sqrt(1 + A_i/(np.pi*weight**2)))

    normalization = 1
    if normalize:
        n1, n1_err = quad(small_q_kernel, 0.0, 0.5, args=(h_i))
        n2, n2_err = quad(large_q_kernel, 0.5, 1.0, args=(h_i))
        normalization = 4*np.pi*(n1 + n2)

    volume_integral = solid_angle_prefactor*(small_q_integral + large_q_integral)/normalization

    return volume_integral


def localized_ram_pressure(
    particle_data,
    sat_position,
    sat_velocity,
    sat_rad,
    cyl_multiplier=3
):
    # rotate sat & gas coordinates to align with sat velocity
    # make sat velocity in the arbitrary +z direction
    z_direction = np.array([0,0,1])
    rot_matrix = mf.rotation_matrix_from_vectors(sat_velocity, z_direction)
    gas_coords_rotated = ut.basic.coordinate.get_coordinates_rotated(
        particle_data['gas']['position'], rotation_tensor=rot_matrix)
    sat_coords_rotated = ut.basic.coordinate.get_coordinates_rotated(
        sat_position, rotation_tensor=rot_matrix)
    sat_vel_rotated = ut.basic.coordinate.get_coordinates_rotated(
        sat_velocity, rotation_tensor=rot_matrix)

    # check to make sure rotation worked
    sat_vel_mag = np.linalg.norm(sat_velocity)*ut.constant.kpc_per_km*ut.constant.sec_per_Gyr
    #assert np.isclose(sat_vel_mag, sat_vel_rotated[2], atol=10, equal_nan=True), 'rotation error'
    assert np.isclose(sat_vel_rotated[0], 0, atol=0.1, equal_nan=True), 'rotation error'
    assert np.isclose(sat_vel_rotated[1], 0, atol=0.1, equal_nan=True), 'rotation error'
    assert sat_vel_rotated[2] > 0, 'rotation error'

    # if sat_rad is invalid (e.g., -1 or NaN) then set to 1 kpc as default
    if not sat_rad > 0:
        sat_rad = 1

    # height: velocity*25 Myr (avg snapshot spacing) in kpc
    #sat_vel_mag = np.linalg.norm(sat_velocity)*ut.constant.kpc_per_km*ut.constant.sec_per_Gyr
    cyl_height = 0.025*sat_vel_mag
    # radius: 2-5X R*50? in kpc
    cyl_radius = cyl_multiplier*sat_rad

    # get gass kernel radii and total masses
    gas_size = particle_data['gas'].prop('size')
    gas_mass = particle_data['gas']['mass']

    # translate coordinate system so the origin is at center of closest face of the cylinder
    # start closest face of the cylinder 2-5X R*50 from the sat
    #origin_shift = np.array([0,0,radius_of_cyl])
    # CHANGE TO?
    origin_shift = np.array([0,0,cyl_radius+0.5*cyl_height])
    new_origin = sat_coords_rotated + origin_shift
    gas_coords_shifted = gas_coords_rotated - new_origin

    # use a simple spherical distance from the cylinder center to determine gas integration
    max_cyl_dimension = np.max([cyl_height/2, cyl_radius])
    gas_distance_to_cyl = np.linalg.norm(gas_coords_shifted, axis=1)

    # if the gas cell center is within the cylinder, just use its mass
    in_cyl_mask = gas_distance_to_cyl < max_cyl_dimension
    mass_center_within = np.sum(gas_mass[in_cyl_mask])
    
    # if the gas cell partially overlaps the cylinder, weight the integral 
    # according to the approximate solid angle subtended by the cylinder
    overlap_mask = (gas_distance_to_cyl < gas_size + max_cyl_dimension) & ~in_cyl_mask
    vector_integrate_kernel = np.vectorize(integrate_kernel)
    inner_int_limit = gas_distance_to_cyl - max_cyl_dimension

    mass_cell_overlap = vector_integrate_kernel(
        gas_size[overlap_mask], 
        inner_int_limit[overlap_mask],
        gas_distance_to_cyl[overlap_mask], 
        max_cyl_dimension)

    # multiply normalized integral by gas cell mass
    mass_cell_overlap *= gas_mass[overlap_mask]
    cyl_gas_mass = np.sum(mass_center_within + mass_cell_overlap)

    # CONVERT TO NUMBER DENSITY ASSUMING HYDROGEN COMPOSITION
    cyl_volume = np.pi*cyl_height*cyl_radius**2
    cyl_gas_mass_density = cyl_gas_mass/cyl_volume# Msun/kpc^3
    cyl_gas_number_density = cyl_gas_mass_density*ut.constant.hydrogen_per_sun*ut.constant.kpc_per_cm**3

    # GET AN AVERAGE VELOCITY OF ALL GAS CELLS THAT CONTRIBUTED TO DENSITY
    new_mask = in_cyl_mask | overlap_mask

    # get only velocity component along satellite velocity (z direction) in km/s
    # wrt galaxy velocity (vgas-vgal)
    cylinder_velocity = particle_data['gas']['velocity'][new_mask]
    cylinder_velocity_rotated = ut.basic.coordinate.get_coordinates_rotated(
        cylinder_velocity, rotation_tensor=rot_matrix)
    cylinder_velocity_wrt_sat = np.abs(cylinder_velocity_rotated[:,2] - sat_vel_rotated[2])

    # save cylinder parameters, densities, and velocities in a dictionary
    cylinder_dict = {
        'height':np.array(cyl_height),
        'radius':np.array(cyl_radius),
        'n.gas.in.cyl':np.array(np.sum(new_mask)),
        'median.gas.size':np.array(np.median(gas_size[new_mask])),
        'density':np.array(cyl_gas_number_density),
        'velocity.median':np.array(np.median(cylinder_velocity_wrt_sat)),
        'velocity.mean':np.array(np.mean(cylinder_velocity_wrt_sat)),
        'velocity.low90':np.array(np.percentile(cylinder_velocity_wrt_sat, 5)),
        'velocity.high90':np.array(np.percentile(cylinder_velocity_wrt_sat, 95)),
    }

    return cylinder_dict

def localized_ram_pressure_old(
    particle_data,
    sat_position,
    sat_velocity,
    sat_rad,
    cyl_multiplier=3
):
    # rotate sat & gas coordinates to align with sat velocity
    # make sat velocity in the arbitrary +z direction
    z_direction = np.array([0,0,1])
    rot_matrix = mf.rotation_matrix_from_vectors(sat_velocity, z_direction)
    gas_coords_rotated = ut.basic.coordinate.get_coordinates_rotated(
        particle_data['gas']['position'], rotation_tensor=rot_matrix)
    sat_coords_rotated = ut.basic.coordinate.get_coordinates_rotated(
        sat_position, rotation_tensor=rot_matrix)
    sat_vel_rotated = ut.basic.coordinate.get_coordinates_rotated(
        sat_velocity, rotation_tensor=rot_matrix)

    # check to make sure rotation worked
    sat_vel_mag = np.linalg.norm(sat_velocity)*ut.constant.kpc_per_km*ut.constant.sec_per_Gyr
    #assert np.isclose(sat_vel_mag, sat_vel_rotated[2], atol=10, equal_nan=True), 'rotation error'
    assert np.isclose(sat_vel_rotated[0], 0, atol=0.1, equal_nan=True), 'rotation error'
    assert np.isclose(sat_vel_rotated[1], 0, atol=0.1, equal_nan=True), 'rotation error'
    assert sat_vel_rotated[2] > 0, 'rotation error'

    # if sat_rad is invalid (e.g., -1 or NaN) then set to 1 kpc as default
    if not sat_rad > 0:
        sat_rad = 1

    # translate coordinate system so the origin is at center of closest face of the cylinder
    # start closest face of the cylinder 2-5X R*50 from the sat
    origin_shift = np.array([0,0,cyl_multiplier*sat_rad])
    new_origin = sat_coords_rotated + origin_shift
    gas_coords_shifted = gas_coords_rotated - new_origin

    # height: velocity*25 Myr (avg snapshot spacing) in kpc
    #sat_vel_mag = np.linalg.norm(sat_velocity)*ut.constant.kpc_per_km*ut.constant.sec_per_Gyr
    height_of_cyl = 0.025*sat_vel_mag

    # radius: 2-5X R*50? in kpc
    radius_of_cyl = cyl_multiplier*sat_rad

    # cylindrical mask to define gas parcel
    # assuming the center of one cylinder face is at the new coordinate origin
    height_mask = (gas_coords_shifted[:,2] > 0) & (gas_coords_shifted[:,2] < height_of_cyl)
    radius_mask = np.sqrt(gas_coords_shifted[:,0]**2 + gas_coords_shifted[:,1]**2) < radius_of_cyl

    # get only velocity component along satellite velocity (z direction) in km/s
    # wrt galaxy velocity (vgas-vgal)
    cylinder_velocity = particle_data['gas']['velocity'][height_mask & radius_mask]
    cylinder_velocity_rotated = ut.basic.coordinate.get_coordinates_rotated(
        cylinder_velocity, rotation_tensor=rot_matrix)
    cylinder_velocity_wrt_sat = cylinder_velocity_rotated[:,2] - sat_vel_rotated[2]


    ####################################
    # volume-weighted median/quantiles #
    ####################################

    # rank order by size of gas cells with np.argsort(gas_size)
    gas_size = particle_data['gas'].prop('size')[height_mask & radius_mask]
    indx = np.argsort(gas_size)
    cylinder_density = particle_data['gas'].prop('number.density')[height_mask & radius_mask]
    cylinder_density = cylinder_density[indx]
    cylinder_velocity_wrt_sat = cylinder_velocity_wrt_sat[indx]

    # get index of median value
    if (len(cylinder_density)%2) == 0:
        c = int(len(cylinder_density)/2 - 1)
    else:
        c = int((len(cylinder_density) - 1)/2)
    
    if len(cylinder_density) > 2:
        median_gas_size = gas_size[indx][c]
        low_95 = int(np.round((len(cylinder_density)-1)*0.025))
        hi_95 = int(np.round((len(cylinder_density)-1)*0.975))

        cylinder_density_median = cylinder_density[c]
        cylinder_density_low95 = cylinder_density[low_95]
        cylinder_density_high95 = cylinder_density[hi_95]

        cylinder_velocity_median = cylinder_velocity_wrt_sat[c]
        cylinder_velocity_low95 = cylinder_velocity_wrt_sat[low_95]
        cylinder_velocity_high95 = cylinder_velocity_wrt_sat[hi_95]

    elif len(cylinder_density) > 0:
        median_gas_size = gas_size[indx][c]
        cylinder_density_median = cylinder_density[c]
        cylinder_density_low95 = np.nan
        cylinder_density_high95 = np.nan

        cylinder_velocity_median = cylinder_velocity_wrt_sat[c]
        cylinder_velocity_low95 = np.nan
        cylinder_velocity_high95 = np.nan
        
    else:
        median_gas_size = np.nan
        cylinder_density_median = np.nan
        cylinder_density_low95 = np.nan
        cylinder_density_high95 = np.nan

        cylinder_velocity_median = np.nan
        cylinder_velocity_low95 = np.nan
        cylinder_velocity_high95 = np.nan

    #########################################
    ### simple median/average/percentiles ###
    #########################################

    # get median and 95% limits on density within the cylinder in atoms/cm^3
    cylinder_density = particle_data['gas'].prop('number.density')[height_mask & radius_mask]
    cylinder_density_mean_s = np.nanmean(cylinder_density)
    cylinder_density_median_s = np.nanmedian(cylinder_density)
    cylinder_density_low95_s = np.nanpercentile(cylinder_density, 2.5)
    cylinder_density_high95_s = np.nanpercentile(cylinder_density, 97.5)

    # get median and 95% limits on velocity wrt satellite within the cylinder in km/s
    cylinder_velocity_median_s = np.nanmedian(cylinder_velocity_wrt_sat)
    cylinder_velocity_mean_s = np.nanmedian(cylinder_velocity_wrt_sat)
    cylinder_velocity_low95_s = np.nanpercentile(cylinder_velocity_wrt_sat, 2.5)
    cylinder_velocity_high95_s = np.nanpercentile(cylinder_velocity_wrt_sat, 97.5)


    # save cylinder parameters, densities, and velocities in a dictionary
    cylinder_dict = {
        'height':np.array(height_of_cyl),
        'radius':np.array(radius_of_cyl),
        'dist.to.cyl':np.array(origin_shift[2]),
        'n.gas.in.cyl':np.array(np.sum(height_mask & radius_mask)),
        'median.gas.size':median_gas_size,
        'density.median':np.array(cylinder_density_median),
        'density.low95':np.array(cylinder_density_low95),
        'density.high95':np.array(cylinder_density_high95),
        'velocity.median':np.array(cylinder_velocity_median),
        'velocity.low95':np.array(cylinder_velocity_low95),
        'velocity.high95':np.array(cylinder_velocity_high95),
        'density.median.simple':np.array(cylinder_density_median_s),
        'density.mean.simple':np.array(cylinder_density_mean_s),
        'density.low95.simple':np.array(cylinder_density_low95_s),
        'density.high95.simple':np.array(cylinder_density_high95_s),
        'velocity.median.simple':np.array(cylinder_velocity_median_s),
        'velocity.mean.simple':np.array(cylinder_velocity_mean_s),
        'velocity.low95.simple':np.array(cylinder_velocity_low95_s),
        'velocity.high95.simple':np.array(cylinder_velocity_high95_s),
    }

    return cylinder_dict
