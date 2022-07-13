import numpy as np
import pandas as pd
import utilities as ut
import gizmo_analysis as gizmo
from satellite_analysis import math_funcs as mf


def localized_ram_pressure(
    sim_name,
    particle_data,
    snapshot_index,
    galaxy_properties,
    galaxy_histories,
    cyl_multiplier=3
):
    for sat_index, tree_index in enumerate(galaxy_properties['tree.index']):
        # get satellite properties from infall history
        sat_position = galaxy_histories[sim_name][0]['position'][snapshot_index, sat_index]
        sat_velocity = galaxy_histories[sim_name][0]['velocity'][snapshot_index, sat_index]
        sat_rad = galaxy_histories[sim_name][0]['star.radius.50'][snapshot_index, sat_index]

        # rotate sat & gas coordinates to align with sat velocity
        # make sat velocity in the +z direction
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
