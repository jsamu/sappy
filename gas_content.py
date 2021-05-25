import numpy as np
import pandas as pd
import utilities as ut
from collections import defaultdict
from astropy.utils.console import ProgressBar


class GalaxyGas():
    """
    Class to associate gas particles with subhalos and store subhalo gas
    properties.

    Parameters
    ----------
    gas_particle_data : dict
        dictionary of gas particle properties
    galaxy_table : dict
        pandas dataframe or other dictionary-like object containing subhalo info
        at the same snapshot as gas_particle_data. if no table available, must
        set sat_position/velocity/rad/vel_max/vel_std.
    radius_type : str
        type of radius listed in galaxy_table inside of which to search for gas
        particles. radius gets multiplied by radius_factor in calculation.
    radius_factor : float or int
        factor by which to multiply radius of search for a subhalo
    vel_factor : float or int
        factor by which to multiply velocity search to avoid spurious particle
        assignment
    sat_position : array
        3D subhalo position(s) in cartesian simulation box coordinates of shape
        (Nsub,3). only use if galaxy_table is None.
    sat_velocity : array
        3D subhalo velocity(s) in cartesian simulation box coordinates, has 
        shape (Nsub,3). only use if galaxy_table is None.
    sat_rad : array
        subhalo radii inside of which to search for gas particles, has shape 
        (Nsub). gets multiplied by radius_factor in calculation.
    sat_vel_max : array
        subhalo vel_circ_max inside of which to search for gas particles, has 
        shape (Nsub). gets multiplied by vel_factor in calculation.
    sat_vel_std : array
        subhalo velocity dispersion inside of which to search for gas particles,
        has shape (Nsub). gets multiplied by vel_factor in calculation.
    """
    def __init__(
        self, 
        gas_particle_data, 
        galaxy_table=None, 
        radius_type='star.radius.50',
        radius_factor=2, 
        vel_factor=2, 
        sat_position=None, 
        sat_velocity=None, 
        sat_rad=None, 
        sat_vel_max=None, 
        sat_vel_std=None
    ):
        """
        Store particle array/catalog indices of particles assigned to subhalos.
        """
        if galaxy_table is not None:
            self.sat_gas_ind = self.get_sat_gas_indices(
                gas_particle_data, galaxy_table, radius_type=radius_type,
                radius_factor=radius_factor, vel_factor=vel_factor
            )
        else:
            self.sat_gas_ind = self.get_sat_gas_indices(
                gas_particle_data, radius_factor=radius_factor, 
                vel_factor=vel_factor, sat_position=sat_position, 
                sat_velocity=sat_velocity, sat_rad=sat_rad, 
                sat_vel_max=sat_vel_max, sat_vel_std=sat_vel_std
            )            
        self.gas_props = None

    def get_gas_props(self, gas_particle_data):
        """
        Save subhalo gas properties based on assigned indices.
        """
        assert self.sat_gas_ind is not None
        gas_props = {
            'gas.mass.total':self.total_gas_mass(gas_particle_data),
            'hydrogen.mass.total':self.total_hydrogen_mass(gas_particle_data),
            'neutral.hydrogen.mass':self.neutral_hydrogen_mass(gas_particle_data)
        }

        gas_by_temp = self.gas_mass_by_temp(gas_particle_data)
        avg_gas_props = self.get_avg_gas_props(gas_particle_data)
        gas_props = {**gas_props, **gas_by_temp, **avg_gas_props}
        self.gas_props = gas_props

        return gas_props

    def get_sat_table_with_gas_props(
        self, gas_particle_data, galaxy_table, file_name=None):
        """
        Add gas properties to galaxy table and save to text file.
        """
        if self.gas_props is None:
            gas_props = self.get_gas_props(gas_particle_data)
        for gas_key in self.gas_props.keys():
            galaxy_table[gas_key] = self.gas_props[gas_key]
        if file_name is not None:
            galaxy_table.to_csv(file_name, sep=' ', index=False)

        return galaxy_table

    def get_sat_gas_indices(
        self, 
        gas_particle_data, 
        galaxy_table=None, 
        radius_type='star.radius.50',
        radius_factor=2, 
        vel_factor=2, 
        sat_position=None, 
        sat_velocity=None, 
        sat_rad=None, 
        sat_vel_max=None, 
        sat_vel_std=None,
        progress_bar=False
    ):
        """
        Assign gas particles to subhalo(s).
        """
        if galaxy_table is not None:
            # get subhalo/galaxy properties from table
            # distances and velocities are host-centric
            sat_position = np.dstack((np.array(galaxy_table['r.x']), 
                np.array(galaxy_table['r.y']), np.array(galaxy_table['r.z'])))[0]
            sat_velocity = np.dstack((np.array(galaxy_table['v.x']), 
                np.array(galaxy_table['v.y']), np.array(galaxy_table['v.z'])))[0]
            sat_rad = np.array(galaxy_table[radius_type])
            sat_vel_max = np.array(galaxy_table['vel.circ.max'])
            sat_vel_std = np.array(galaxy_table['vel.std'])

            # store all gas particle distances and velocities in new variables
            gas_particle_pos = gas_particle_data.prop('host.distance')
            gas_particle_vel = gas_particle_data.prop('host.velocity')

        # define distance and velocity cutoffs
        radius_limit = radius_factor*sat_rad
        velocity_limit = vel_factor*np.max((sat_vel_std, sat_vel_max), axis=0)

        # store all gas particle distances and velocities in new variables
        gas_particle_pos = gas_particle_data['position']
        gas_particle_vel = gas_particle_data['velocity']

        def find_indices(
            sat_position_, gas_positions_, radius_limit_, sat_velocity_, 
            gas_velocities_, velocity_limit_):
            """
            Get indices of gas particles within set distance and velocity limits.
            """
            pos_mask = ut.coordinate.get_distances(sat_position_, gas_positions_, 
                total_distance=True) < radius_limit_
            vel_mask = ut.coordinate.get_distances(sat_velocity_, gas_velocities_, 
                total_distance=True) < velocity_limit_
            return np.where(pos_mask & vel_mask)[0]

        if len(sat_position) == 3:
            # if only one subhalo, make these quantities iterable
            sat_position = np.array([sat_position])
            sat_velocity = np.array([sat_velocity])
            radius_limit = np.array([radius_limit])
            velocity_limit = np.array([velocity_limit])

        # set up empty list to save assigned gas particle indices
        sat_gas_indices = []

        if progress_bar:
            # use Astropy progress bar to visualize run time interactively
            with ProgressBar(len(sat_position)) as bar:
                for i,sat_pos_i in enumerate(sat_position):
                    ginds = find_indices(
                        sat_pos_i, gas_particle_pos, radius_limit[i], 
                        sat_velocity[i], gas_particle_vel, velocity_limit[i]
                    )
                    sat_gas_indices.append(ginds)
                    bar.update()
        else:
            for i,sat_pos_i in enumerate(sat_position):
                ginds = find_indices(
                    sat_pos_i, gas_particle_pos, radius_limit[i], 
                    sat_velocity[i], gas_particle_vel, velocity_limit[i]
                )
                sat_gas_indices.append(ginds)

        return np.array(sat_gas_indices)

    def total_gas_mass(self, gas_particle_data):
        """
        Get total gas mass associated with a subhalo(s).
        """
        sat_gas_masses = []
        for sat in self.sat_gas_ind:
            sat_gas_masses.append(np.sum(gas_particle_data['mass'][sat]))
        return np.array(sat_gas_masses)

    def gas_mass_by_temp(self, gas_particle_data, low_temp=1e4, high_temp=1e5):
        """
        Separate subhalo(s) gas mass by temperature.
        """
        mass_by_temp = defaultdict(list)
        for sat in self.sat_gas_ind:
            all_mass = gas_particle_data['mass'][sat]
            all_temp = gas_particle_data['temperature'][sat]
            cold_mask = all_temp <= low_temp
            warm_mask = (all_temp > low_temp) & (all_temp <= high_temp)
            hot_mask = all_temp > high_temp
            mass_by_temp['cold.gas.mass'].append(np.sum(all_mass[cold_mask]))
            mass_by_temp['warm.gas.mass'].append(np.sum(all_mass[warm_mask]))
            mass_by_temp['hot.gas.mass'].append(np.sum(all_mass[hot_mask]))
        mass_by_temp['cold.gas.mass'] = np.array(mass_by_temp['cold.gas.mass'])
        mass_by_temp['warm.gas.mass'] = np.array(mass_by_temp['warm.gas.mass'])
        mass_by_temp['hot.gas.mass'] = np.array(mass_by_temp['hot.gas.mass'])
        return mass_by_temp

    def total_hydrogen_mass(self, gas_particle_data):
        """
        Get total hydrogen mass associated with a subhalo(s).
        """
        sat_h_gas_masses = []
        for sat in self.sat_gas_ind:
            sat_h_gas_masses.append(np.sum(gas_particle_data.prop('mass.hydrogen')[sat]))
        return np.array(sat_h_gas_masses)

    def neutral_hydrogen_mass(self, gas_particle_data):
        """
        Get total neutral hydrogen mass associated with a subhalo(s).
        """
        sat_nh_gas_masses = []
        for sat in self.sat_gas_ind:
            sat_nh_gas_masses.append(np.sum(gas_particle_data.prop('mass.hydrogen.neutral')[sat]))
        return np.array(sat_nh_gas_masses)

    def get_avg_gas_props(self, gas_particle_data):
        """
        Get the average temp & density of all gas associated with a subhalo(s).
        """
        sat_gas_avg_temp = []
        sat_gas_avg_density = []
        for sat in self.sat_gas_ind:
            sat_gas_avg_temp.append(np.nanmean(gas_particle_data['temperature'][sat]))
            sat_gas_avg_density.append(np.nanmean(gas_particle_data['density'][sat]))
        return {
            'avg.gas.density':np.array(sat_gas_avg_density),
            'avg.gas.temperature':np.array(sat_gas_avg_temp)
        }
    