import numpy as np
import pandas as pd
import utilities as ut
from collections import defaultdict


class GalaxyGas():
    def __init__(
        self, gas_particle_data, galaxy_table, radius_type='star.radius.50',
        radius_factor=2, vel_factor=2):
        # needs to be gas particle data only, and table read from file only
        self.sat_gas_ind = self.get_sat_gas_indices(
            gas_particle_data, galaxy_table, radius_type=radius_type,
            radius_factor=radius_factor, vel_factor=vel_factor
        )
        self.gas_props = None

    def get_gas_props(self, gas_particle_data):
        gas_props = {
            'gas.mass.total':self.total_gas_mass(gas_particle_data),
            'hydrogen.mass.total':self.total_hydrogen_mass(gas_particle_data),
            'neutral.hydrogen.mass':self.neutral_hydrogen_mass(gas_particle_data)
        }

        gas_by_temp = self.gas_mass_by_temp(gas_particle_data)
        gas_props = {**gas_props, **gas_by_temp}
        #gas_props = gas_props | gas_by_temp
        self.gas_props = gas_props

        return gas_props

    def get_sat_table_with_gas_props(
        self, gas_particle_data, galaxy_table, file_name=None):
        # add gas properties to galaxy table
        if self.gas_props is None:
            gas_props = self.get_gas_props(gas_particle_data)
        for gas_key in self.gas_props.keys():
            galaxy_table[gas_key] = gas_props[gas_key]
        if file_name is not None:
            galaxy_table.to_csv(file_name, sep=' ', index=False)

        return galaxy_table

    def get_sat_gas_indices(
        self, gas_particle_data, galaxy_table, radius_type='star.radius.50',
        radius_factor=2, vel_factor=2):
        # galaxy_table coordinates are wrt host in box basis
        sat_position = np.dstack((np.array(galaxy_table['r.x']), 
            np.array(galaxy_table['r.y']), np.array(galaxy_table['r.z'])))[0]
        sat_velocity = np.dstack((np.array(galaxy_table['v.x']), 
            np.array(galaxy_table['v.y']), np.array(galaxy_table['v.z'])))[0]

        sat_rad = np.array(galaxy_table[radius_type])
        radius_limit = radius_factor*sat_rad
        sat_vel_max = np.array(galaxy_table['vel.circ.max'])
        sat_vel_std = np.array(galaxy_table['vel.std'])
        velocity_limit = vel_factor*np.max((sat_vel_std, sat_vel_max), axis=0)

        gas_particle_pos = gas_particle_data.prop('host.distance')
        gas_particle_vel = gas_particle_data.prop('host.velocity')

        sat_gas_indices = []
        for sat_pos, sat_vel, rad_lim, vel_lim in zip(
            sat_position, sat_velocity, radius_limit, velocity_limit):
            pos_mask = ut.coordinate.get_distances(sat_pos, gas_particle_pos, 
                total_distance=True) < rad_lim
            vel_mask = ut.coordinate.get_distances(sat_vel, gas_particle_vel, 
                total_distance=True) < vel_lim
            sat_gas_indices.append(np.where(pos_mask & vel_mask)[0])

        return sat_gas_indices

    def total_gas_mass(self, gas_particle_data):
        sat_gas_masses = []
        for sat in self.sat_gas_ind:
            sat_gas_masses.append(np.sum(gas_particle_data['mass'][sat]))
        return np.array(sat_gas_masses)

    def gas_mass_by_temp(self, gas_particle_data):
        mass_by_temp = defaultdict(list)
        for sat in self.sat_gas_ind:
            all_mass = gas_particle_data['mass'][sat]
            all_temp = gas_particle_data['temperature'][sat]
            cold_mask = all_temp <= 1e4
            warm_mask = (all_temp > 1e4) & (all_temp <= 1e5)
            hot_mask = all_temp > 1e5
            mass_by_temp['cold.gas.mass'].append(np.sum(all_mass[cold_mask]))
            mass_by_temp['warm.gas.mass'].append(np.sum(all_mass[warm_mask]))
            mass_by_temp['hot.gas.mass'].append(np.sum(all_mass[hot_mask]))
        return mass_by_temp

    def total_hydrogen_mass(self, gas_particle_data):
        sat_h_gas_masses = []
        for sat in self.sat_gas_ind:
            sat_h_gas_masses.append(np.sum(gas_particle_data.prop('mass.hydrogen')[sat]))
        return np.array(sat_h_gas_masses)

    def neutral_hydrogen_mass(self, gas_particle_data):
        sat_nh_gas_masses = []
        for sat in self.sat_gas_ind:
            sat_nh_gas_masses.append(np.sum(gas_particle_data.prop('mass.hydrogen.neutral')[sat]))
        return np.array(sat_nh_gas_masses)

    def avg_gas_temp(self, gas_particle_data):
        sat_gas_avg_temp = []
        for sat in self.sat_gas_ind:
            sat_gas_avg_temp.append(np.nanmean(gas_particle_data['temperature'][sat]))
        return np.array(sat_gas_avg_temp)

    def avg_gas_density(self, gas_particle_data):
        sat_gas_avg_density = []
        for sat in self.sat_gas_ind:
            sat_gas_avg_density.append(np.nanmean(gas_particle_data['density'][sat]))
        return np.array(sat_gas_avg_density)
    