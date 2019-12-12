from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import utilities as ut
from satellite_analysis import satellite_io as sio
from satellite_analysis import plot_general as pg
from satellite_analysis import spatial as spa
from satellite_analysis import kinematics as kin
from satellite_analysis import isotropic as iso
from satellite_analysis import observation as obs
from satellite_analysis import rand_axes as ra


class SatParam(object):
    """
    Set plotting and calculation parameters that can be passed ubiquitously
    for all data sets to secondary functions.
    """
    # timing info, defaults are present day
    snapshot = 600
    redshift = 0
    time = 13.8

    # radial bins to use for plotting/calculations
    r_width = 10
    r_range = [5, 300]
    r_bins = ut.binning.BinClass([0, r_range[1]], width=r_width).maxs
    r_bins = np.insert(r_bins, 0, 0)

    # angular bins to use for plotting/calculations
    a_width = 5
    a_bins = ut.binning.BinClass([0, 90], width=a_width).maxs
    a_bins_plt = 2*a_bins

    # other general parameters
    n_iter = 10000
    star_mass = [1e5, 1e10]
    star_number = 10.0
    star_density = 1e3
    low_res = 0.02
    vel_circ_max = 10
    abs_number = 11
    v_peak = 35
    mass_peak = 5.7e9
    m_bound = 1e7

    # completeness distance(s)
    d_complete = [150, 300]
    sm_complete = [1e5, 1e7]

class SatelliteTree(SatParam):
    """
    Load and collect halo catalogs for different host halos for a range of
    redshifts into a list of lists. Load in comparable observational data.
    Secondary functions in spatial.py, etc. can calculate and plot relevant
    metrics of the distributions. Inherits plotting and calculation parameters
    from SatParam class.
    """

    def __init__(
        self, directory_list, redshift_list=None, host_name_list=None,
        mask_names=None, prop_subset=None, dmo=False, dmo_baryon_compare=None,
        host_number=1, star_mass_limit=None, radius_limit=None,
        radius_limits=None, radius_bin_width=None, snapshot_indices=None,
        observation_dir=None, vel_circ_max_lim=None, mass_bound=None,
        v_peak=None, mass_peak=None, assign_species=True, isotropic=False,
        number_sats=None, time_list=None, time_info_file_path=None,
        redshift_limits=None):
        """
        Parameters
        ----------
        directory_list : list
            List of strings specifying where the hdf5 halo catalogs are.
        redshift_list : list
            List of floats specifying which redshifts (hdf5 files) to read in
            from the directories.
        host_name_list : list
            List of strings specifying the name of each host halo that is being
            read in, used in plots.
        mask_label_dict : dictionary
            Dictionary of host names (keys) matched to names for plotting with
            specific masks.
        mask_names : list
            Strings specifying which masks to create for the trees.
        prop_subset : list
            List of strings specifying properties output by halo_analysis 
            package to be kept in satellite analysis. If None, all properties 
            are kept in tree. If a list is supplied then all other properties in 
            tree that are not listed in prop_subset will be deleted from the 
            tree.
        dmo : boolean
            True if the data being loaded in are for dark matter only (DMO)
            simulations, False by default.
        dmo_baryon_compare : SatelliteTree object
            If intended analysis includes comparison of DMO and baryonic runs,
            pass baryonic SatelliteTree object as argument here to mask DMO on
            baryonic properties.
        host_number : int
            2 if the data being loaded in are for Local Group-like pairs of
            galaxies/halos, 1 by default for isolated host-halos.

        Attributes
        ----------
        merger_tree : dictionary
            Merger tree output by halo.io.IO.read_trees(). Each host halo
            corresponds to one key of this dictionary.
        tree_mask : dictionary
            Mask for the merger trees for each host, at each redshift, and for
            each mask condition.
        """
        # organize simulation and mask names
        self.sat_type = 'tree'
        self.hal_name = host_name_list
        self.mask_names = mask_names

        # get snapshot info from keywords if provided, if not get from a table
        # or use z=0 defaults
        time_table = None
        if time_info_file_path is not None:
            time_table = pd.read_csv(time_info_file_path, sep=' ')
        time_dict = {'snapshot':snapshot_indices, 'redshift':redshift_list, 'time':time_list}
        for time_key in time_dict.keys():
            if time_dict[time_key] is not None:
                exec("self.{} = {}".format(time_key, list(time_dict[time_key])))
            elif time_table is not None:
                if redshift_limits is not None:
                    z_mask = np.array(
                        (np.array(time_table['redshift'].values) >= redshift_limits[0]) & 
                        (np.array(time_table['redshift'].values) <= redshift_limits[1])
                        )
                else:
                    z_mask = np.ones(len(time_table['redshift'].values), dtype='bool')
                exec("self.{} = {}".format(time_key, list(time_table[time_key].values[z_mask])))
            else:
                print("No {} information provided, defaulting to z=0 snapshot.".format(time_key))
        del(time_table)

        if vel_circ_max_lim:
            self.vel_circ_max = vel_circ_max_lim

        if v_peak:
            self.v_peak = v_peak

        if mass_peak:
            self.mass_peak = mass_peak

        if mass_bound:
            self.m_bound = mass_bound

        if star_mass_limit:
            self.star_mass = [star_mass_limit, 1e10]

        if radius_bin_width:
            self.r_width = radius_bin_width

        if number_sats:
            self.abs_number = number_sats

        if radius_limit:
            self.r_range = [5, radius_limit]
            self.r_bins = ut.binning.BinClass(self.r_range, width=self.r_width).maxs
        elif radius_limits:
            self.r_range = [radius_limits[0], radius_limits[1]]
            self.r_bins = ut.binning.BinClass(self.r_range, width=self.r_width).maxs

        self.hal_label = sio.hal_label_names(self)

        # load trees. if dmo, the corrected circular velocity, etc. is added to 
        # tree.
        self.tree = sio.load_trees(directory_list, self.hal_name,
                                   prop_subset=prop_subset, dmo=dmo,
                                   host_number=host_number,
                                   assign_species=assign_species,
                                   snapshot_ind=self.snapshot)

        if dmo:
            # generate dmo masks
            if dmo_baryon_compare:
                # calculate values for comparison to baryonic runs.
                self.num_sats = spa.total_sats(self, dmo_baryon_compare, mask_key='star.mass', radius=self.r_range[1])
                self.med_v_circ = kin.med_velcircmax_z0(dmo_baryon_compare, mask_key='star.number')
            self.tree_mask = sio.mask_tree_dmo(self)
            # generate isotropic distributions
            #self.isotropic = sio.tree_iso(self, mask_names)

        # Local Group functionality
        elif host_number > 1:
            self.sat_type = 'tree.lg'
            self.tree_mask = sio.mask_tree_lg(self)

        else:
            # generate baryonic masks
            self.tree_mask = sio.mask_tree(self)
            # generate isotropic distributions
            if isotropic:
                self.isotropic = sio.tree_iso(self, mask_names)

        # observational data
        if observation_dir:
            self.observation = obs.load_obs(table_dir=observation_dir)
            self.observation_mask = obs.obs_mask(self.observation, star_mass=self.star_mass, r_range=self.r_range)

class SatelliteHalo(SatParam):
    """
    Load and collect halo catalogs for different host halos for a range of
    redshifts into a single list of lists. Load in comparable observational data.
    Secondary functions in spatial.py, etc. can calculate and plot relevant
    metrics of the distributions.
    """

    def __init__(
        self, directory_list, redshift_list=None, snapshot_indices=None, 
        time_list=None, host_name_list=None, 
        mask_names=None, dmo=False, baryon_sat=None, star_mass_limit=None,
        radius_limit=None, host_number=1, observation_dir=None,
        vel_circ_max_lim=None, v_peak=None, mass_peak=None, mass_bound=None,
        assign_species=True, radius_limits=None, radius_bin_width=None, 
        number_sats=None, time_info_file_path=None, redshift_limits=None):
        """
        Parameters
        ----------
        directory_list : list
            List of strings specifying where the hdf5 halo catalogs are.
        redshift_list : list
            List of floats specifying which redshifts (hdf5 files) to read in
            from the directories.
        host_name_list : list
            List of strings specifying the name of each host halo that is being
            read in, used in plots.
        mask_label_dict : dictionary
            Dictionary of host names (keys) matched to names for plotting with
            specific masks.
        mask_names : list
            Strings specifying which masks to create for the trees.
        dmo : boolean
            True if the data being loaded in are for dark matter only (DMO)
            simulations, False by default.
        baryon_sat : Satellite object
            Baryonic counterpart object, used to create masks for dark matter
            only simulations.

        Attributes
        -------
        halo_catalog : list
            Halo catalog dictionaries output by halo.io.Read.read_catalogs().
            The format is one overarching list whose elements are a list for each
            host halo, which is in turn a list of dictionaries at each input redshift.
        catalog_mask : list
            Boolean values that implement the masks of satellite_analysis.mask_cuts().
            The format is the same as halo_catalog, but instead of dictionaries as the
            most basic unit there are 1D boolean lists.
        """
        if host_number > 1:
            self.sat_type = 'hal.lg'
            self.hal_name = host_name_list
        else:
            self.sat_type = 'hal'
            self.hal_name = np.array(host_name_list)

        # get snapshot info from keywords if provided, if not get from a table
        # or use z=0 defaults
        time_table = None
        if time_info_file_path is not None:
            time_table = pd.read_csv(time_info_file_path, sep=' ')
        time_dict = {'snapshot':snapshot_indices, 'redshift':redshift_list, 'time':time_list}
        for time_key in time_dict.keys():
            if time_dict[time_key] is not None:
                exec("self.{} = {}".format(time_key, list(time_dict[time_key])))
            elif time_table is not None:
                if redshift_limits is not None:
                    z_mask = np.array(
                        (np.array(time_table['redshift'].values) >= redshift_limits[0]) & 
                        (np.array(time_table['redshift'].values) <= redshift_limits[1])
                        )
                else:
                    z_mask = np.ones(len(time_table['redshift'].values), dtype='bool')
                exec("self.{} = {}".format(time_key, list(time_table[time_key].values[z_mask])))
            else:
                print("No {} information provided, defaulting to z=0 snapshot.".format(time_key))
        del(time_table)

        if vel_circ_max_lim:
            self.vel_circ_max = vel_circ_max_lim

        if v_peak:
            self.v_peak = v_peak

        if mass_peak:
            self.mass_peak = mass_peak

        if mass_bound:
            self.m_bound = mass_bound

        if star_mass_limit:
            self.star_mass = [star_mass_limit, 1e10]

        if number_sats:
            self.abs_number = number_sats

        if radius_limit:
            self.r_range = [5, radius_limit]
            self.r_bins = ut.binning.BinClass(self.r_range, width=self.r_width).maxs

        if radius_bin_width:
            self.r_width = radius_bin_width

        if radius_limits:
            self.r_range = [radius_limits[0], radius_limits[1]]
            self.r_bins = ut.binning.BinClass(self.r_range, width=self.r_width).maxs

        self.hal_label = sio.hal_label_names(self)
        self.mask_names = mask_names

        # load halo catalogs

        if host_number > 1:
            # Local Group simulations
            if dmo:
                self.hal_catalog = sio.load_hals(directory_list, self.snapshot, self.hal_name, baryon_frac=True)
                #if baryon_sat is not None:
                #    self.num_sats = spa.total_sats(self, baryon_sat, mask_key='star.mass', radius=self.r_range[1])
                #    self.med_v_circ = kin.med_velcircmax_z0(baryon_sat, mask_key='star.mass')
                self.catalog_mask = sio.mask_lg_dmo_cat(self)
            else:
                self.hal_catalog = sio.load_hals(directory_list, self.snapshot, 
                    self.hal_name, host_number=host_number, assign_species=assign_species)
                self.catalog_mask = sio.mask_lg_baryon_cat(self)

        else:
            # m12/isolated Milky Way simulations
            if dmo:
                self.hal_catalog = sio.load_hals(directory_list, self.snapshot, self.hal_name, baryon_frac=True)
                #if baryon_sat is not None:
                #    self.num_sats = spa.total_sats(self, baryon_sat, mask_key='star.mass', radius=self.r_range[1])
                #    self.med_v_circ = kin.med_velcircmax_z0(baryon_sat, mask_key='star.mass')
            else:
                self.hal_catalog = sio.load_hals(directory_list, self.snapshot, self.hal_name)

            # generate masks
            self.catalog_mask = sio.mask_catalogs(self, mask_keys=mask_names, dmo=dmo)

        # generate isotropic distributions
        # self.isotropic = sio.hal_iso(self, mask_names=mask_names)

        # load observational data
        if observation_dir:
            self.observation = obs.load_obs(table_dir=observation_dir)
            self.observation_mask = obs.obs_mask(self.observation, star_mass=self.star_mass, r_range=self.r_range)
