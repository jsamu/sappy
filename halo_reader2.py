import numpy as np
import pandas as pd
import utilities as ut
from collections import defaultdict
import satellite_analysis as sappy
from satellite_analysis.satellite_io2 import MaskHalo


class SatelliteParameter(object):
    """
    Set plotting and calculation parameters that can be passed ubiquitously
    for all data sets to secondary functions.

    Attributes
    ----------
    n_iter : int
        Number of isotropic distributions to generate.
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
    a_width = 1
    a_bins = ut.binning.BinClass([0, 90], width=a_width).maxs
    a_bins_plt = 2*a_bins

    # other general parameters
    n_iter = 1000
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


class SatelliteSelect(SatelliteParameter, MaskHalo):
    """
    Masking and looping over halo catalogs and merger trees.
    """
    def __init__(
        self, 
        sat_data, 
        sat_type='hal',
        host_name_list=None, 
        host_number=1,
        dmo=False,
        mask_names=None, 
        star_mass_limit=None,
        radius_limit=None, 
        vel_circ_max_lim=None, 
        v_peak=None, 
        mass_peak=None, 
        mass_bound=None,
        radius_limits=None, 
        radius_bin_width=None,
        time_info_file_path=None,
        snapshot_indices=None,
        redshift_list=None,
        time_list=None
        ):
        """
        Parameters
        ----------
        sat_data : dictionary
            either a dictionary of merger trees or a dictionary of halo catalogs
            where the keys are simulation names
        sat_type : string
            type of data contained in sat_data, 'hal' for halo catalogs by 
            default, but can be 'tree' for merger trees
        redshift_list : list
            List of floats specifying which redshifts (hdf5 files) to read in
            from the directories.
        host_name_list : list
            List of strings specifying the name of each host halo that is being
            read in, used in plots.
        mask_names : list
            Strings specifying which masks to create for the trees.
        """

        assert sat_data is not None, "Need to pass sat_data as a dictionary or list of dictionaries."

        time_table = None

        if time_info_file_path is not None:
            time_table = pd.read_csv(time_info_file_path, sep=' ')

        time_dict = {'snapshot':snapshot_indices, 'redshift':redshift_list, 'time':time_list}

        for time_key in time_dict.keys():
            if time_dict[time_key] is not None:
                exec("self.{} = {}".format(time_key, list(time_dict[time_key])))
            elif time_table is not None:
                exec("self.{} = {}".format(time_key, list(time_table[time_key].values)))
            else:
                raise ValueError("No {} information provided.".format(time_key))
        del(time_table)

        # set or reset essential parameters
        self.mask_names = mask_names
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
        if radius_limit:
            self.r_range = [5, radius_limit]
            self.r_bins = ut.binning.BinClass(self.r_range, width=self.r_width).maxs
        if radius_bin_width:
            self.r_width = radius_bin_width
        if radius_limits:
            self.r_range = [radius_limits[0], radius_limits[1]]
            self.r_bins = ut.binning.BinClass(self.r_range, width=self.r_width).maxs

        # set type of object to be read by other functions
        if host_number > 1:
            self.sat_type = sat_type+'.lg'
            self.hal_name = host_name_list
        else:
            self.sat_type = sat_type
            self.hal_name = np.array(host_name_list)
        
        assert sat_type in ['hal', 'tree'], "The chosen sat_type is not recognized."

        if sat_type == 'hal':
            # create and store masks
            if host_number > 1:
                if dmo:
                    ### put a default mask here
                    self.catalog_mask = self.mask_lg_dmo_cat(self)
                    self.num_sats = 0
                    self.med_v_virc = 0
                else:
                    ### put a default mask here
                    self.catalog_mask = self.mask_lg_baryon_cat(self)

            else:
                if dmo:
                    self.num_sats = 0
                    self.med_v_virc = 0
                ### put a default mask here
                self.catalog_mask = self.mask_hosts_lite(self, sat_data=sat_data, mask_keys=mask_names, dmo=dmo)
        elif sat_type == 'tree':
            if host_number > 1:
                if dmo:
                    #self.tree_mask = self.mask_tree_lg_dmo(self)
                    pass                
                else:
                    self.tree_mask = self.mask_tree_lg(self)
            
            else:
                if dmo:
                    self.tree_mask = self.mask_tree_dmo(self)
                else:
                    self.tree_mask = self.mask_tree(self)
        else:
            raise ValueError("The chosen sat_type is not recognized.")

    def loop_hal(self, sat_data, mask_key, exec_func, **kwargs):
        '''
        Loop a function over a series of halo catalogs for each simulation in a 
        sat object.

        Parameters
        ----------
        sat : Satellite object
            class created in halo_reader.py
        mask_key : str
            keyword specifying the type of halo property to cut/mask halos on
        exec_func : function
            function to call repeatedly/pass sat to
        kwargs : dictionary
            dictionary of key value pairs where keys are the keywords of exec_func

        Returns
        -------
        loop_dict : dictionary
            stores the output of exec_func for each simulation and snapshot in sat
        '''
        loop_dict = defaultdict(list)
        if self.sat_type == 'tree':
            for host_name in sat_data.keys():
                for redshift_index in range(len(self.redshift)):
                    tree = sat_data[host_name]
                    tree_mask = self.tree_mask[host_name][redshift_index][mask_key]
                    loop_dict[host_name].append(exec_func(hal=tree, hal_mask=tree_mask, **kwargs))
        elif self.sat_type == 'tree.lg':
            for pair_name in sat_data.keys():
                for host_name, host_str in zip(self.hal_name[pair_name], ['host.', 'host2.']):
                    for redshift_index in range(len(self.redshift)):
                        tree = sat_data[pair_name]
                        tree_mask = self.tree_mask[pair_name][host_name][redshift_index][mask_key]
                        loop_dict[host_name].append(exec_func(hal=tree, hal_mask=tree_mask, host_str=host_str, **kwargs))
        elif self.sat_type == 'hal':
            for hal_name in self.hal_name:
                for redshift_index in range(len(self.redshift)):
                # this implicitly loops over the snapshots/redshifts that have been loaded
                #for hal, hal_mask in zip(sat.hal_catalog[hal_name], sat.catalog_mask[mask_key][hal_name]):
                    hal = sat_data[hal_name][redshift_index]
                    hal_mask = self.catalog_mask[hal_name][redshift_index][mask_key]
                    loop_dict[hal_name].append(exec_func(hal=hal, hal_mask=hal_mask, **kwargs))
        elif self.sat_type == 'hal.lg':
            for pair_name in sat_data.keys():
                for host_name, host_str in zip(self.hal_name[pair_name], ['host.', 'host2.']):
                    for redshift_index in range(len(self.redshift)):
                        hal = sat_data[pair_name][redshift_index]
                        hal_mask = self.catalog_mask[pair_name][host_name][redshift_index][mask_key]
                        loop_dict[host_name].append(exec_func(hal=hal, hal_mask=hal_mask, host_str=host_str, **kwargs))
        else:
            print('sat type not recognized')

        return loop_dict