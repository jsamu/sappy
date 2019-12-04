import numpy as np
import pandas as pd
import utilities as ut
import satellite_analysis as sappy
from satellite_analysis.satellite_io2 import MaskHalo
from satellite_analysis.satellite_io2 import LoopHalo


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


class SatelliteCatalog(SatelliteParameter, MaskHalo, LoopHalo):
    """
    Load and collect halo catalogs for different host halos for a range of
    redshifts into a single list of lists. Load in comparable observational data.
    Secondary functions in spatial.py, etc. can calculate and plot relevant
    metrics of the distributions.
    """

    def __init__(
        self, 
        sat_data, 
        host_name_list=None, 
        mask_names=None, 
        dmo=False, 
        baryon_sat=None, 
        star_mass_limit=None,
        radius_limit=None, 
        host_number=1,
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

        Returns
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

        # set type of object to be read by other functions
        if host_number > 1:
            self.sat_type = 'hal.lg.lite'
            self.hal_name = host_name_list
        else:
            self.sat_type = 'hal.lite'
            self.hal_name = np.array(host_name_list)

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
                print("No {} information provided, defaulting to z=0 snapshot.".format(time_key))
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

        # create and store masks
        if host_number > 1:
            if dmo:
                ### put a default mask here
                self.catalog_mask = self.mask_lg_dmo_cat(self)
                #self.num_sats = spa.total_sats(self, baryon_sat, mask_key='star.mass', radius=self.r_range[1])
                #self.med_v_circ = kin.med_velcircmax_z0(baryon_sat, mask_key='star.number')
                # don't need these for now so just set to 0
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
