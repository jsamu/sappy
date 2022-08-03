import numpy as np
import pandas as pd
from sappy import halo_reader as hr
from sappy import satellite_io as sio


# load in the merger tree data and select galaxies based on mass/distance
m12_st = hr.SatelliteTree(
    directory_list=['/path/to/sim/'],#can work with multiple sims at once
    host_name_list=['name_of_sim'],#same as above
    mask_names=['star.mass'],#type of mask to use, see satellite_io.py for options
    star_mass_limit=1e5,#specify a lower limit
    #star_mass_limits=[],#or specify a lower and upper limit
    radius_limits=[5,300],#same idea as above, and keep 5 as a lower limit
    assign_species=True,#whether or not you want to load star catalog data
    snapshots_to_load=np.arange(1,601,1),#list or integer of snapshots to load in star data
    snapshots_to_mask=600#use this to load in many snaps but select sats at a single snap
)

# make functions that operate on a single snapshot
# the functions need to have at least these arguments: hal, hal_mask=None, host_str='host.'
def get_sat_props(hal, hal_mask=None, host_str='host.'):
    mpeak, mpeaksnap = v_peak(
        hal, hal_mask, host_str, hal_property='mass', return_snap=True)
    return {
        'tree.index':np.where(hal_mask)[0],
        'star.mass':hal.prop('star.mass')[hal_mask],
        'mass.peak':mpeak,
        'mass.peak.snapshot':mpeaksnap, 
        'velocity':hal.prop(host_str+'velocity')[hal_mask], 
        'position':hal.prop(host_str+'distance')[hal_mask],
        'm200m':hal.prop('mass')[hal_mask], 
        'mvir':hal.prop('mass.vir')[hal_mask], 
        'vel.circ.max':hal.prop('vel.circ.max')[hal_mask], 
        'vel.std':hal.prop('vel.std')[hal_mask],
        'radius.200m':hal.prop('radius')[hal_mask], 
        'host.distance.total':hal.prop(host_str+'distance.total')[hal_mask], 
        'star.radius.50':hal.prop('star.radius.50')[hal_mask],
        'star.form.time.50':hal['star.form.time.50'][hal_mask],
        'star.form.time.95':hal['star.form.time.95'][hal_mask],
        'star.form.time.100':hal['star.form.time.100'][hal_mask]
    }

def v_peak(hal, hal_mask=None, host_str='host.', hal_property='vel.circ.max', return_snap=False):
    all_snapshots = np.arange(0,601,1)
    initial_ids = np.where(hal_mask)[0]
    tracked_ids = sio.halo_track(hal, initial_ids, snapshot_list=all_snapshots)
    tracked_vmax = np.array([hal[hal_property][ids.astype(int)] for ids in tracked_ids])
    peak_vmax = np.max(tracked_vmax, axis=0)
    
    if return_snap:
        peak_vmax_snapshot = []
        for i,peak in enumerate(peak_vmax):
            peak_vmax_snapshot.append(all_snapshots[-1+np.where(tracked_vmax[:,i]==peak)[0][0]])
        return peak_vmax, peak_vmax_snapshot
    else:
        return peak_vmax

# get properties of each selected satellite over time
m12_sat_props = sio.loop_hal(m12_st, 'star.mass', get_sat_props)

# save these properties to a text file
for host in m12_sat_props.keys():
    x = pd.DataFrame({
        'tree.index':np.array(m12_sat_props[host][0]['tree.index']),
        'star.mass':np.array(m12_sat_props[host][0]['star.mass']),
        'mass.peak':np.array(m12_sat_props[host][0]['mass.peak']),
        'mass.peak.snapshot':np.array(m12_sat_props[host][0]['mass.peak.snapshot']),
        'r.x':np.array(m12_sat_props[host][0]['position'])[:,0],
        'r.y':np.array(m12_sat_props[host][0]['position'])[:,1],
        'r.z':np.array(m12_sat_props[host][0]['position'])[:,2],
        'v.x':np.array(m12_sat_props[host][0]['velocity'])[:,0],
        'v.y':np.array(m12_sat_props[host][0]['velocity'])[:,1],
        'v.z':np.array(m12_sat_props[host][0]['velocity'])[:,2],
        'm200m':np.array(m12_sat_props[host][0]['m200m']),
        'mvir':np.array(m12_sat_props[host][0]['mvir']),
        'vel.circ.max':np.array(m12_sat_props[host][0]['vel.circ.max']),
        'vel.std':np.array(m12_sat_props[host][0]['vel.std']),
        'radius.200m':np.array(m12_sat_props[host][0]['radius.200m']),
        'host.distance.total':np.array(m12_sat_props[host][0]['host.distance.total']),
        'star.radius.50':np.array(m12_sat_props[host][0]['star.radius.50']),
        'star.form.time.50':np.array(m12_sat_props[host][0]['star.form.time.50']),
        'star.form.time.95':np.array(m12_sat_props[host][0]['star.form.time.95']),
        'star.form.time.100':np.array(m12_sat_props[host][0]['star.form.time.100'])
        })
    sat_props_df = pd.DataFrame(x)
    sat_props_df.to_csv('properties.txt', sep=' ', index=False)

print(sat_props_df)