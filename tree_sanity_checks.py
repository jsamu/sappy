import numpy as np
import pandas as pd
from collections import defaultdict
import utilities as ut
from satellite_analysis import halo_reader as hr
from satellite_analysis import spatial as spa
from satellite_analysis import satellite_io as sio


redshifts = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
snap_indices = [600, 585, 579, 573, 567, 561, 555, 550, 544, 539, 534]



# ELVIS info
lg_names = {'RomeoJuliet':['Romeo', 'Juliet'], 
            'ThelmaLouise':['Thelma', 'Louise'],
            'RomulusRemus':['Romulus', 'Remus']}

lg_dirs_pel = ['/home/jsamuel/scratch/m12_elvis/m12_elvis_RomeoJuliet_res3500',
               '/home/jsamuel/scratch/m12_elvis/m12_elvis_ThelmaLouise_res4000',
               '/home/jsamuel/scratch/m12_elvis/m12_elvis_RomulusRemus_res4000']

### Local Grop runs
lg_sh = hr.SatelliteHalo(directory_list=lg_dirs_pel,
                         redshift_list=redshifts,
                         host_name_list=lg_names,
                         mask_names=['star.mass'],
                         star_mass_limit=1e6,
                         host_number=2,
                         assign_species=True,
                         snapshot_list=snap_indices)

def fraction_missing(hal, hal_mask, host_str='host.'):
    negative_tid = hal.prop('tree.index') < 0
    missing = np.sum(hal_mask & negative_tid)
    total = np.sum(hal_mask)
    return missing/total

def distance_jump_check(hal, hal_mask, host_str='host.'):
    negative_tid = hal.prop('tree.index') < 0
    missing_sat_rad = hal.prop(host_str+'distance.total')[hal_mask & negative_tid]

    return missing_sat_rad

fraction = sio.loop_hal(lg_sh, 'star.mass', fraction_missing)
print(fraction)

distances = sio.loop_hal(lg_sh, 'star.mass', distance_jump_check)
print(distances)


missing_df = pd.DataFrame({'host':[], 'snapshot':[], 'host.distance.total':[],
                'star.mass':[], 'host.velocity':[], 'bound.mass':[]})

for host in all_angle.keys():
    x = pd.DataFrame({'host':np.full(snap_indices.size, host),
        'snapshot':snap_indices,
        'host.distance.total':,
        'host.velocity':,
        'star.mass':,
        'bound.mass':})
    missing_df = missing_df.append(x, ignore_index=True)

missing_df.to_csv('missing_sats_table.txt', sep=' ', index=False)