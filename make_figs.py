from satellite_analysis import halo_reader as hr
from satellite_analysis import isotropic as iso
from satellite_analysis import spatial as spa
from satellite_analysis import plot_spatial as ps
from satellite_analysis import plot_angle as pa
from satellite_analysis import plot_kinematics as pk

redshifts_01 = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
redshifts_015 = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,
                0.11, 0.12, 0.13, 0.14, 0.15]
redshifts_02 = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1,
                0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2]

md_7100_dirs = ['/Users/jsamuel/Desktop/Latte/metal_diffusion/m12i_res7100',
                '/Users/jsamuel/Desktop/Latte/metal_diffusion/m12f_res7100',
                '/Users/jsamuel/Desktop/Latte/metal_diffusion/m12m_res7100']

dmo_7100_dirs = ['/Users/jsamuel/Desktop/Latte/dmo/m12i_res7100_dm',
                 '/Users/jsamuel/Desktop/Latte/dmo/m12f_res7100_dm',
                 '/Users/jsamuel/Desktop/Latte/dmo/m12m_res7100_dm']

prop_subset =  ['snapshot','host.index','am.phantom','mass.lowres','mass',
                'position','host.distance.total','host.distance','velocity',
                'host.velocity','star.mass','star.number','star.density.50',
                'star.radius.50','vel.circ.max','tid','vel.circ.max.fbcorr',
                'progenitor.main.index']

'''
md_57000_dirs = ['../metal_diffusion/m12i_res57000',
                 '../metal_diffusion/m12f_res57000',
                 '../metal_diffusion/m12m_res57000']
'''

# set up label names for plotting different masks of the same data
m12_names = ['m12i', 'm12f', 'm12m']

m12_names_dmo = ['m12i DMO', 'm12f DMO', 'm12m DMO']



############################
#### load halo catalogs ####
############################
'''
md_7100_01 = hr.SatelliteHalo(directory_list=md_7100_dirs,
                          redshift_list=redshifts_01,
                          host_name_list=m12_names,
                          mask_names=['star.mass', 'star.number', 'vel.circ.max', 'top.11'])

dmo_7100_01 = hr.SatelliteHalo(directory_list=dmo_7100_dirs,
                               redshift_list=redshifts_01,
                               host_name_list=m12_names_dmo,
                               mask_names=['number.sats', 'vel.circ.max', 'median.vel.circ.max', 'top.11'],
                               dmo=True,
                               baryon_sat=md_7100_01)
'''
#########################
### load merger trees ###
#########################

md_7100_01 = hr.SatelliteTree(directory_list=md_7100_dirs,
                              redshift_list=redshifts_01,
                              host_name_list=m12_names,
                              mask_names=['star.mass'],# 'star.number', 'vel.circ.max', 'top.11'],
                              prop_subset=prop_subset)
'''
dmo_7100_01 = hr.SatelliteTree(directory_list=dmo_7100_dirs,
                              redshift_list=redshifts_01,
                              host_name_list=m12_names_dmo,
                              mask_names=['number.sats', 'vel.circ.max', 'median.vel.circ.max', 'top.11'],
                              prop_subset=prop_subset,
                              dmo=True,
                              dmo_baryon_compare=md_7100_01)
'''

####################
### make figures ###
####################

################################################################################

### number of satellites plots

ps.nsat_v_time(md_7100_01, 'star.mass')
ps.nsat_v_time(md_7100_01, 'vel.circ.max')
ps.nsat_v_time(md_7100_01, 'star.number')


# needs a normalization option
ps.nsat_v_time2([md_7100_01, dmo_7100_01], ['star.number', 'median.vel.circ.max'])


### radial distribution
'''
ps.plot_radial_dist(md_7100_01, 'star.mass', dmo_7100_01, 'number.sat', norm=False)
ps.plot_radial_dist(md_7100_01, 'star.mass', dmo_7100_01, 'median.vel.circ.max', norm=False)
ps.plot_radial_dist(md_7100_01, 'star.mass', dmo_7100_01, 'median.vel.circ.max', norm=True)
'''

### axis ratio plots

ps.plot_axis_ratio(md_7100_01, 'star.mass')
ps.plot_axis_ratio(dmo_7100_01, 'median.vel.circ.max')
ps.plot_axis_ratio(dmo_7100_01, 'number.sats')

################################################################################

### rms height and radius plots

ps.plot_rms_height(md_7100_01, 'star.mass')
ps.plot_rms_height(dmo_7100_01, 'median.vel.circ.max')
ps.plot_rms_height(dmo_7100_01, 'number.sats')

ps.plot_rms_height2([md_7100_01, dmo_7100_01], ['star.mass', 'number.sats'])
ps.plot_rms_height2([md_7100_01, dmo_7100_01], ['star.mass', 'median.vel.circ.max'])

ps.plot_rms_z_r_vs_frac(md_7100_01, 'star.mass')
ps.plot_rms_z_r_vs_frac(dmo_7100_01, 'median.vel.circ.max')
ps.plot_rms_z_r_vs_frac(dmo_7100_01, 'number.sats')

# does not work
ps.plot_coadd_rms_z_r_vs_frac(md_7100_01, 'star.mass')
ps.plot_coadd_rms_z_r_vs_frac(dmo_7100_01, 'median.vel.circ.max')
ps.plot_coadd_rms_z_r_vs_frac(dmo_7100_01, 'number.sats')


### rms height vs enclosing radius

# does not work
ps.plot_coadd_rmsz_vs_r([md_7100_01, dmo_7100_01, md_7100_01, dmo_7100_01],
                        ['star.mass', 'number.sats', 'star.number', 'median.vel.circ.max'])


################################################################################

### enclosing angle plots

# angle enclosing 68% of satellites
pa.plot_angle_width(md_7100_01, 'star.mass')
pa.plot_angle_width(dmo_7100_01, 'median.vel.circ.max')
pa.plot_angle_width(dmo_7100_01, 'number.sats')

# fraction enclosed vs. angle
pa.plot_coadd_angle(md_7100_01, 'star.mass')
pa.plot_coadd_angle(dmo_7100_01, 'median.vel.circ.max')
pa.plot_coadd_angle(dmo_7100_01, 'number.sats')


# opening angle vs r50
pa.plot_angle_v_r([md_7100_01, dmo_7100_01], ['star.mass', 'median.vel.circ.max'], radius_fraction=0.5, isotropic=True)
pa.plot_angle_v_r([md_7100_01, dmo_7100_01], ['star.mass', 'number.sats'], radius_fraction=0.5, isotropic=True)
pa.plot_angle_v_r([md_7100_01, dmo_7100_01], ['vel.circ.max', 'vel.circ.max'], radius_fraction=0.5, isotropic=True)


# opening angle enclosing some % vs r%/r% (ratio of radii enclosing other %'s)

pa.plot_enc_angle_vs_r_ratio([md_7100_01, dmo_7100_01], ['star.mass', 'number.sats'], angle_frac=0.68, rfrac1=0.9, rfrac2=0.1)
pa.plot_enc_angle_vs_r_ratio([md_7100_01, dmo_7100_01], ['star.number', 'median.vel.circ.max'], angle_frac=0.68, rfrac1=0.9, rfrac2=0.1)

################################################################################



### velocity plots
'''
pk.plot_los_velocity(md_7100_01_mass)
pk.plot_los_velocity(dmo_7100_01_vmin, title=' vmin')
pk.plot_los_velocity(dmo_7100_01_nsat, title=' same $N_{sat}$')
'''
