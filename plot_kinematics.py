import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from satellite_analysis import satellite_io as sio
from satellite_analysis import math_funcs as mf
from satellite_analysis import angular as ang
from satellite_analysis import kinematics as kin
from satellite_analysis import spatial as spa
from satellite_analysis import isotropic as iso
from satellite_analysis import plot_general as pg


CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

line_style = ['-.', '--', ':']

mark_style = ['.', '^', 'x']


def plot_los_velocity(sat, title=''):
    '''
    Plot the positions of the satellites in the x-z (in the MOI frame) plane,
    color-coded by approaching or receding velocity along the line of sight
    (positive MOI y-axis pointing into the figure).
    '''
    #sat_aligned_coords[host index][redshift index][xyz coordinate]
    sat_aligned_coords, sat_aligned_velocities = spa.sat_align_moi(sat)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(15, 5))
    fig.suptitle(title)
    ax1.set_ylabel('z\' [kpc]')
    plt.xlim(-350, 350)
    plt.ylim(-350, 350)
    
    for i, ax in enumerate([ax1, ax2, ax3]):
        ax.set_xlabel('x\' [kpc]')
        approaching = sat_aligned_velocities[i][0][:,1] < 0
        receding = sat_aligned_velocities[i][0][:,1] > 0
        ax.plot(sat_aligned_coords[i][0][:,0][approaching], sat_aligned_coords[i][0][:,2][approaching], 'b.', alpha=0.5, label=sat.host_name[i]+' approaching')
        ax.plot(sat_aligned_coords[i][0][:,0][receding], sat_aligned_coords[i][0][:,2][receding], 'r.', alpha=0.5, label=sat.host_name[i]+' receding')
    
        ax.legend()
    plt.tight_layout()
    plt.show()

def plot_los_velocity2(sat_list):
    '''
    Compare the positions of the satellites in the x-z (in the MOI frame) plane,
    color-coded by approaching or receding velocity along the line of sight
    (positive MOI y-axis pointing into the figure) for len(sat_list) number of
    halo catalogs and len(sat_list[0].host_name) host halos.
    '''
    #sat_aligned_coords[host index][redshift index][xyz coordinate]
    fig, ax = plt.subplots(len(sat_list), 3, sharex=True, sharey=True)
    plt.xlim(-350, 350)
    plt.ylim(-350, 350)

    for i, sat in enumerate(sat_list):
        sat_aligned_coords, sat_aligned_velocities = spa.sat_align_moi(sat)
    
        for j, name in enumerate(sat.hal_name):
            approaching = sat_aligned_velocities[j][0][:,1] < 0
            receding = sat_aligned_velocities[j][0][:,1] > 0
            ax[j, i].plot(sat_aligned_coords[j][0][:,0][approaching], sat_aligned_coords[j][0][:,2][approaching], 'b.', alpha=0.5, label=name+' approaching')
            ax[j, i].plot(sat_aligned_coords[j][0][:,0][receding], sat_aligned_coords[j][0][:,2][receding], 'r.', alpha=0.5, label=name+' receding')
            ax[j, i].legend()
    
    ax[2, 1].set_xlabel('x\' [kpc]')
    ax[1, 0].set_ylabel('z\' [kpc]')
    plt.show()

def plot_ang_momentum_min_axis_projected(sat, mask_key):
    '''
    angle_width = ba.loop_hal(sat, mask_key, ang.open_angle)
    
    for i, hal_name in enumerate(sat.hal_name):
        plt.figure()
        for j, redshift in enumerate(sat.redshift):
            plt.plot(angle_width[hal_name][j], sat_aligned_L[hal_name][j][:,2], '.', label=sat.hal_label[mask_key][i]+' z='+str(redshift))
        plt.yscale('symlog')
        plt.legend()
        plt.show()
    '''
    sat_MOI_projected_j = sio.loop_hal(sat, mask_key, kin.project_orbital_ang_momentum)
    plt.figure()
    for i, hal_name in enumerate(sat.hal_name):
        total_j = np.array([np.sum(sat_MOI_projected_j[hal_name][k], axis=0) for k in range(len(sat.redshift))])
        normed_j_z = np.array([abs(total_j[k][2])/np.linalg.norm(total_j[k]) for k in range(len(sat.redshift))])
        plt.plot(sat.redshift, normed_j_z, label=sat.hal_label[mask_key][i])
    plt.ylim([0,1])
    plt.xlabel('Redshift [z]', fontsize=12)
    plt.ylabel('Normalized total MOI-projected $L_z$ [$km^2/s$]', fontsize=12)
    plt.legend()
    plt.show()
    
    sat_randmin_projected_j = sio.loop_hal(sat, mask_key, kin.project_orb_ang_mom_randminaxes, **{'fraction':1.0, 'angle_range':sat.a_bins, 'n_iter':sat.n_iter})
    plt.figure()
    for i, hal_name in enumerate(sat.hal_name):
        total_j = np.array([np.sum(sat_randmin_projected_j[hal_name][k], axis=0) for k in range(len(sat.redshift))])
        normed_j_z = np.array([abs(total_j[k][2])/np.linalg.norm(total_j[k]) for k in range(len(sat.redshift))])
        plt.plot(sat.redshift, normed_j_z, label=sat.hal_label[mask_key][i])
    plt.ylim([0,1])
    plt.xlabel('Redshift [z]', fontsize=12)
    plt.ylabel('Normalized total minaxis-projected $L_z$ [$km^2/s$]', fontsize=12)
    plt.legend()
    plt.show()

def ang_momentum_spatial_MOI_dot(sat, mask_key, norm=True):
    am_spa_MOI_dot = sio.loop_hal(sat, mask_key, kin.ang_momentum_MOI_axes_dotprod, **{'norm':norm})
    plt.figure()
    for i, hal_name in enumerate(sat.hal_name):
        plt.plot(sat.redshift, am_spa_MOI_dot[hal_name], label=sat.hal_label[mask_key][i])
    plt.ylim([0,1])
    plt.xlabel('Redshift [z]', fontsize=12)
    plt.ylabel('MOI $\| {\hat{j}_{z}} \cdot {\hat{r}_{z}} \|$')
    plt.legend()
    plt.show()

def draw_sphere(xCenter, yCenter, zCenter, r):
    #draw sphere
    u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    # shift and scale sphere
    x = r*x + xCenter
    y = r*y + yCenter
    z = r*z + zCenter

    return (x,y,z)

def plot_3d_position(sat):
    #for i, host in enumerate(sat.halo_catalog):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sat_coords = sat.halo_catalog[0][0].prop('host.distance')[sat.catalog_mask[0][0]]
    x = np.array([vec[0] for vec in sat_coords])
    y = np.array([vec[1] for vec in sat_coords])
    z = np.array([vec[2] for vec in sat_coords])
    sph_radius = 3*np.log10(sat.halo_catalog[0][0].prop('star.mass')[sat.catalog_mask[0][0]])

    # draw a sphere for each data point
    for (xi,yi,zi,ri) in zip(x,y,z,sph_radius):
        (xs,ys,zs) = draw_sphere(xi,yi,zi,ri)
        ax.plot_surface(xs, ys, zs, rstride=1, cstride=1, cmap='viridis')

    host_x = sat.halo_catalog[0][0].prop('host.distance')[0][0]
    host_y = sat.halo_catalog[0][0].prop('host.distance')[0][1]
    host_z = sat.halo_catalog[0][0].prop('host.distance')[0][2]
    host_sm = 3*np.log10(sat.halo_catalog[0][0].prop('star.mass')[0])

    (xs,ys,zs) = draw_sphere(host_x,host_y,host_z,host_sm)
    ax.plot_surface(xs, ys, zs, rstride=1, cstride=1, cmap='magma')

    plt.show()

def plot_sat_position_mov(sat):
    '''
    Make a movie of the 3D satellite positions over the len(sat.reshift) snapshots.
    '''
    writer = animation.FFMpegWriter(fps=5, bitrate=500)
    
    sat_coords = [sat.halo_catalog[0][i].prop('host.distance')[sat.catalog_mask[0][i]] for i in range(len(sat.redshift))]
    host_index = [sat.halo_catalog[0][i].prop('host.index')[0] for i in range(len(sat.redshift))]
    host_position = [sat.halo_catalog[0][i].prop('host.distance')[host_index[i]] for i in range(len(sat.redshift))]
    #host_stellarmass = sat.halo_catalog[0].prop('star.mass')[host_index]

    sat_coords = sat_coords[::-1]
    host_position = host_position[::-1]
    redshift = sat.redshift[::-1]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def animate(j):
        x = sat_coords[j][:,0]
        y = sat_coords[j][:,1]
        z = sat_coords[j][:,2]

        hx = host_position[j][0]
        hy = host_position[j][1]
        hz = host_position[j][2]
        ax.clear()
        ax.set_title('$z = {}$'.format(redshift[j]))
        ax.scatter(x, y, z, c='k')
        ax.scatter(hx, hy, hz, c='r')

        ax.set_xlim3d([-350, 350])
        ax.set_xlabel('X')
        ax.set_ylim3d([-350, 350])
        ax.set_ylabel('Y')
        ax.set_zlim3d([-350, 350])
        ax.set_zlabel('Z')

    ani = animation.FuncAnimation(fig, animate, frames=range(len(sat.redshift)), interval=1000)
    #ani.save('test.mp4', writer=writer)

    return ani

def plot_sat_position_mov3d(sat, id=0):
    '''
    Make a movie of the 3D satellite positions over the len(sat.reshift) snapshots.
    '''
    writer = animation.FFMpegWriter(fps=5, bitrate=5000)
    
    sat_coords = [sat.halo_catalog[id][i].prop('host.distance')[sat.catalog_mask[id][i]] for i in range(len(sat.redshift))]
    sat_sm = [2*np.log10(sat.halo_catalog[id][i].prop('star.mass')[sat.catalog_mask[id][i]]) for i in range(len(sat.redshift))]

    host_index = [sat.halo_catalog[id][i].prop('host.index')[0] for i in range(len(sat.redshift))]
    host_position = [sat.halo_catalog[id][i].prop('host.distance')[host_index[i]] for i in range(len(sat.redshift))]
    host_sm = [2*np.log10(sat.halo_catalog[id][i].prop('star.mass')[host_index[i]]) for i in range(len(sat.redshift))]

    sat_coords = sat_coords[::-1]
    host_position = host_position[::-1]
    redshift = sat.redshift[::-1]

    #plt.style.use('dark_background')
    fig = plt.figure(figsize=(8,8), dpi=250)
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(b=False)
    ax.xaxis.pane.set_edgecolor('black')
    ax.yaxis.pane.set_edgecolor('black')
    ax.zaxis.pane.set_edgecolor('black')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.set_facecolor('black')
    ax.set_axis_off()

    def animate(j):
        x = sat_coords[j][:,0]
        y = sat_coords[j][:,1]
        z = sat_coords[j][:,2]
        r = sat_sm[j]

        hx = host_position[j][0]
        hy = host_position[j][1]
        hz = host_position[j][2]
        hr = host_sm[j]

        ax.clear()
        ax.grid(b=False)
        ax.set_title('{} $z = {}$'.format(sat.host_name[id], redshift[j]), color='white')

        for (xi,yi,zi,ri) in zip(x,y,z,r):
            (xs,ys,zs) = draw_sphere(xi,yi,zi,ri)
            ax.plot_surface(xs, ys, zs, rstride=1, cstride=1, cmap='YlGn')

        (x_s,y_s,z_s) = draw_sphere(hx,hy,hz,hr)
        ax.plot_surface(x_s, y_s, z_s, rstride=1, cstride=1, cmap='BuPu')

        ax.set_axis_off()
        ax.set_xlim3d([-350, 350])
        ax.set_xlabel('X')
        ax.set_ylim3d([-350, 350])
        ax.set_ylabel('Y')
        ax.set_zlim3d([-350, 350])
        ax.set_zlabel('Z')

    ani = animation.FuncAnimation(fig, animate, frames=range(len(sat.redshift)))
    ani.save('../movies/m12i.mp4', writer=writer)

    return ani
'''
def plot_ang_momentum(sat, bin_list=None):
    #Plot satellite angular momentum aligned with the MOI tensor axes.

    ang_moment_list, ang_moment_list_vec = kin.align_angular_momentum(sat)
    
    if bin_list == None:
        #bin_list = np.linspace(min(ang_moment_list[0][0]), max(ang_moment_list[0][0]), 15)
        bin_list = [-1e8, -1e7, -1e6, -1e5, -1e4, -1e3, -1e2, -1e1, -1e0, 0, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8]

    pg.plot_prop(
        bin_list,
        prop_list1=ang_moment_list,
        prop_list2=None,
        halo_names=sat.host_name,
        dmo_halo_names=None,
        label_names1=sat.host_label_name,
        label_names2=None,
        plot_title='Angular momentum at z=0',
        xlabel='$L_{z}$', 
        ylabel='$N$',
        color_list=CB_color_cycle,
        xscale='symlog',
        location=1)

    mark_list = ['x', '^', '.']

    plt.figure()
    for i, vec_list in enumerate(ang_moment_list_vec):
        plt.plot(vec_list[0][0], vec_list[0][1], marker=mark_list[i], markersize=10, linestyle='', label=(sat.host_name)[i])
    plt.xlabel('$L_x$')
    plt.ylabel('$L_y$')
    plt.xscale('symlog')
    plt.yscale('symlog')
    plt.legend()
    plt.show()

    plt.figure()
    for i, vec_list in enumerate(ang_moment_list_vec):
        plt.plot(vec_list[0][0], vec_list[0][2], marker=mark_list[i], markersize=10, linestyle='', label=(sat.host_name)[i])
    plt.xlabel('$L_x$')
    plt.ylabel('$L_z$')
    plt.xscale('symlog')
    plt.yscale('symlog')
    plt.legend()
    plt.show()

    plt.figure()
    for i, vec_list in enumerate(ang_moment_list_vec):
        plt.plot(vec_list[0][1], vec_list[0][2], marker=mark_list[i], markersize=10, linestyle='', label=(sat.host_name)[i])
    plt.xlabel('$L_y$')
    plt.ylabel('$L_z$')
    plt.xscale('symlog')
    plt.yscale('symlog')
    plt.legend()
    plt.show()
'''