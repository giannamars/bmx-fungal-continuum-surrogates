import numpy as np
from scipy import optimize
from dolfin import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
import pandas as pd
from matplotlib.pyplot import cm
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import shapely
from shapely import Polygon, Point
from matplotlib.collections import LineCollection
import re
from scipy import stats
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import CubicSpline
from numpy import linalg
import yt
from numpy import inf
import sys
import json
from scipy.spatial import QhullError
import csv
import warnings
import glob

def make_rings(data, ring_width):
    """
    From data for points in network and width of rings to divide network by, returns vertices of convex hull and concentric rings
    and radius for rings approximated as circles.
    """
    hull = ConvexHull(data)
    vertices0_index = hull.vertices
    vertices0_index = np.append(vertices0_index, vertices0_index[0])
    vertices0 = data[vertices0_index]

    rings = [vertices0]

    vertices_current = vertices0
    while Polygon(vertices_current).area > 0:
        # make rings
        buffer = shapely.buffer(Polygon(vertices_current), - ring_width).exterior.coords
        vertices_smaller = np.zeros(np.shape(vertices0))
        i = 0
        for x,y in buffer:
            vertices_smaller[i][0] = x
            vertices_smaller[i][1] = y
            i += 1
        vertices_current = vertices_smaller[~np.all(vertices_smaller == 0, axis=1)]
        rings.append(vertices_current)
    rings = rings[:-1] # remove last zero area polygon
    rings = rings[::-1]

    r = []
    for ring in rings:
        area = Polygon(ring).area
        r.append(np.sqrt(area / np.pi)) 

    return rings, r

def sigmoid_r(r, K, L, r0):
    return K / (1 + np.exp(-L * (r0 - r)))

def sigmoid_t(t, K, L, tn):
    return K / (1 + np.exp(L * (tn - t)))

def initial_pulse(r, A, s, r0):
    return A * np.exp(-s * np.abs(r - r0) ** 2)

def tolerant_mean(arrs):
    lens = [len(i) for i in arrs]
    arr = np.ma.empty((np.max(lens),len(arrs)))
    arr.mask = True
    for idx, l in enumerate(arrs):
        arr[:len(l),idx] = l
    return arr.mean(axis = -1), arr.std(axis=-1)
    
def round_to_multiple(number, multiple):
    return multiple * np.round(number / multiple)

def density(data, rings):
    """
    From data for points in a network and concentric rings dividing up network,
    returns density of points in each ring.
    """
    density = [0]

    for point in data:
        # inner most density is separate since it is not a ring
        if shapely.within(Point(point), Polygon(rings[0])):
            density[0] += 1
    density[0] /= Polygon(rings[0]).area

    j = 1        
    for ring in rings[1:]:
        # calculate density of each ring
        density.append(0)
        for point in data:
            if Polygon(ring).contains(Point(point)) and ~shapely.within(Point(point), Polygon(rings[j - 1])):
                density[j] += 1
        density[j] /= (Polygon(ring).area - Polygon(rings[j - 1]).area)
        j += 1

    return np.array(density)

def setup_fs_data(s_data, f_data, outf):
    s_data['split'] = 1 # 1 = yes split
    f_data['fuse'] = 1
    s_data['step_split'] = ((s_data['time_split'] / 0.25) / 400).round().astype(int) * 400
    f_data['step_fuse'] = ((f_data['time_fuse'] / 0.25) / 400).round().astype(int) * 400
    s_data['hour_split'] = (np.floor(s_data['time_split']) / (60 ** 2)).round().astype(int)
    f_data['hour_fuse'] = (np.floor(f_data['time_fuse']) / (60 ** 2)).round().astype(int)
    s_data['output_split'] = s_data['hour_split'] * 14400
    f_data['output_fuse'] = f_data['hour_fuse'] * 14400
    s_data = s_data[s_data['output_split'] < outf]
    f_data = f_data[f_data['output_fuse'] < outf]

    return s_data, f_data

def readin_data(run_name):
    path =  '/Users/laurenyan/bmxfungi_surrogate/256_outputs/' + run_name + '/'
    
    out_count = len(glob.glob1(path,"plt*")) - 1
    if out_count = 35:
        outf = 500000
    else:
        outf = 14400 * out_count
    out0 = 14400
    dout = 14400 # get data for each hour
    outs = np.arange(out0, outf, dout)
    data_names = [str(x) for x in outs]
    time = len(data_names) # hours
    
    # print('data from: ' + path)
    datas = {}

    for i in range(len(data_names)):
        ds = yt.load(path + 'plt' + data_names[i])
        doi = ds.all_data()
        dfi = doi.to_dataframe([('all', 'particle_unique_id'), ('all', 'particle_position_x'), ('all', 'particle_position_y'), ('all', 'particle_position_z'), ('all', 'particle_position'),  ('all', 'particle_cell_volume')])
        dfi = dfi.rename(columns={'particle_unique_id': 'unique_id', 'particle_position_x': 'x', 'particle_position_y': 'y', 'particle_position_z': 'z', 'particle_position': 'position', 'particle_cell_volume': 'cell_volume'})
        datas[int(data_names[i])] = dfi

    # print(datas.keys())
    try:
        split_count = pd.read_csv(path + 'split_count.txt', names = ['count', 'time_split'])
        splits = pd.read_csv(path + 'splits.txt', names = ['unique_id', 'time_split'])
        fuse_count = pd.read_csv(path + 'fuse_count.txt', names = ['count', 'time_fuse'])
        fuses = pd.read_csv(path + 'fuses.txt', names = ['unique_id', 'time_fuse'])
    except FileNotFoundError:
        fuses = pd.DataFrame(columns=['unique_id', 'time_fuse'])
        fuse_count = pd.DataFrame(columns=['count', 'time_fuse'])
    splits, fuses = setup_fs_data(splits, fuses, outf)

    # data for all outputs where event occurs
    out_fs = pd.concat([fuses['output_fuse'], splits['output_split']]).unique()
    data_fs_names = [str(x) for x in out_fs if x != 0]
    datas_fs = {}

    for i in range(len(data_fs_names)):
        ds = yt.load(path + 'plt' + data_fs_names[i])
        doi = ds.all_data()
        dfi = doi.to_dataframe([('all', 'particle_unique_id'), ('all', 'particle_position_x'), ('all', 'particle_position_y'), ('all', 'particle_position_z'), ('all', 'particle_position'),  ('all', 'particle_cell_volume')])
        dfi = dfi.rename(columns={'particle_unique_id': 'unique_id', 'particle_position_x': 'x', 'particle_position_y': 'y', 'particle_position_z': 'z', 'particle_position': 'position', 'particle_cell_volume': 'cell_volume'})
        datas_fs[int(data_fs_names[i])] = dfi
    # print(datas_fs.keys())

    return datas, datas_fs, split_count, splits, fuse_count, fuses, time

def analyze_network(run_name, data_all, data_fs, fuses, splits, ring_width):
    """
    From list of data for points in network at different times and width of rings to divide network, 
    plots network at final time, tip and filament density in time and space, position of wave front, and calculates mean speed of wave front.
    """
    outf = list(data_all.keys())[-1]
    dataf = data_all[outf]
    times = [int(t) * (0.25 / 3600) for t in data_all.keys()]
    rings, rs = make_rings(dataf[['x','y']].to_numpy(), ring_width) # make rings from convex hull of final network shape
    n_rings = len(rings)
    density_tip_sum = {}
    density_in_sum = {}
    tip_r = []
    in_r = []
    ring_tip_t = []
    ring_in_t = []
    tns = []
    in_sat = []
    hr_f = int(times[-1]) # last output hour
    data_s = {}
    data_f = {}
    density_s_sum = {}
    density_f_sum = {}
    ring_s_t = []
    ring_f_t = []
    tip_pull_x = np.zeros(len(times))
    tip_pull_v = np.zeros(len(times))
    tip_all_x = np.zeros(len(times))
    tip_all_v = np.zeros(len(times))
    vdrift = np.zeros(len(times))
    xdrift = np.zeros(len(times))
    vdrift_pull = np.zeros(len(times))
    xdrift_pull = np.zeros(len(times))
    l_filament = 35

    for r in range(n_rings):
        ring_tip_t.append({}) # one dictionary for each ring, will append density at each time
        ring_in_t.append({})
        ring_s_t.append({})
        ring_f_t.append({})

    dataf = data_all[outf][['x', 'y']].to_numpy()
    hullf = ConvexHull(dataf)
    tip_pull_i = hullf.vertices
    tip_pull_i = np.append(tip_pull_i, tip_pull_i[0])
    tip_pullf = dataf[tip_pull_i]
    origin = shapely.centroid(Polygon(tip_pullf))
    x0 = origin.x
    y0 = origin.y

    i = 0
    for out in data_all.keys():
        time = int(out * (0.25 / 3600))

        # sorting out split and fusion data
        if out in data_fs.keys():
            data_fs_n = data_fs[out]

            if out in fuses['output_fuse'].values:
                fuses_out = fuses[fuses['output_fuse'] == out]
                data_fs_n = data_fs_n.merge(fuses_out, on = "unique_id", how = "left")
            else:
                data_fs_n['fuse'] = np.nan

            if out in splits['output_split'].values:
                splits_out = splits[splits['output_split'] == out]
                data_fs_n = data_fs_n.merge(splits_out, on = "unique_id", how = "left")
            else:
                data_fs_n['split'] = np.nan

            data_s[out] = data_fs_n[data_fs_n['split'] == 1] # particle data of all particles that split
            data_f[out] = data_fs_n[data_fs_n['fuse'] == 1] # particle data of all particles that fuse

            # density in each ring
            density_s = density(data_s[out][['x','y']].to_numpy(), rings)
            density_f = density(data_f[out][['x', 'y']].to_numpy(), rings)
        else:
            density_s = [0] * n_rings
            density_f = [0] * n_rings

        q = 0
        for ring in rings:
            ring_s_t[q][time] = density_s[q]
            ring_f_t[q][time] = density_f[q]
            q += 1

        data = data_all[out]

        data = data[['x', 'y', 'position']]
        data_tip = data[data['position'] == 0]
        tip_all1 = data_tip[['x', 'y']].to_numpy()
        data_in = data[data['position'] != 0]
        data = data[['x', 'y']].to_numpy()

        # getting puller tips
        try:
            hull = ConvexHull(data)
            tip_pull_i = hull.vertices
            tip_pull_i = np.append(tip_pull_i, tip_pull_i[0])
            tip_pull1 = data[tip_pull_i]
        except QhullError as q:
            tip_pull1 = data

        # density in each ring
        density_tip = density(data_tip[['x','y']].to_numpy(), rings)
        density_in = density(data_in[['x', 'y']].to_numpy(), rings) * l_filament
        tip_r.append(density_tip)
        in_r.append(density_in)

        k = 0
        for ring in rings:
            ring_tip_t[k][time] = density_tip[k]
            ring_in_t[k][time] = density_in[k]
            k += 1

        # displacement of puller tips from previous time step
        if i > 0: # can't calculate velocity in first time step
            j = 0
            for pull in tip_pull0:
                xtip = pull[0]
                ytip = pull[1]

                # velocity of puller tips
                distances = np.linalg.norm(tip_pull1 - pull, axis=1)
                min_index = np.argmin(distances)
                tip_pull_dx[j] = distances[min_index]

                # drift velocity of puller tips
                vtip = np.array([tip_pull1[min_index][0] - xtip, tip_pull1[min_index][1] - ytip])
                rtip = np.array([xtip-x0, ytip-y0])
                with warnings.catch_warnings():
                    warnings.filterwarnings("error", category=RuntimeWarning)
                    try:
                        theta = np.dot(rtip, vtip) / (np.linalg.norm(rtip) * np.linalg.norm(vtip))
                        vdrift_pull[i] += np.linalg.norm(vtip) * np.sin(theta) / 3600 # cm/s
                    except RuntimeWarning as r:                    
                        vdrift_pull[i] += 0 # cm/s
                        print(r)
                        print('tip velocity = ' + str(vtip))
                j += 1
            tip_pull_x[i] = tip_pull_x[i - 1] + np.mean(tip_pull_dx) # total displacement 
            tip_pull_v[i] = np.mean(tip_pull_dx) / 3600 # mean velocity over all tips cm/s
            vdrift_pull[i] /= len(tip_pull0)
            xdrift_pull[i] = xdrift_pull[i - 1] + vdrift_pull[i] * 3600 # cm

            j = 0
            for tip in tip_all0:
                xtip = tip[0]
                ytip = tip[1]

                # velocity of all tips
                distances = np.linalg.norm(tip_all1 - tip, axis=1)
                min_index = np.argmin(distances)
                tip_all_dx[j] = distances[min_index]

                # drift velocity of all tips
                vtip = np.array([tip_all1[min_index][0] - xtip, tip_all1[min_index][1] - ytip])
                rtip = np.array([xtip-x0, ytip-y0])
                with warnings.catch_warnings():
                    warnings.filterwarnings("error", category=RuntimeWarning)
                    try:
                        theta = np.dot(rtip, vtip) / (np.linalg.norm(rtip) * np.linalg.norm(vtip))
                        vdrift[i] += np.linalg.norm(vtip) * np.sin(theta) / 3600 # cm/s
                    except RuntimeWarning as r:                    
                        vdrift[i] += 0 # cm/s
                        print(r)
                        print('tip velocity = ' + str(vtip))
                j += 1
            tip_all_x[i] = tip_all_x[i - 1] + np.mean(tip_all_dx) # total displacement 
            tip_all_v[i] = np.mean(tip_all_dx) / 3600 # mean velocity over all tips cm/s
            vdrift[i] /= len(tip_all0)
            xdrift[i] = xdrift[i - 1] + vdrift[i] * 3600

        i += 1   
        tip_pull0 = tip_pull1 # saving tips to calculate velocity at next time
        tip_all0 = tip_all1
        tip_pull_dx = np.zeros(len(tip_pull0))
        tip_all_dx = np.zeros(len(tip_all0))

    pulse_max = max(tip_r[:][:3], key = max)
    A= np.max(pulse_max) / 100
    s= np.max(pulse_max)
    r0 = rs[np.argmax(pulse_max)]
    
    fig4, ax4 = plt.subplots(figsize = (8,8))
    for ring in rings:
        ax4.plot(ring[:, 0], ring[:, 1], 'r--', lw = 1)
    ax4.plot(rings[np.argmax(pulse_max)][:, 0], rings[np.argmax(pulse_max)][:, 1], 'b--', lw = 1)
    ax4.scatter(data_tip['x'].to_numpy(), data_tip['y'].to_numpy(), marker = 's')
    ax4.scatter(data_in['x'].to_numpy(), data_in['y'].to_numpy(), marker = 's')
    ax4.set_title('Network at ' + str(hr_f) + 'hrs')
    ax4.set_xlabel('x(cm)')
    ax4.set_ylabel('y(cm)')
    fig4.savefig(run_name + '_network.png')
    
    # fitting sigmoid curve to filament density over time of each ring
    ring_in_t_fit = []
    omit_ring = [] # omit these rings for all averages over rings and with shifted time
    p = 0
    for ring in ring_in_t:
        try:
            ring_t = np.array(list(ring.values()))
            p0 = [np.max(ring_t), np.min(times), np.median(times)] # initial guess
            popt, pcov = curve_fit(sigmoid_t, times, ring_t, p0) # best fit parameters and covariance
            if popt[1] < 0:
                omit_ring.append(p)
                tns.append(0)
            else:
                if popt[2] > times[-1]:
                    tns.append(0)
                else:
                    tns.append(np.floor(popt[2]))
                in_sat.append(popt[0])
                ring_in_t_fit.append(sigmoid_t(times, popt[0], popt[1], popt[2]))

        except RuntimeError as e:
            omit_ring.append(p)
            tns.append(0)

    # t0 = 0 # int(times[-1] / 3)
    r0 = 0 # np.nonzero(tip_r[t0])[0][0]
    # time when network first enters the innermost ring
    try:
        t0 = np.nonzero(np.array(list(ring_tip_t[0].values())))[0][0]
    except IndexError as i:
        t0 = np.nonzero(np.array(list(ring_tip_t[1].values())))[0][0]

    shifted_t = np.arange(-int(times[-1] - tns[r0]), int(times[-1] - tns[r0]))

    # ax4.plot(rings[r0][:, 0], rings[r0][:, 1], 'g--', lw = 1)

    # make dictionary with keys for shifted time
    for t in shifted_t:
        density_s_sum[t] = 0
        density_f_sum[t] = 0

    # calculating mean density of each ring at shifted time
    m = r0
    for ring_s, ring_f in zip(ring_s_t[r0:], ring_f_t[r0:]):
        if m not in omit_ring:
            tn = tns[m] # time to shift by
            for t in density_s_sum.keys():
                if t + tn in ring_s.keys():
                    density_s_sum[t] += ring_s[t + tn]
                else:
                    density_s_sum[t] += 0 # no data for this time, so network has not arrived there yet

            for t in density_f_sum.keys():
                if t + tn in ring_f.keys():
                    density_f_sum[t] += ring_f[t + tn]
                else:
                    density_f_sum[t] += 0
        m += 1

    density_s_mean = [x / (n_rings - r0) for x in list(density_s_sum.values())]
    density_f_mean = [x / (n_rings - r0) for x in list(density_f_sum.values())]

    # make dictionary with keys for shifted time
    for t in shifted_t:
        density_tip_sum[t] = 0
        density_in_sum[t] = 0

    # calculating mean density of each ring at shifted time
    m = r0
    for ring_tip, ring_in in zip(ring_tip_t[r0:], ring_in_t[r0:]):
        if m not in omit_ring:
            tn = tns[m] # time to shift by
            for t in density_tip_sum.keys():
                if t + tn in ring_tip.keys():
                    density_tip_sum[t] += ring_tip[t + tn]
                else:
                    density_tip_sum[t] += 0 # no data for this time, so network has not arrived there yet

            for t in density_in_sum.keys():
                if t + tn in ring_in.keys():
                    density_in_sum[t] += ring_in[t + tn]
                elif ((t + tn) > np.array(list(ring_in.keys()))).all(): # after the network passes through the ring the density is constant
                    density_in_sum[t] += ring_in[np.max(list(ring_in.keys()))]
                else:
                    density_in_sum[t] += 0
        m += 1

    density_tip_mean = [x / (n_rings - r0) for x in list(density_tip_sum.values())]
    density_in_mean = [x / (n_rings - r0) for x in list(density_in_sum.values())]

    res = stats.linregress(density_tip_mean, density_s_mean)
    alpha = res.slope 
    intercept_alpha = res.intercept
    n = np.array(density_tip_mean)
    alphan = np.array(density_tip_mean) * alpha
    b = np.array(density_s_mean)

    res = stats.linregress(np.array(density_tip_mean) * np.array(density_in_mean), density_f_mean)
    beta = res.slope * 100000
    intercept_beta = res.intercept
    nrho = np.array(density_tip_mean) * np.array(density_in_mean)
    betanrho = np.array(density_tip_mean) * np.array(density_in_mean) * beta
    a = np.array(density_f_mean)

    res = stats.linregress(times, xdrift) # cm/hr
    vd = res.slope * 10 # mm/hr
    sg = np.mean(tip_pull_v[1:]) * 36000 # mm/hr

    return tip_r, in_r, ring_tip_t, ring_in_t, xdrift, xdrift_pull, rs, r0, t0, shifted_t, alpha, intercept_alpha, n, alphan, b, beta, intercept_beta, nrho, betanrho, a, vd, sg

def density_c(tip_r, in_r, ring_tip_t, ring_in_t, rs, time, r0, t0):
    times = np.arange(t0, time)
    n_sat_t = []
    rho_sat_t = []
    wavefront_t = []
    r_t = []
    n_sat_r = []
    rho_sat_r = []

    # fig1, axs = plt.subplots(1, 2, figsize = (12, 6))
    # for t in times:
    #     if (tip_r[t][r0:] > 0).any():
    #         t0 = t
    #         times = np.arange(t0, time)
    #         break
    k = 0
    for r in rs:
        tip_t_r = np.array(list(ring_tip_t[k].values()))
        in_t_r = np.array(in_r[np.argmax(tip_t_r)])
        n_sat_r.append(np.max(tip_t_r))
        rho_sat_r.append(np.median(in_t_r[in_t_r != 0]))
        k += 1

    for t in times:
        in_r_t = np.array(in_r[t][r0:])
        tip_r_t = np.array(tip_r[t][r0:])
        n_sat_t.append(np.max(tip_r_t))
        rho_sat_t.append(np.median(in_r_t[in_r_t != 0]))
        r_t.append(np.argmax(tip_r_t) + r0)
        # try:
        #     p0 = [np.max(in_r_t) / 100000, 1, np.median(rs[r0:]) * 100] # initial guess 
        #     popt, pcov = curve_fit(sigmoid_r, np.array(rs[r0:]) * 100, np.array(in_r_t) / 100000, p0) # best fit parameters and covariance
        #     # axs[1].plot(np.array(rs[r0:]) * 100, sigmoid_r(np.array(rs[r0:]) * 100, np.max(in_r)/ 100000, np.min(rs[r0:]) * 100, np.median(rs[r0:]) * 100), label = 'initial guess')
        #     if (popt[0] > np.max(in_r_t) / 100000) or (popt[0] < 0) or (popt[1] < 0):
        #         rho_max_t.append(np.max(in_r_t))
        #         # print(density_sat)
        #         # print(popt[2])
        #         # axs[0].plot(np.array(rs[r0:]) * 100, in_r_t / 100000) # across all rings from r
        #         # axs[0].hlines(rho_max, rs[r0] * 100, rs[-1] * 100, linestyle = '--', color = 'r')
        #     else:
        #         rho_max_t.append(popt[0] * 100000)
        #         # print(density_sat)
        #         # print(popt[2])
        #         # axs[1].plot(np.array(rs[r0:]) * 100, sigmoid_r(np.array(rs[r0:]) * 100, popt[0], popt[1], popt[2]), 'r', label = 'best fit')
        #         # axs[1].vlines(popt[2], 0, np.max(in_r_t) / 100000, linestyle = '--')
        #         # axs[1].hlines(popt[0], rs[r0] * 100, rs[-1] * 100, linestyle = '--', color = 'g')
        #         # axs[0].plot(np.array(rs[r0:]) * 100, in_r_t / 100000) # across all rings from r
        #         # axs[0].vlines(popt[2], 0, np.max(in_r_t) / 100000, linestyle = '--')
        #         # axs[0].hlines(popt[0], rs[r0] * 100, rs[-1] * 100, linestyle = '--', color = 'g')
        # except RuntimeError as e:
        #     # axs[0].plot(np.array(rs[r0:]) * 100, in_r_t / 100000)
        #     rho_max_t.append(np.max(in_r_t))
        #     # print(density_sat)
        #     # axs[0].hlines(rho_max, rs[r0] * 100, rs[-1] * 100, linestyle = '--', color = 'b')
        #     # axs[1].hlines(rho_max, rs[r0] * 100, rs[-1] * 100, linestyle = '--', color = 'b')
        #     print('hour ' + str(t))
        #     print(e)
        # try:
        #     p0 = [np.max(in_r_t)/ 100000, 1, np.median(rs[r0:]) * 100] # initial guess 
        #     popt, pcov = curve_fit(sigmoid_r, np.array(rs[r0:]) * 100, np.array(in_r_t) / 100000, p0) # best fit parameters and covariance
        #     if popt[0] > np.max(in_r_t) / 100000:
        #         rho_max = np.max(in_r_t / 100000)
        #     else:
        #         rho_max = popt[0]
        # except RuntimeError as e:
        #     rho_max = np.max(in_r_t) / 100000
        #     print('hour ' + str(t))
        #     print(e)

        # rho_max_t.append(rho_max)

        try:
            wavefront = rs[r0:][np.argmax(tip_r_t)]
        except IndexError:
            wavefront = rs[r0:][-1]

        wavefront_t.append(wavefront)

    return np.array(n_sat_t), np.array(rho_sat_t), np.array(r_t), np.array(wavefront_t), np.array(n_sat_r), np.array(rho_sat_r)

def ABM(sim_name, n_runs, ring_width):
    '''
    sim_name = name of runs stored
    (ex. bmxdefault)
    n_runs = number of reruns
    Returns average and standard deviation over all runs in time. 
    '''
    datas = []
    datas_fs = []
    splits = []
    fuses = []
    split_counts = []
    fuse_counts = []
    outputs = []
    n_maxs_t = []
    rho_maxs_t = []
    n_maxs_r = []
    rho_maxs_r = []
    wavefronts = []
    xdrifts = []
    xdrift_pulls = []
    wavefronts_slope = []
    xdrifts_slope = []
    xdrift_pulls_slope = []
    wavefronts_intercept = []
    xdrifts_intercept = []
    xdrift_pulls_intercept = []
    r_max = []
    t0min = 34
    tfmax = 0
    alpha0 = 0
    beta0 = 0
    v_sval0 = 0
    s_scal0 = 0
    A0 = 0
    s0 = 0
    r00 = 0
    
    
    fig5, ax5 = plt.subplots(2, 4, figsize = (16, 8))
    for i in range(n_runs):
        run = sim_name + '_' + str(i)
        datai, data_fsi, split_counti, splitsi, fuse_counti, fusesi, tfi = readin_data(run)
        # tip_r, in_r, ring_tip_t, ring_in_t, xdrift, xdrift_pull, rs, r0, t0, shifted_t, alpha, intercept_alpha, n, alphan, b, beta, intercept_beta, nrho, betanrho, a, vd, sg
        outputi = analyze_network(run, datai, data_fsi, fusesi, splitsi, ring_width)
        rs = outputi[6]
        r0 = outputi[7]
        t0 = outputi[8]
        print(run + ':')
        print('starting at time = ' + str(t0) + ' and ring = ' + str(r0))
        alpha0 += outputi[10]
        beta0 += outputi[15]
        v_sval0 += outputi[20]
        s_scal0 += outputi[21]

        pulse_max = max(outputi[0][:][:3], key = max)
        A0 += np.max(pulse_max) / 100
        s0 += np.max(pulse_max)
        r00 += rs[np.argmax(pulse_max)]

        ax5[1,3].plot(rs, initial_pulse(rs, np.max(pulse_max) / 100, np.max(pulse_max), rs[np.argmax(pulse_max)]), 'b--', alpha = 0.3)

        nm_t, rm_t, r_t, wf, nm_r, rm_r = density_c(outputi[0], outputi[1], outputi[2], outputi[3], rs, tfi, r0, t0)
        nm_t /= 100
        rm_t *= 1e-5
        nm_r /= 100
        rm_r *= 1e-5
        wf -= 0.0025
        wf *= 10
        wf -= wf[0]
        xd = np.array(outputi[4]) * 10
        xd_pull = np.array(outputi[5]) * 10
        n_maxs_t.append(nm_t)
        rho_maxs_t.append(rm_t)
        n_maxs_r.append(nm_r)
        rho_maxs_r.append(rm_r)
        wavefronts.append(wf)
        xdrifts.append(xd)
        xdrift_pulls.append(xd_pull)

        times = np.arange(t0, tfi) - t0
        res = stats.linregress(times, wf)
        wavefronts_slope.append(res.slope)
        wavefronts_intercept.append(res.intercept)
        res = stats.linregress(times, xd_pull[t0:])
        xdrift_pulls_slope.append(res.slope)
        xdrift_pulls_intercept.append(res.intercept)
        res = stats.linregress(times, xd[t0:])
        xdrifts_slope.append(res.slope)
        xdrifts_intercept.append(res.intercept)

        if len(rs) > len(r_max):
            r_max = rs
        if t0 < t0min:
            t0min = t0
        if tfi > tfmax:
            tfmax = tfi

        i += 1

    alpha0 /= i
    beta0 /= i
    v_sval0 /= i
    s_scal0 /= i
    A0 /= i
    s0 /= i
    r00 /= i

    r_max -= r_max[0]
    r_max_round = round_to_multiple(r_max * 10, 0.005 * 10)


    upar = [A0, s0, r00]
    ax5[1,3].plot(rs, initial_pulse(rs, A0, s0, r00), 'y')

    nmt_mean, nmt_std = tolerant_mean(n_maxs_t) # np.mean(np.vstack(n_maxs_t), axis = 0) # / 100
    # nmt_std = np.std(np.vstack(n_maxs_t), axis = 0) # / 100
    rmt_mean, rmt_std = tolerant_mean(rho_maxs_t) # = np.mean(np.vstack(rho_maxs_t), axis = 0)
    # rmt_std = np.std(np.vstack(rho_maxs_t), axis = 0)
    nmr_mean, nmr_std = tolerant_mean(n_maxs_r)
    rmr_mean, rmr_std = tolerant_mean(rho_maxs_r)
    c_wf_mean = np.mean(np.array(wavefronts_slope))
    c_wf_std = np.std(np.array(wavefronts_slope))
    c_wf_intercept = np.mean(np.array(wavefronts_intercept))
    c_xd_mean = np.mean(np.array(xdrift_pulls_slope))
    c_xd_std = np.std(np.array(xdrift_pulls_slope))
    c_xd_intercept = np.mean(np.array(xdrift_pulls_intercept))
    xd_mean = np.mean(np.array(xdrifts_slope))
    xd_std = np.std(np.array(xdrifts_slope))
    xd_intercept = np.mean(np.array(xdrifts_intercept))
    # c_wf_mean, c_wf_std = tolerant_mean(wavefronts) # np.mean(np.vstack(wavefronts), axis = 0) # * 10
    # c_wf_std = np.std(np.vstack(wavefronts), axis =0) # * 10
    # c_xd_mean, c_xd_std = tolerant_mean(xdrift_pulls) # np.mean(np.vstack(xdrift_pulls), axis = 0)[t0:] # * 10
    # c_xd_std = np.std(np.vstack(xdrift_pulls), axis =0)[t0:] # * 10
    # xd_mean, xd_std = tolerant_mean(xdrifts) # np.mean(np.vstack(xdrifts), axis = 0)[t0:] # * 10
    # xd_std = np.std(np.vstack(xdrifts), axis =0)[t0:] # * 10

    times = np.arange(t0min, tfmax) - t0min

    ax5[0,0].plot(times, nmt_mean)
    ax5[0,0].fill_between(times, nmt_mean - nmt_std, nmt_mean + nmt_std, alpha = 0.3)
    ax5[0,0].set_title('Maximum tip density')
    ax5[0,0].set_xlabel('time(hr)')
    ax5[0,0].set_ylabel(r'$density(mm^{-2})$')
    for nm in n_maxs_t:
        t = times[:len(nm)]
        ax5[0,0].scatter(t, np.array(nm))

    ax5[0,1].plot(times, rmt_mean)
    ax5[0,1].fill_between(times, rmt_mean - rmt_std, rmt_mean + rmt_std, alpha = 0.3)
    ax5[0,1].set_title('Median filament density')
    ax5[0,1].set_xlabel('time(hrs)')
    ax5[0,1].set_ylabel(r'$density(mmmm^{-2})$')
    for rm in rho_maxs_t:
        t = times[:len(rm)]
        ax5[0,1].scatter(t, np.array(rm))

    ax5[0,2].plot(r_max_round, nmr_mean)
    ax5[0,2].fill_between(r_max_round, nmr_mean - nmr_std, nmr_mean + nmr_std, alpha = 0.3)
    ax5[0,2].set_title('Maximum tip density')
    ax5[0,2].set_xlabel('radius(mm)')
    ax5[0,2].set_ylabel(r'$density(mm^{-2})$')
    for nm in n_maxs_r:
        r = r_max_round[:len(nm)]
        ax5[0,2].scatter(r, np.array(nm))

    ax5[0,3].plot(r_max_round, rmr_mean)
    ax5[0,3].fill_between(r_max_round, rmr_mean - rmr_std, rmr_mean + rmr_std, alpha = 0.3)
    ax5[0,3].set_title('Median filament density')
    ax5[0,3].set_xlabel('radius(mm)')
    ax5[0,3].set_ylabel(r'$density(mmmm^{-2})$')
    for rm in rho_maxs_r:
        r = r_max_round[:len(rm)]
        ax5[0,3].scatter(r, np.array(rm))

    # two ways of determining wave speed
    # fitting to filament density to find wavefront position
    t = np.arange(0, tfmax)
    # res = stats.linregress(t, c_wf_mean)
    c_mean = c_wf_mean
    c_std = c_wf_std
    c_intercept = c_wf_intercept
    ax5[1,0].plot(t, t * c_wf_mean + c_wf_intercept, label = f'r={c_wf_mean:.2f}t+{c_wf_intercept:.2f}')
    ax5[1,0].fill_between(t, t * (c_wf_mean - c_wf_std) + c_wf_intercept, t * (c_wf_mean + c_wf_std) + c_wf_intercept, alpha = 0.3)
    ax5[1,0].set_title('Position of wavefront')
    ax5[1,0].legend(loc="upper right")
    ax5[1,0].set_xlabel('time(hr)')
    ax5[1,0].set_ylabel(r'$position(mm)$')
    for wf in wavefronts:
        t = times[:len(wf)]
        ax5[1,0].scatter(t, np.array(wf))
    # "drift position" of puller tips
    t = np.arange(0, tfmax)
    # res = stats.linregress(t, c_xd_mean) 
    ax5[1,1].plot(t, t * c_xd_mean + c_xd_intercept, label = f'r={c_xd_mean:.2f}t+{c_xd_intercept:.2f}')
    # ax5[1,1].plot(t, c_xd_mean)
    ax5[1,1].fill_between(t, t * (c_xd_mean - c_xd_std) + c_xd_intercept, t * (c_xd_mean + c_xd_std) + c_xd_intercept, alpha = 0.3)
    ax5[1,1].set_title('Drift velocity of pulling tips')
    ax5[1,1].legend(loc="upper right")
    ax5[1,1].set_xlabel('time(hr)')
    ax5[1,1].set_ylabel(r'$position(mm)$')
    for xd_pull in xdrift_pulls:
        t = times[:len(xd_pull)]
        ax5[1,1].scatter(t, np.array(xd_pull[t0min:]))
    # "drift position" of hyphae
    t = np.arange(0, tfmax)
    # res = stats.linregress(t, xd_mean)
    ax5[1,2].plot(t, t * xd_mean + xd_intercept, label = f'r={xd_mean:.2f}t+{xd_intercept:.2f}')
    # ax5[1,2].plot(t, xd_mean)
    ax5[1,2].fill_between(t, t * (xd_mean - xd_std) + xd_intercept, t * (xd_mean + xd_std) + xd_intercept, alpha = 0.3)
    ax5[1,2].set_title('Drift velocity of all tips')
    ax5[1,2].legend(loc="upper right")
    ax5[1,2].set_xlabel('time(hr)')
    ax5[1,2].set_ylabel(r'$position(mm)$')
    for xd in xdrifts:
        t = times[:len(xd)]
        ax5[1,2].scatter(t, np.array(xd[t0min:]))
    
    ax5[1,3].set_title('Initial tip density profile')
    ax5[1,3].set_xlabel('radius(mm)')
    ax5[1,3].set_ylabel(r'$density(mm^{-2})$')

    fig5.suptitle('Simulation output')
    fig5.tight_layout()
    
    fig5.savefig(sim_name + '_out.png')
    # res = stats.linregress(times, xd_mean * (np.max(xd_mean) - np.min(xd_mean)) + np.min(xd_mean))
    D0 = (((c_mean - v_sval0) / 2) ** 2) * (1 / alpha0)

    data_r = pd.DataFrame({'n': nmr_mean, 'n_std': nmr_std, 'rho': rmr_mean, 'rho_std': rmr_std, 'r': r_max_round})
    data_r['n_wt'] = (1 / data_r['n_std']) ** 2
    data_r.loc[data_r.n_wt == inf, 'n_wt'] = data_r['n_wt'].min()
    data_r['rho_wt'] = (1 / data_r['rho_std']) ** 2
    data_r.loc[data_r.rho_wt == inf, 'rho_wt'] = data_r['rho_wt'].min()
    data_t = pd.DataFrame({'n': nmt_mean, 'n_std': nmt_std, 'rho': rmt_mean, 'rho_std': rmt_std, 'c': c_mean * times + c_intercept, 'c_std': c_std * times, 't': times})
    data_t['n_wt'] = (1 / data_t['n_std']) ** 2
    data_t.loc[data_t.n_wt == inf, 'n_wt'] = data_t['n_wt'].min()
    data_t['rho_wt'] = (1 / data_t['rho_std']) ** 2
    data_t.loc[data_t.rho_wt == inf, 'rho_wt'] = data_t['rho_wt'].min()
    data_t['c_wt'] = (1 / data_t['c_std']) ** 2
    data_t.loc[data_t.c_wt == inf, 'c_wt'] = data_t['c_wt'].min()
    data_const = pd.DataFrame({'alpha': alpha0, 'beta': beta0, 'vd': v_sval0, 'sg': s_scal0, 'D': D0, 'A': upar[0], 's': upar[1], 'r0': upar[2], 't0': t0min}, index=[0])

    with open(r'constantguess.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([sim_name, alpha0, beta0, v_sval0, s_scal0, D0, upar[0], upar[1], upar[2]])

    data_r.to_csv(sim_name + '_robjectives.csv', index=False)
    data_t.to_csv(sim_name + '_tobjectives.csv', index=False)
    data_const.to_csv(sim_name + '_constantguess.csv', index=False)

    return data_r, data_t, data_const

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # The first argument after the script name is at index 1
        ABM_args = json.loads(sys.argv[1])
        print('Processing data from ' + ABM_args['sim_name'] + '.')
        z = ABM(ABM_args['sim_name'], ABM_args['n_runs'], ABM_args['n_steps'], ABM_args['ring_width'])
    else:
        print("No string argument provided.")