import numpy as np
from scipy import optimize
# get_ipython().run_line_magic('matplotlib', 'widget')
from dolfin import *
import matplotlib as mpl
mpl.use('Agg')
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

def round_to_signif(x, p):
    x = np.asarray(x)
    x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10**(p-1))
    mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
    return np.round(x * mags) / mags

def PDE(pars, upar, time, plot):
    '''
    pars = dictionary of parameters to fit (key = parameter name, value = parameter value).
    upar = parameters of initial condition [A, s, r0].
    time = hours.
    plot = whether or not to show plot of PDE solution (1=yes).
    '''
    alphval = pars[0]
    beta = Constant(pars[1])
    v_sval = pars[2]
    s_scal =  Constant(pars[3])
    D = Constant(pars[4])

    # Define the problem
    Length = 3 # 40.0
    left = 1e-3
    dX = 0.01
    X = np.arange(left, Length, dX)
    nx = len(X)
    nT = int(time * 10) # 900
    keep = 10 # get every hour
    mesh = IntervalMesh(nx, left, Length)
    P1 = FiniteElement('P', interval, 1)
    element = MixedElement([P1, P1])
    V = FunctionSpace(mesh, element)
    cmap2 = mpl.colormaps["spring"]
    cmap = mpl.colormaps["winter"]
    set_log_level(LogLevel.WARNING)

    density_tip_t = np.zeros((int(nT / keep), nx))
    density_in_t = np.zeros((int(nT / keep), nx))
    n_sat_r = []
    rho_sat_r = []
    x_r = []

    # Define the parameters
    T = time # 90
    dt = T/nT
    # alphval = 0.039
    cval = 0.6
    # Define the initial condition
    u0_str = str(upar[0]) + '*exp(-0.5*' + str(upar[1]) + '*pow(abs(x[0]-' + str(upar[2]) + '),2))'
    u0 = Expression((u0_str, '0.0'), degree=1)
    #define radial coordinate
    rexp = Expression(('1.0/x[0]', '0.0'), degree=1)
    r = interpolate(rexp, V)
    invr, _ = split(r)
    # Define other elements

    u_n = interpolate(u0, V)
    u = Function (V)
    n, rho = split(u)
    n_n, rho_n= split(u_n)
    v_1, v_2 = TestFunctions(V)
    k = Constant((1,))

    # s_scal =  Constant(0.0288) # Constant(0.2)
    # beta = Constant(0.022)
    alpha = Constant(alphval)
    # D = Constant(0.02)

    n_mid = 0.5*(n_n + n)
    # v_sval = 0.235
    v_s = Constant((v_sval,))

    F = n * v_1 * dx - n_n * v_1 * dx
    # Diffusion term
    F += D * dot(grad(n), grad(v_1)) * dt * dx
    # Cylindrical correction terms using invr
    F += invr * D * dot(grad(n), Constant((1.0,))) * v_1 * dt * dx
    F += dt * dot(grad(n_mid), v_s) * v_1 * dx
    F += invr * dt * n_mid * Constant(v_sval) * v_1 * dx

    F -= dt * s_scal * n * v_2 * dx - (rho - rho_n) * v_2 * dx
    F += beta * v_1 * rho * n * dt * dx
    F -= alpha * n * v_1 * dt * dx

    # Apply the boundary conditions (left=left, right=Length)
    u_L = (Constant(0.0), Constant(0.0))
    u_R = (Constant(0.0), Constant(0.0))
    bc_L = DirichletBC(V, u_L, f'near(x[0], {left})')
    bc_R = DirichletBC(V, u_R, f'near(x[0], {Length})')
    bcs = [bc_L, bc_R]

    t = 0.0
    a, L = lhs(F), rhs(F)
    V2 = FunctionSpace(mesh, 'P', 1)
    # X = np.arange(left, Length, dx)
    poss = []

    i=0

    if plot == 1:
        fig, axs = plt.subplots(3, 1)
    for t in np.linspace(0,T,nT):
        try:
            solve(F==0, u,bcs)
            u_n.assign(u)
            _n, _rho= u.split()

            if i%keep==0:# i%100==0:
                n_field = interpolate(_n, V2)
                rho_field = interpolate(_rho, V2)
                density_tip_t[int(i / keep), :] = [n_field(x) for x in X]
                density_in_t[int(i / keep), :] = [rho_field(x) for x in X]
                poss.append(np.argmax([n_field(x) for x in X]))
                if plot == 1:
                    axs[0].plot(X, [n_field(x) for x in X], color=cmap2(t / T))
                    axs[1].plot(X, [rho_field(x) for x in X], color=cmap(t / T))

                    if i > nT / 2:
                        axs[2].plot(X, [n_field(x) for x in X], color=cmap2(t / T))

            xi = np.argmax([n_field(x) for x in X]) * (Length / nx) - upar[2]
            if round(xi, 2) % 0.05 <= 0.02:
                if round_to_multiple(xi, 0.05) not in x_r:
                    rho_nonzero = density_in_t[int(i / keep), :][density_in_t[int(i / keep), :] > 0.1]
                    n_sat_r.append(max([n_field(x) for x in X]))
                    if len(rho_nonzero) > 0:
                        rho_sat_r.append(np.median(rho_nonzero))
                    else:
                        rho_sat_r.append(0)
                    x_r.append(round_to_multiple(xi, 0.05))

        except RuntimeError as e:
            print('hour ' + str(t))
            print(e)
            if i%keep==0:# i%100==0:
                density_tip_t[int(i / keep), :] = np.zeros(len(X))
                density_in_t[int(i / keep), :] = np.zeros(len(X))
                poss.append(0)
        i += 1

    plt.close()

    dxdt = np.array(poss)*Length/nx - upar[2]

    n_sat_t = []
    rho_sat_t = []

    for t in np.arange(T):
        # collect PDE output for each hour
        # saturation density
        n_sat_t.append(np.max(density_tip_t[t, :]))
        rho_nonzero = density_in_t[t, :][density_in_t[t, :] > 0.1]
        if len(rho_nonzero) > 0:
            rho_sat_t.append(np.median(rho_nonzero))
        else:
            rho_sat_t.append(0)
            
        # to_fit_in = density_in[t, :]
        # cs = CubicSpline(X, to_fit_in)

        # for j in range(len(X)):
        #     if cs(X, 1)[j] < 0 and cs(X, 1)[j-1] > 0:
        #         cutoff = j
        #         break
        #     else:
        #         cutoff = 0
        # try:
        #     K_guess = np.max(to_fit_in[cutoff:])
        #     L_guess = np.max(X)
        #     r0_guess = np.median(X[cutoff:])
        #     p0_sig = [K_guess, L_guess, r0_guess] # initial guess 
        #     popt_sig, pcov_sig = curve_fit(sigmoid_r, X[cutoff:], to_fit_in[cutoff:], p0_sig)
        #     if popt_sig[0] > np.max(to_fit_in):
        #         rho_sat.append(np.max(to_fit_in))
        #     else:
        #         rho_sat.append(popt_sig[0])
        # except RuntimeError as e:
        #     rho_sat.append(np.max(to_fit_in))
        #     print('hour ' + str(t))
        #     print(e)


    x_r = np.array(list(x_r))

    data_t = pd.DataFrame({'n': np.array(n_sat_t), 'rho': np.array(rho_sat_t), 'c': dxdt, 't': np.arange(T)})
    data_r = pd.DataFrame({'n': np.array(n_sat_r), 'rho': np.array(rho_sat_r), 'r': x_r})

    for r in np.arange(data_r['r'].max() + 0.05, 1.05, 0.05):
        data_r.loc[len(data_r)] = [0, 0, round_to_multiple(r, 0.05)]

    return data_t, data_r

def err_func(pars, args):
    '''
    pars = parameters of PDE to optimize about
    args = arguments based on simulation ({'z': z, 'time': len(z[0]), 'mode' : slice(None)})
    z = ABM output for each hour except 0 (ABM(['bmx_default0', 'bmx_default1', 'bmx_default2'], 500000, 0.005))
    time = hours (from length of z data)
    mode = 0, 1, 2, slice(None) to return n, rho, or c residual, or the full array.
    '''
    # print(pars)
    z, time, mode = args['z'], args['time'], args['mode']

    # z = [z_t, z_r, 
    # z_const['alpha'], z_const['beta'], z_const['vd'], z_const['sg'], z_const['D'],
    # z_const['A'], z_const['s'], z_const['r0'], z_const['t0']]
    
    upar = [z[7], z[8], z[9]]
    upar[2] = 0.1
    
    t0 = z[10]
    shifted_time = time - t0
    
    z_t, z_r = z[0:2]
    y_t, y_r = PDE(pars, upar, time, 0)
 
    r_data = z_r.merge(y_r, on = 'r', how = 'inner', suffixes = ('_z', '_y'))
    r_data['n_res'] = abs(r_data['n_z'] - r_data['n_y'])
    r_data['rho_res'] = abs(r_data['rho_z'] - r_data['rho_y'])
    r_data['n_wssr'] = (r_data['n_wt'] * r_data['n_res']) ** 2
    r_data['rho_wssr'] = (r_data['rho_wt'] * r_data['rho_res']) ** 2

    t_data = z_t.merge(y_t, on = 't', how = 'inner', suffixes = ('_z', '_y'))
    t_data['n_res'] = abs(t_data['n_z'] - t_data['n_y'])
    t_data['rho_res'] = abs(t_data['rho_z'] - t_data['rho_y'])
    t_data['n_wssr'] = (t_data['n_wt'] * t_data['n_res']) ** 2
    t_data['rho_wssr'] = (t_data['rho_wt'] * t_data['rho_res']) ** 2
    t_data['c_res'] = abs(t_data['c_z'] - t_data['c_y'])
    t_data['c_wssr'] = (t_data['c_wt'] * t_data['c_res']) ** 2

    return np.array([t_data['n_wssr'].sum(), t_data['rho_wssr'].sum(), t_data['c_wssr'].sum()])[mode]

def single_opt(m, d, par_init, bnds, args):
    '''
    m = number of objectives.
    d = number of parameters to fit.
    par_init = initial guess of parameters.
    bnds = bounds of guess range for fitting.
    args = arguments based on simulation.
    '''
    par_best = np.zeros((m, d)) 
    for i in np.arange(m):
        args['mode'] = i
        popt = optimize.minimize(err_func, x0=par_init, args=(args), bounds = bnds, options = {'maxfun': 100, 'maxiter': 100})
        print(popt.x)
        par_best[i, :] = popt.x

    return par_best


def fit_dis(par_dis, alpha, sim_name):
    '''
    par_dis = list of pareto distributions for each parameter and the calculated number of bins for the histogram [[distribution1, nbins1], ...].
    alpha = alpha for ks test fitting distributions.
    sim_name = name of simulation for labeling output file.
    '''
    par_names = ['alpha', 'beta', 'vd', 'sg', 'D']
    par_units = [r'$hr^{-1}$', r'$mmhr^{-1}$', r'$mmhr^{-1}$', r'$mmhr^{-1}$', r'$mm^{2}hr^{-1}$']
    lower = []
    upper = []
    d = len(par_dis)
    fig, ax = plt.subplots(1,d, figsize = (3 * d, 3))
    for i in range(d):
        pd = par_dis[i]
        data = pd[0]
        bins = pd[1]
        pval_best = 0
        common_dis = [stats.lognorm,  stats.chi2, stats.gamma, stats.uniform, stats.expon, stats.pareto, stats.weibull_min, stats.rayleigh]
        for cd in common_dis:
            # fitting each common distribution
            par_guess = cd.fit(data)
            ks_stat, pval = stats.kstest(data, cd.cdf, args = par_guess)
            if pval > pval_best:
                # saving best fit distribution
                pval_best = pval
                dis_best = cd
                par_best = par_guess
        x = np.linspace(min(data), max(data), 1000)
        pdf_best = dis_best.pdf(x, *par_best)
    
        lower.append(dis_best.ppf(alpha / 2, *par_best))
        upper.append(dis_best.ppf(1 - alpha / 2, *par_best))
        
        y_hist, x_hist, _ = ax[i].hist(data, bins, density=True)
        ax[i].plot(x, pdf_best)
        ax[i].vlines(lower[i], 0, y_hist.max(), linestyle = '--', color = 'r')
        ax[i].vlines(upper[i], 0, y_hist.max(), linestyle = '--', color = 'r')
        ax[i].set_title(par_names[i] + '(' + dis_best.name + ')')
        ax[i].set_xlabel(par_names[i] + ' value(' + par_units[i] + ')')

    fig.tight_layout()
    plt.savefig(sim_name + '_parameter_distributions.png')
    plt.close()
    return lower, upper
    
def plot_residual(best, upper, lower, args):
    '''
    best = best parameters of PDE
    upper = upper bound parameters of PDE (use [0, 0, 0, 0, 0] to skip)
    lower = lower bound parameters of PDE
    args = arguments based on simulation ({'z': z, 'time': len(z[0]))
    z = ABM output for each hour except 0 (ABM(['bmx_default0', 'bmx_default1', 'bmx_default2'], 500000, 0.005))
    time = hours (from length of z data)
    '''
    z, time = args['z'], args['time']
    
    upar = [z[7], z[8], z[9]]
    upar[2] = 0.1
    
    t0 = z[10]
    shifted_time = time - t0
    
    z_t, z_r = z[0:2]
    y_t, y_r = PDE(best, upar, time, 0)
 
    r_data = z_r.merge(y_r, on = 'r', how = 'inner', suffixes = ('_z', '_y'))
    r_data['n_res'] = abs(r_data['n_z'] - r_data['n_y'])
    r_data['rho_res'] = abs(r_data['rho_z'] - r_data['rho_y'])
    r_data['n_wssr'] = (r_data['n_wt'] * r_data['n_res']) ** 2
    r_data['rho_wssr'] = (r_data['rho_wt'] * r_data['rho_res']) ** 2

    t_data = z_t.merge(y_t, on = 't', how = 'inner', suffixes = ('_z', '_y'))
    t_data['n_res'] = abs(t_data['n_z'] - t_data['n_y'])
    t_data['rho_res'] = abs(t_data['rho_z'] - t_data['rho_y'])
    t_data['n_wssr'] = (t_data['n_wt'] * t_data['n_res']) ** 2
    t_data['rho_wssr'] = (t_data['rho_wt'] * t_data['rho_res']) ** 2
    t_data['c_res'] = abs(t_data['c_z'] - t_data['c_y'])
    t_data['c_wssr'] = (t_data['c_wt'] * t_data['c_res']) ** 2

    rmse_n_z = np.sqrt(((t_data['n_res'] ** 2).sum()) / len(t_data['n_res'])) / (t_data['n_z'].max() - t_data['n_z'].min())
    rmse_rho_z = np.sqrt(((t_data['rho_res'] ** 2).sum()) / len(t_data['rho_res'])) / (t_data['rho_z'].max() - t_data['rho_z'].min())
    rmse_c_z = np.sqrt(((t_data['c_res'] ** 2).sum()) / len(t_data['c_res'])) / (t_data['c_z'].max() - t_data['c_z'].min())
    rmse_n_r = np.sqrt(((r_data['n_res'] ** 2).sum()) / len(r_data['n_res'])) / (r_data['n_z'].max() - r_data['n_z'].min())
    rmse_rho_r = np.sqrt(((r_data['rho_res'] ** 2).sum()) / len(r_data['rho_res'])) / (r_data['rho_z'].max() - r_data['rho_z'].min())


    if upper[0] != 0:
        y_t_lower, y_r_lower = PDE(lower, upar, time, 0)
        y_t_upper, y_r_upper = PDE(upper, upar, time, 0)
        y_t_uncertainty = y_t.merge(y_t_lower, on = 't', how = 'inner', suffixes = (None, '_lower'))
        y_t_uncertainty = y_t_uncertainty.merge(y_t_upper, on = 't', how = 'inner', suffixes = (None, '_upper'))
        y_r_uncertainty = y_r.merge(y_r_lower, on = 'r', how = 'inner', suffixes = (None, '_lower'))
        y_r_uncertainty = y_r_uncertainty.merge(y_r_upper, on = 'r', how = 'inner', suffixes = (None, '_upper'))

        fig, axs = plt.subplots(3, 5, figsize = (30, 12))
        subfigs = fig.subfigures(1, 5) 
        axs[0,0].plot(t_data['t'], t_data['n_z'], color = 'y')
        axs[0,0].plot(t_data['t'], t_data['n_y'], color ='r')
        axs[0,0].fill_between(t_data['t'], t_data['n_z'] - t_data['n_std'], t_data['n_z'] + t_data['n_std'], alpha = 0.3, color = 'y')
        axs[0,0].set_title(f'Maximum tip density: RMSE={rmse_n_z:.2f}')
        axs[0,1].plot(t_data['t'], t_data['rho_z'], color = 'y')
        axs[0,1].plot(t_data['t'], t_data['rho_y'], color ='r')
        axs[0,1].fill_between(t_data['t'], t_data['rho_z'] - t_data['rho_std'], t_data['rho_z'] + t_data['rho_std'], alpha = 0.3, color = 'y')
        axs[0,1].set_title(f'Median filament density: RMSE={rmse_rho_z:.2f}')
        axs[0,2].plot(t_data['t'], t_data['c_z'], color = 'y')
        axs[0,2].plot(t_data['t'], t_data['c_y'], color ='r')
        axs[0,2].fill_between(t_data['t'], t_data['c_z'] - t_data['c_std'], t_data['c_z'] + t_data['c_std'], alpha = 0.3, color = 'y')
        axs[0,2].set_title(f'Wave speed: RMSE={rmse_c_z:.2f}')
        axs[0,3].plot(r_data['r'], r_data['n_z'], color = 'y')
        axs[0,3].plot(r_data['r'], r_data['n_y'], color ='r')
        axs[0,3].fill_between(r_data['r'], r_data['n_z'] - r_data['n_std'], r_data['n_z'] + r_data['n_std'], alpha = 0.3, color = 'y')
        axs[0,3].set_title(f'Median tip density: RMSE={rmse_n_r:.2f}')
        axs[2,3].set_xlabel('radius(mm)')
        axs[0,4].plot(r_data['r'], r_data['rho_z'], color = 'y')
        axs[0,4].plot(r_data['r'], r_data['rho_y'], color ='r')
        axs[0,4].fill_between(r_data['r'], r_data['rho_z'] - r_data['rho_std'], r_data['rho_z'] + r_data['rho_std'], alpha = 0.3, color = 'y')
        axs[0,4].set_title(f'Median filament density: RMSE={rmse_rho_r:.2f}')
        axs[1,0].plot(t_data['t'], t_data['n_res'])
        axs[1,0].set_title('Residuals')
        axs[1,1].plot(t_data['t'], t_data['rho_res'])
        axs[1,1].set_title('Residuals')
        axs[1,2].plot(t_data['t'], t_data['c_res'])
        axs[1,2].set_title('Residuals')
        axs[1,3].plot(r_data['r'], r_data['n_res'])
        axs[1,3].set_title('Residuals')
        axs[1,4].plot(r_data['r'], r_data['rho_res'])
        axs[1,4].set_title('Residuals')
        axs[2,0].plot(t_data['t'], t_data['n_wssr'], color = 'mediumpurple')
        axs[2,0].set_title('Weighted residuals')
        axs[2,1].plot(t_data['t'], t_data['rho_wssr'], color = 'mediumpurple')
        axs[2,1].set_title('Weighted residuals')
        axs[2,2].plot(t_data['t'], t_data['c_wssr'], color = 'mediumpurple')
        axs[2,2].set_title('Weighted residuals')
        axs[2,3].plot(r_data['r'], r_data['n_wssr'], color = 'mediumpurple')
        axs[2,3].set_title('Weighted residuals')
        axs[2,4].plot(r_data['r'], r_data['rho_wssr'], color = 'mediumpurple')
        axs[2,4].set_title('Weighted residuals')
        axs[0,0].plot(y_t_uncertainty['t'], y_t_uncertainty['n_lower'], 'r--')
        axs[0,0].plot(y_t_uncertainty['t'], y_t_uncertainty['n_upper'], 'r--')
        axs[0,1].plot(y_t_uncertainty['t'], y_t_uncertainty['rho_lower'], 'r--')
        axs[0,1].plot(y_t_uncertainty['t'], y_t_uncertainty['rho_upper'], 'r--')
        axs[0,2].plot(y_t_uncertainty['t'], y_t_uncertainty['c_lower'], 'r--')
        axs[0,2].plot(y_t_uncertainty['t'], y_t_uncertainty['c_upper'], 'r--')
        axs[0,3].plot(y_r_uncertainty['r'],y_r_uncertainty['n_lower'], 'r--')
        axs[0,3].plot(y_r_uncertainty['r'],y_r_uncertainty['n_upper'], 'r--')
        axs[0,4].plot(y_r_uncertainty['r'], y_r_uncertainty['rho_lower'], 'r--')
        axs[0,4].plot(y_r_uncertainty['r'], y_r_uncertainty['rho_upper'], 'r--')
        subfigs[0].supxlabel('time(hr)')
        subfigs[0].supylabel(r'$density(mm^{-2})$')
        subfigs[1].supxlabel('time(hrs)')
        subfigs[1].supylabel(r'$density(mmmm^{-2})$')
        subfigs[2].supxlabel('time(hr)')
        subfigs[2].supylabel(r'$position(mm)$')
        subfigs[3].supxlabel(r'$density(mm^{-2})$')
        subfigs[3].supylabel(r'$density(mm^{-2})$')
        subfigs[4].supxlabel('radius(mm)')
        subfigs[4].supylabel(r'$density(mmmm^{-2})$')
    else:
        fig, axs = plt.subplots(3, 5, figsize = (30, 12))
        subfigs = fig.subfigures(1, 5) 
        axs[0,0].plot(t_data['t'], t_data['n_z'], color = 'y')
        axs[0,0].plot(t_data['t'], t_data['n_y'], color ='r')
        axs[0,0].fill_between(t_data['t'], t_data['n_z'] - t_data['n_std'], t_data['n_z'] + t_data['n_std'], alpha = 0.3, color = 'y')
        axs[0,0].set_title(f'Maximum tip density: RMSE={rmse_n_z:.2f}')
        axs[0,1].plot(t_data['t'], t_data['rho_z'], color = 'y')
        axs[0,1].plot(t_data['t'], t_data['rho_y'], color ='r')
        axs[0,1].fill_between(t_data['t'], t_data['rho_z'] - t_data['rho_std'], t_data['rho_z'] + t_data['rho_std'], alpha = 0.3, color = 'y')
        axs[0,1].set_title(f'Median filament density: RMSE={rmse_rho_z:.2f}')
        axs[0,2].plot(t_data['t'], t_data['c_z'], color = 'y')
        axs[0,2].plot(t_data['t'], t_data['c_y'], color ='r')
        axs[0,2].fill_between(t_data['t'], t_data['c_z'] - t_data['c_std'], t_data['c_z'] + t_data['c_std'], alpha = 0.3, color = 'y')
        axs[0,2].set_title(f'Wave speed: RMSE={rmse_c_z:.2f}')
        axs[0,3].plot(r_data['r'], r_data['n_z'], color = 'y')
        axs[0,3].plot(r_data['r'], r_data['n_y'], color ='r')
        axs[0,3].fill_between(r_data['r'], r_data['n_z'] - r_data['n_std'], r_data['n_z'] + r_data['n_std'], alpha = 0.3, color = 'y')
        axs[0,3].set_title(f'Median tip density: RMSE={rmse_n_r:.2f}')
        axs[2,3].set_xlabel('radius(mm)')
        axs[0,4].plot(r_data['r'], r_data['rho_z'], color = 'y')
        axs[0,4].plot(r_data['r'], r_data['rho_y'], color ='r')
        axs[0,4].fill_between(r_data['r'], r_data['rho_z'] - r_data['rho_std'], r_data['rho_z'] + r_data['rho_std'], alpha = 0.3, color = 'y')
        axs[0,4].set_title(f'Median filament density: RMSE={rmse_rho_r:.2f}')
        axs[1,0].plot(t_data['t'], t_data['n_res'])
        axs[1,0].set_title('Residuals')
        axs[1,1].plot(t_data['t'], t_data['rho_res'])
        axs[1,1].set_title('Residuals')
        axs[1,2].plot(t_data['t'], t_data['c_res'])
        axs[1,2].set_title('Residuals')
        axs[1,3].plot(r_data['r'], r_data['n_res'])
        axs[1,3].set_title('Residuals')
        axs[1,4].plot(r_data['r'], r_data['rho_res'])
        axs[1,4].set_title('Residuals')
        axs[2,0].plot(t_data['t'], t_data['n_wssr'], color = 'mediumpurple')
        axs[2,0].set_title('Weighted residuals')
        axs[2,1].plot(t_data['t'], t_data['rho_wssr'], color = 'mediumpurple')
        axs[2,1].set_title('Weighted residuals')
        axs[2,2].plot(t_data['t'], t_data['c_wssr'], color = 'mediumpurple')
        axs[2,2].set_title('Weighted residuals')
        axs[2,3].plot(r_data['r'], r_data['n_wssr'], color = 'mediumpurple')
        axs[2,3].set_title('Weighted residuals')
        axs[2,4].plot(r_data['r'], r_data['rho_wssr'], color = 'mediumpurple')
        axs[2,4].set_title('Weighted residuals')
        subfigs[0].supxlabel('time(hr)')
        subfigs[0].supylabel(r'$density(mm^{-2})$')
        subfigs[1].supxlabel('time(hrs)')
        subfigs[1].supylabel(r'$density(mmmm^{-2})$')
        subfigs[2].supxlabel('time(hr)')
        subfigs[2].supylabel(r'$position(mm)$')
        subfigs[3].supxlabel(r'$density(mm^{-2})$')
        subfigs[3].supylabel(r'$density(mm^{-2})$')
        subfigs[4].supxlabel('radius(mm)')
        subfigs[4].supylabel(r'$density(mmmm^{-2})$')
    
    fig.suptitle('PDE fit to simulation')
    plt.tight_layout()
    plt.close()
    # plt.show()
    return fig



    