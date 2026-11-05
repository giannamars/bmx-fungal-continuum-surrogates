from SALib.sample import sobol
from SALib.sample import latin
from SALib.analyze import sobol
from SALib.analyze import pawn
import matplotlib.pyplot as plt
import numpy as np
import re
import pandas as pd
from scipy.interpolate import RBFInterpolator
from SALib.sample import saltelli
from scipy import optimize
from dolfin import *
import matplotlib as mpl
from scipy.optimize import curve_fit
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
from numpy import inf
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import differential_evolution
import dill

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

def get_n(s):
    match = re.search(r'\d+$', s)
    if match:
        return int(match.group())
    else:
        return None

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

    plt.show()

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
    
    x_r = np.array(list(x_r))

    data_t = pd.DataFrame({'n': np.array(n_sat_t), 'rho': np.array(rho_sat_t), 'c': dxdt, 't': np.arange(T)})
    data_r = pd.DataFrame({'n': np.array(n_sat_r), 'rho': np.array(rho_sat_r), 'r': x_r})

    for r in np.arange(data_r['r'].max() + 0.05, 1.05, 0.05):
        data_r.loc[len(data_r)] = [0, 0, round_to_multiple(r, 0.05)]

    return data_t, data_r

def evaluate_PDE_rho(best, upper, lower, upar, time):
    y_t_best, y_r_best = PDE(best, upar, time, 0)
    y_t_best = y_t_best.rename(columns={'n': 'n_best', 'rho': 'rho_best', 'c': 'r_best'})
    y_t_upper, y_r_upper = PDE(upper, upar, time, 0)
    y_t_upper = y_t_upper.rename(columns={'n': 'n_upper', 'rho': 'rho_upper', 'c': 'r_upper'})
    y_t_lower, y_r_lower = PDE(lower, upar, time, 0)
    y_t_lower = y_t_lower.rename(columns={'n': 'n_lower', 'rho': 'rho_lower', 'c': 'r_lower'})
    y_t = y_t_best.merge(y_t_lower, on = 't', how = 'inner')
    y_t = y_t.merge(y_t_upper, on = 't', how = 'inner')
    # y_t['n_mean'] = y_t[['n_best', 'n_lower', 'n_upper']].mean(axis=1)
    y_t['rho_mean'] = y_t[['rho_best', 'rho_lower', 'rho_upper']].mean(axis=1)
    return y_t['rho_mean'].mean()

def evaluate_PDE_c(best, upper, lower, upar, time):
    y_t_best, y_r_best = PDE(best, upar, time, 0)
    y_t_best = y_t_best.rename(columns={'n': 'n_best', 'rho': 'rho_best', 'c': 'r_best'})
    y_t_upper, y_r_upper = PDE(upper, upar, time, 0)
    y_t_upper = y_t_upper.rename(columns={'n': 'n_upper', 'rho': 'rho_upper', 'c': 'r_upper'})
    y_t_lower, y_r_lower = PDE(lower, upar, time, 0)
    y_t_lower = y_t_lower.rename(columns={'n': 'n_lower', 'rho': 'rho_lower', 'c': 'r_lower'})
    y_t = y_t_best.merge(y_t_lower, on = 't', how = 'inner')
    y_t = y_t.merge(y_t_upper, on = 't', how = 'inner')
    y_t['n_mean'] = y_t[['n_best', 'n_lower', 'n_upper']].mean(axis=1)
    y_t['rho_mean'] = y_t[['rho_best', 'rho_lower', 'rho_upper']].mean(axis=1)
    y_t['r_mean'] = y_t[['r_best', 'r_lower', 'r_upper']].mean(axis=1)
    res = stats.linregress(y_t['t'], y_t['r_mean'])
    return res.slope

def evaluate_PDE_par(best, upper, lower, i):
    bul = [best, upper, lower]
    if i < 5:
        return np.mean((best[i], upper[i], lower[i]))
    
    else:
        rho_sat = []
        c = []
        alphas = [best[0], upper[0], lower[0]]
        betas = [best[1], upper[1], lower[1]]
        vds = [best[2], upper[2], lower[2]]
        Ds = [best[4], upper[4], lower[4]]
        ab, ba = np.meshgrid(alphas, betas, indexing='ij')
        vad = np.meshgrid(vds, alphas, Ds, indexing = 'ij') 
        ab_cartesian = np.stack([ab.ravel(), ba.ravel()], axis=-1)
        vad_cartesian = np.stack(vad, axis=-1).reshape(-1, 3)
        for ab in ab_cartesian:
            rho_sat.append(2 * ab[0] / ab[1])
        for vad in vad_cartesian:
            c.append(vad[0] + 2 * np.sqrt(vad[1] * vad[2]))

        if i == 5:
            return np.mean(rho_sat)
        if i == 6:
            return np.mean(c)
        
                        
        
if __name__ == "__main__":
    params = ['max_l', 'split_l', 'max_r', 's_p', 'f_p', 'f_separation', 'W_A']
    param_names = ['chem_species.max_seg_length', 'chem_species.seg_split_length',
                   'chem_species.max_seg_radius', 'chem_species.splitting_probability',
                   'chem_species.fusion_probability', 'chem_species.max_fusion_separation',
                   'cell_force.fungi_boundary_width'] 
    
    headers = ['sim_name', 'alpha', 'beta', 'vd', 'sg', 'D']
    PDE_names = pd.read_csv('/Users/laurenyan/bmxfungi_surrogate/pareto_mean.csv', header=None, names=headers)['sim_name']
    PDE_mn = pd.read_csv('/Users/laurenyan/bmxfungi_surrogate/pareto_mean.csv', header=None, names=headers)[['alpha', 'beta', 'vd', 'sg', 'D']].to_numpy()
    mn_scaler = MinMaxScaler()
    mn_scaler.fit(PDE_mn)
    PDE_mn_norm = mn_scaler.transform(PDE_mn)
    PDE_lb = pd.read_csv('/Users/laurenyan/bmxfungi_surrogate/pareto_lb.csv', header=None, names=headers)[['alpha', 'beta', 'vd', 'sg', 'D']].to_numpy()
    lb_scaler = MinMaxScaler()
    lb_scaler.fit(PDE_lb)
    PDE_lb_norm = lb_scaler.transform(PDE_lb)
    PDE_ub = pd.read_csv('/Users/laurenyan/bmxfungi_surrogate/pareto_ub.csv', header=None, names=headers)[['alpha', 'beta', 'vd', 'sg', 'D']].to_numpy()
    ub_scaler = MinMaxScaler()
    ub_scaler.fit(PDE_ub)
    PDE_ub_norm = ub_scaler.transform(PDE_ub)
    
    ABM_256_inputs = pd.read_csv('/Users/laurenyan/bmxfungi_surrogate/256_inputs/256_inputs.csv')
    
    ABM_sample = np.zeros((len(PDE_names), 7))
    j = 0
    sim_names = PDE_names.to_list()
    for sim in sim_names:
        n = get_n(sim)
        sim_row = ABM_256_inputs.iloc[n]
        i = 0
        for name in param_names:
            ABM_sample[j, i] = sim_row[name]
            i += 1
        j += 1
    
    ABM_scaler = MinMaxScaler()
    ABM_toscale = np.append(ABM_sample, [[17.5e-4, 17.5e-4 - 7.5e-4, 1.25e-4, 0.1, 0.005, 3.0e-4, 0.5e-4]], axis=0) # adding smallest and largest possible values
    ABM_toscale = np.append(ABM_toscale, [[52.5e-4, 52.5e-4 - 2.5e-4, 3.75e-4, 0.3, 0.015, 9.0e-4, 1.5e-4]], axis=0)
    ABM_scaler.fit(ABM_toscale)
    ABM_norm = ABM_scaler.transform(ABM_sample)
    
    with open('/Users/laurenyan/bmxfungi_surrogate/rbf_pickles/alpha_mn_rbf.pkl', 'rb') as f:
         alpha_mn_rbf = dill.load(f)
    with open('/Users/laurenyan/bmxfungi_surrogate/rbf_pickles/beta_mn_rbf.pkl', 'rb') as f:
         beta_mn_rbf = dill.load(f)
    with open('/Users/laurenyan/bmxfungi_surrogate/rbf_pickles/vd_mn_rbf.pkl', 'rb') as f:
         vd_mn_rbf = dill.load(f)
    with open('/Users/laurenyan/bmxfungi_surrogate/rbf_pickles/sg_mn_rbf.pkl', 'rb') as f:
         sg_mn_rbf = dill.load(f)
    with open('/Users/laurenyan/bmxfungi_surrogate/rbf_pickles/D_mn_rbf.pkl', 'rb') as f:
         D_mn_rbf = dill.load(f)
    
    with open('/Users/laurenyan/bmxfungi_surrogate/rbf_pickles/alpha_lb_rbf.pkl', 'rb') as f:
         alpha_lb_rbf = dill.load(f)
    with open('/Users/laurenyan/bmxfungi_surrogate/rbf_pickles/beta_lb_rbf.pkl', 'rb') as f:
         beta_lb_rbf = dill.load(f)
    with open('/Users/laurenyan/bmxfungi_surrogate/rbf_pickles/vd_lb_rbf.pkl', 'rb') as f:
         vd_lb_rbf = dill.load(f)
    with open('/Users/laurenyan/bmxfungi_surrogate/rbf_pickles/sg_lb_rbf.pkl', 'rb') as f:
         sg_lb_rbf = dill.load(f)
    with open('/Users/laurenyan/bmxfungi_surrogate/rbf_pickles/D_lb_rbf.pkl', 'rb') as f:
         D_lb_rbf = dill.load(f)
    
    with open('/Users/laurenyan/bmxfungi_surrogate/rbf_pickles/alpha_ub_rbf.pkl', 'rb') as f:
         alpha_ub_rbf = dill.load(f)
    with open('/Users/laurenyan/bmxfungi_surrogate/rbf_pickles/beta_ub_rbf.pkl', 'rb') as f:
         beta_ub_rbf = dill.load(f)
    with open('/Users/laurenyan/bmxfungi_surrogate/rbf_pickles/vd_ub_rbf.pkl', 'rb') as f:
         vd_ub_rbf = dill.load(f)
    with open('/Users/laurenyan/bmxfungi_surrogate/rbf_pickles/sg_ub_rbf.pkl', 'rb') as f:
         sg_ub_rbf = dill.load(f)
    with open('/Users/laurenyan/bmxfungi_surrogate/rbf_pickles/D_ub_rbf.pkl', 'rb') as f:
         D_ub_rbf = dill.load(f)
    
    SA_in = {
        'num_vars': 7,
        'names': ['max_l', 'split_l', 'max_r', 's_p', 'f_p', 'f_separation', 'W_A'],
        'bounds': [[np.min(ABM_sample[:, 0]) * 1.2, np.max(ABM_sample[:, 0]) * 0.8],
                   [(np.min(ABM_sample[:, 0] - ABM_sample[:, 1])) * 1.2, (np.max(ABM_sample[:, 0] - ABM_sample[:, 1])) * 0.8],
                   [np.min(ABM_sample[:, 2]) * 1.2, np.max(ABM_sample[:, 2]) * 0.8],
                   [np.min(ABM_sample[:, 3]) * 1.2, np.max(ABM_sample[:, 3]) * 0.8],
                   [np.min(ABM_sample[:, 4]) * 1.2, np.max(ABM_sample[:, 4]) * 0.8],
                   [np.min(ABM_sample[:, 5]) * 1.2, np.max(ABM_sample[:, 5]) * 0.8],
                   [np.min(ABM_sample[:, 6]) * 1.2, np.max(ABM_sample[:, 6]) * 0.8]]
    }
    
    ABM_in = saltelli.sample(SA_in, 1024)
    ABM_in[:, 1] = ABM_in[:, 0] - ABM_in[:, 1] # split length is sampled as the difference between split length and maximum length
    ABM_in_norm = ABM_scaler.transform(ABM_in)
    
    PDE_in_mn = mn_scaler.inverse_transform(np.column_stack((alpha_mn_rbf(ABM_in_norm),
                            beta_mn_rbf(ABM_in_norm),
                            vd_mn_rbf(ABM_in_norm),
                            sg_mn_rbf(ABM_in_norm),
                            D_mn_rbf(ABM_in_norm))))
    # PDE_in_mn = np.column_stack((PDE_in_mn,
    #                              2 * PDE_in_mn[:, 0] / PDE_in_mn[:, 1], 
    #                              PDE_in_mn[:, 2] + 2 * np.sqrt(PDE_in_mn[:, 4] * PDE_in_mn[:, 0])))
    PDE_in_lb = lb_scaler.inverse_transform(np.column_stack((alpha_lb_rbf(ABM_in_norm),
                            beta_lb_rbf(ABM_in_norm),
                            vd_lb_rbf(ABM_in_norm),
                            sg_lb_rbf(ABM_in_norm),
                            D_lb_rbf(ABM_in_norm))))
    # PDE_in_lb = np.column_stack((PDE_in_lb,
    #                              2 * PDE_in_lb[:, 0] / PDE_in_lb[:, 1], 
    #                              PDE_in_lb[:, 2] + 2 * np.sqrt(PDE_in_lb[:, 4] * PDE_in_lb[:, 0])))
    PDE_in_ub = ub_scaler.inverse_transform(np.column_stack((alpha_ub_rbf(ABM_in_norm),
                            beta_ub_rbf(ABM_in_norm),
                            vd_ub_rbf(ABM_in_norm),
                            sg_ub_rbf(ABM_in_norm),
                            D_ub_rbf(ABM_in_norm))))
    # PDE_in_ub = np.column_stack((PDE_in_ub,
    #                              2 * PDE_in_ub[:, 0] / PDE_in_ub[:, 1], 
    #                              PDE_in_ub[:, 2] + 2 * np.sqrt(PDE_in_ub[:, 4] * PDE_in_ub[:, 0])))

    PDE_in_mn[:, 0:5][PDE_in_mn[:, 0:5] < 0] = 1e-3
    PDE_in_lb[:, 0:5][PDE_in_lb[:, 0:5] < 0] = 1e-3
    PDE_in_ub[:, 0:5][PDE_in_ub[:, 0:5] < 0] = 1e-3
    
    # PDE_in_mn[:, 5] = 2 * PDE_in_mn[:, 0] / PDE_in_mn[:, 1]
    # PDE_in_lb[:, 5] = 2 * PDE_in_lb[:, 0] / PDE_in_lb[:, 1]
    # PDE_in_ub[:, 5] = 2 * PDE_in_ub[:, 0] / PDE_in_ub[:, 1]
    
    # PDE_in_mn[:, 6] = PDE_in_mn[:, 2] + 2 * np.sqrt(PDE_in_mn[:, 4] * PDE_in_mn[:, 0])
    # PDE_in_lb[:, 6] = PDE_in_lb[:, 2] + 2 * np.sqrt(PDE_in_lb[:, 4] * PDE_in_lb[:, 0])
    # PDE_in_ub[:, 6] = PDE_in_ub[:, 2] + 2 * np.sqrt(PDE_in_ub[:, 4] * PDE_in_ub[:, 0])

    obj_name = ['alpha', 'beta', 'vd', 'sg', 'D', 'rho', 'c']
    # for j in range(len(obj_name)):
    #     PDE_out = np.zeros([PDE_in_mn.shape[0]])
    #     for i, X in enumerate(zip(PDE_in_mn, PDE_in_ub, PDE_in_lb)):
    #         print(i)
    #         PDE_out[i] = evaluate_PDE_par(X[0], X[1], X[2], j)

    #     print(obj_name[j])
        # Si = sobol.analyze(SA_in, PDE_out, print_to_console=True)
        # total_Si, first_Si, second_Si = Si.to_df()
        # total_Si.to_csv('2sobolt_' + obj_name[j] + '.csv')
        # first_Si.to_csv('2sobol1_' + obj_name[j] + '.csv')
        # second_Si.to_csv('2sobol2_' + obj_name[j] + '.csv')

    ABM_in = latin.sample(SA_in, 2000)
    ABM_in[:, 1] = ABM_in[:, 0] - ABM_in[:, 1] # split length is sampled as the difference between split length and maximum length
    ABM_in_norm = ABM_scaler.transform(ABM_in)
    
    PDE_in_mn = mn_scaler.inverse_transform(np.column_stack((alpha_mn_rbf(ABM_in_norm),
                            beta_mn_rbf(ABM_in_norm),
                            vd_mn_rbf(ABM_in_norm),
                            sg_mn_rbf(ABM_in_norm),
                            D_mn_rbf(ABM_in_norm))))
    # PDE_in_mn = np.column_stack((PDE_in_mn,
    #                              2 * PDE_in_mn[:, 0] / PDE_in_mn[:, 1], 
    #                              PDE_in_mn[:, 2] + 2 * np.sqrt(PDE_in_mn[:, 4] * PDE_in_mn[:, 0])))
    PDE_in_lb = lb_scaler.inverse_transform(np.column_stack((alpha_lb_rbf(ABM_in_norm),
                            beta_lb_rbf(ABM_in_norm),
                            vd_lb_rbf(ABM_in_norm),
                            sg_lb_rbf(ABM_in_norm),
                            D_lb_rbf(ABM_in_norm))))
    # PDE_in_lb = np.column_stack((PDE_in_lb,
    #                              2 * PDE_in_lb[:, 0] / PDE_in_lb[:, 1], 
    #                              PDE_in_lb[:, 2] + 2 * np.sqrt(PDE_in_lb[:, 4] * PDE_in_lb[:, 0])))
    PDE_in_ub = ub_scaler.inverse_transform(np.column_stack((alpha_ub_rbf(ABM_in_norm),
                            beta_ub_rbf(ABM_in_norm),
                            vd_ub_rbf(ABM_in_norm),
                            sg_ub_rbf(ABM_in_norm),
                            D_ub_rbf(ABM_in_norm))))
    # PDE_in_ub = np.column_stack((PDE_in_ub,
    #                              2 * PDE_in_ub[:, 0] / PDE_in_ub[:, 1], 
    #                              PDE_in_ub[:, 2] + 2 * np.sqrt(PDE_in_ub[:, 4] * PDE_in_ub[:, 0])))

    in_out = ABM_in
    in_out = np.column_stack((in_out, in_out[:,0] / (in_out[:,2] ** 2)))
    
    obj_name = ['alpha', 'beta', 'vd', 'sg', 'D', 'rho', 'c']
    for j in range(len(obj_name)):
        PDE_out = np.zeros([PDE_in_mn.shape[0]])
        for i, X in enumerate(zip(PDE_in_mn, PDE_in_ub, PDE_in_lb)):
            print(i)
            PDE_out[i] = evaluate_PDE_par(X[0], X[1], X[2], j)
        
        print(obj_name[j])
        in_out = np.column_stack((in_out, PDE_out))
        
        # Si = pawn.analyze(SA_in, ABM_in, PDE_out, S=10, print_to_console=True)
        # Si.to_df().to_csv('2pawn_' + obj_name[j] + '.csv')

    in_out_df = pd.DataFrame(in_out, columns=['max_l', 'split_l', 'max_r', 's_p', 'f_p', 'f_separation', 'W_A', 'resistance', 'alpha', 'beta', 'vd', 'sg', 'D', 'rho', 'c'])

    print(in_out_df)