import numpy as np
import os
import sys
import json
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy import optimize
import time as timer
import csv
from AMALGAM import AMALGAM
import bmxfungi_functions

run_dir = os.getcwd()					                    # Add current directory to Python path
if run_dir not in sys.path:
    sys.path.append(run_dir)

parent_dir = os.path.abspath(os.path.join(run_dir, '..'))   # Go up one directory
sys.path.append(parent_dir)	    # Add miscellaneous directory to Python path

# start_time = timer.time()

# Define AMALGAM parameters
AMALGAMPar = {}
AMALGAMPar['N'] = 50    # Population size
AMALGAMPar['T'] = 50    # Number of generations
AMALGAMPar['d'] = 5     # Number of parameters
AMALGAMPar['m'] = 3      # Number of objective functions

# Parameter information for Latin hypercube sampling-based initialization
Par_info = {}
Par_info['initial'] = 'latin'                   # Latin hypercube sampling
Par_info['boundhandling'] = 'reflect'             # Boundary handling

# Define the function name
Func_name = 'bmxfungi_functions.err_func'         # The benchmark function

# Set optimization options
options = {}
options['print'] = 'yes'   # Print output to screen (figures)
options['parallel'] = 'no'

par_names = ['alpha', 'beta', 'vd', 'sg', 'D']

if __name__ == '__main__':
    if len(sys.argv) > 1:
        # setting up, reading in data
        pareto_args = json.loads(sys.argv[1])
        sim_name = pareto_args['sim_name']
        time = pareto_args['time']
        print('Fitting ' + sim_name + '.')
        
        z_t = pd.read_csv(sim_name + '_tobjectives.csv')
        z_r = pd.read_csv(sim_name + '_robjectives.csv')
        z_const = pd.read_csv(sim_name + '_constantguess.csv')
        z = [z_t, z_r, z_const['alpha'][0], z_const['beta'][0], z_const['vd'][0], z_const['sg'][0], z_const['D'][0], z_const['A'][0], z_const['s'][0], z_const['r0'][0], z_const['t0'][0]]
        alpha0 = z[2]
        beta0 = z[3]
        vd0 = z[4]
        sg0 = z[5]
        D0 = z[6]
        par_init = [alpha0, beta0, vd0, sg0, D0]
        upar = [z[7], z[8], z[9]]
        upar[2] = 0.1 # shifting to avoid boundary effects
        bnds = [[par / 10, par * 5] for par in par_init]
        
        # look at initial residuals
        PDE = bmxfungi_functions.PDE
        y0 = PDE([alpha0, beta0, vd0, sg0, D0], upar, time, 1)

        plot_residual = bmxfungi_functions.plot_residual
        err0 = plot_residual(par_init, [0,0,0,0,0], [0,0,0,0,0], {'z': z, 'time': len(z[0])})
        err0.savefig(sim_name + '_initial_residuals.png')

        # seeding initial guess
        Par_info['seed'] = np.array([par_init])
    
        print('\n-----------------seeded parameters from simulation data-----------------\n')
        print(pd.DataFrame(np.array(par_init).reshape(-1,len(par_init)), columns=['alpha', 'beta', 'vd', 'sg', 'D']))
        print('\n')

        
        # Pareto setup 
        Par_info['min'] = (1 / 10) * np.array(par_init)    # Minimum parameter values
        Par_info['max'] = 5 * np.array(par_init)      # Maximum parameter values

        plugin = {
            'fieldnames': ['z', 'time', 'mode'],
            'z': z,
            'time': time, 
            'mode': slice(None)
        }

    # Run the AMALGAM optimization and obtain non-dominated solution set
        X, FX, output, Z, YX, par_best, XZ1 = AMALGAM(AMALGAMPar, sim_name, Func_name, Par_info, options, plugin)
        np.savetxt(sim_name + '_X', X, delimiter=',')
        np.savetxt(sim_name + '_XZ1', XZ1, delimiter=',')

    # X = final population
    # X1 = final population of rank 1
    # XZ1 = rank 1 from any generation

        
        par_median =[par_best.loc['median','alpha'], par_best.loc['median','beta'], par_best.loc['median','vd'], par_best.loc['median','sg'], par_best.loc['median','D']]
        par_std = [par_best.loc['standard deviation','alpha'], par_best.loc['standard deviation','beta'], par_best.loc['standard deviation','vd'], par_best.loc['standard deviation','sg'], par_best.loc['standard deviation','D']]

        fit_dis = bmxfungi_functions.fit_dis
        par_dis = []
        for i in range(AMALGAMPar['d']):
            data = X[:, i]
            par_dis.append([data, int(np.sqrt(int(len(data))))])
        par_lower, par_upper = fit_dis(par_dis, 0.05, sim_name)

        j = 0
        for par in par_lower:
            if par <= 0:
                par_lower[j] = np.min(X[:, j])
            j += 1
            
        pareto_med_row =[sim_name] + par_median
        pareto_pstd_row =[sim_name] + list(np.array(par_median) + np.array(par_std))
        pareto_mstd_row =[sim_name] + list(np.array(par_median) - np.array(par_std))
        
        with open(parent_dir + r'/pareto_median.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(pareto_med_row)
        with open(parent_dir + r'/pareto_pstd.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(pareto_pstd_row)
        with open(parent_dir + r'/pareto_mstd.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow(pareto_mstd_row)
        with open(parent_dir + r'/pareto_ub.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([sim_name] + par_upper)
        with open(parent_dir + r'/pareto_lb.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([sim_name] + par_lower)

        uncb_unc = plot_residual(np.array(par_median), par_upper, par_lower, {'z': z, 'time': len(z[0])})
        uncb_unc.savefig(sim_name + '_uncb_residuals.png')

        # std_unc = plot_residual(np.array(par_median), np.array(par_median) + np.array(par_std), np.array(par_median) - np.array(par_std), {'z': z, 'time': len(z[0])})
        # std_unc.savefig(sim_name + '_std_residuals.png')
        
        # end_time = timer.time()    # Record the end time
        # elapsed_time = end_time - start_time
        # print(f"Elapsed time: {elapsed_time} seconds")
    else:
        print("No string argument provided.")