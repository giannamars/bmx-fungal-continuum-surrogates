import shutil
from scipy.stats import qmc
import math

# default parameters (that we might change)
parameters = {}

parameters['D_ABC'] = [6.0e-10, 0.0, 6.0e-10]

parameters['seed'] = 43343
parameters['split_l'] = 30.0e-4
parameters['max_vol'] = 1e-12
parameters['max_l'] = 35.0e-4
parameters['max_r'] = 2.5e-4

parameters['k1'] = 2.0e-5
parameters['kr1'] = 2.0e-5
parameters['k2'] = 4.0
parameters['kr2'] = 0.0006
parameters['k3'] = 2.0e-5
parameters['kr3'] = 2.0e-5
parameters['k_g'] = 4.0
parameters['k_v'] = 40000.0
parameters['M_A'] = 2.0e-5
parameters['M_B'] = 2.0e-5
parameters['M_C'] = 2.0e-5

parameters['b_p'] = 0.0000
parameters['s_p'] = 0.2
parameters['f_p'] = 0.01
parameters['f_separation'] = 6.0e-4

parameters['eta'] = 1.0e6
parameters['zeta'] = 1.0e5
parameters['W_A'] = 1.0e-4
parameters['W_Z'] = 1.0e-4
parameters['f_bond'] = 1.0

# sampling
params = ['max_l', 'split_l', 'max_r', 's_p', 'f_p', 'f_separation', 'W_A'] # parameters that will be sampled, can be changed to any of the above in the parameters dictionary
ranges = [[17.5, 52.5], [2.5, 7.5], [1.25, 3.75], [0.1, 0.3], [0.005, 0.015], [3.0, 9.0], [0.5, 1.5]]
to_change = dict(zip(params, ranges))
samples = {}
n_samples = 256 # must be power of 2 for sobol sequence

sampler = qmc.Sobol(d=len(params), scramble=True)
sample = sampler.random_base2(m=int(math.log2(n_samples)))
i = 0
for param in to_change.keys():
    if param == 'split_l':
        # to ensure split length is less than maximum length
        samples[param] = samples['max_l'] - (sample[:,i] * (to_change[param][1] - to_change[param][0]) + to_change[param][0]) 
    else:
        samples[param] = sample[:, i] * (to_change[param][1] - to_change[param][0]) + to_change[param][0]
    i += 1

# generate input files
for input_number in range(0,n_samples):
    file_name = 'input_fungi_' + str(input_number)
    shutil.copyfile('template_input_fungi', file_name) # copy template to new file

    for param in params:
        parameters[param] = samples[param][input_number]

    with open(file_name, 'a') as f:
        f.write('bmx.seed = ' + str(parameters['seed']) + '\nfluid.chem_species_diff = ' + str(parameters['D_ABC'][0]) + ' ' + str(parameters['D_ABC'][1]) + ' ' + str(parameters['D_ABC'][2]) + '\nchem_species.max_vol = ' + str(parameters['max_vol']) + '\nchem_species.seg_split_length = ' + str(round(parameters['split_l'], 2))  + 'e-4' + '\nchem_species.max_seg_length = '  + str(round(parameters['max_l'], 2)) + 'e-4' + '\nchem_species.max_seg_radius = ' + str(round(parameters['max_r'], 2)) + 'e-4' + '\nchem_species.k1 = ' + str(parameters['k1']) + '\nchem_species.kr1 = ' + str(parameters['kr1']) + '\nchem_species.k2 = ' + str(parameters['k2']) + '\nchem_species.kr2 = ' + str(parameters['kr2']) + '\nchem_species.k3 = ' + str(parameters['k3']) + '\nchem_species.kr3 = ' + str(parameters['kr3']) + '\nchem_species.kg = ' + str(parameters['k_g']) + '\nchem_species.kv = ' + str(parameters['k_v']) + '\nchem_species.mass_transfer_A = ' + str(parameters['M_A']) + '\nchem_species.mass_transfer_B = ' + str(parameters['M_B']) + '\nchem_species.mass_transfer_C = ' + str(parameters['M_C']) + '\nchem_species.branching_probability = ' + str(round(parameters['b_p'], 2)) + '\nchem_species.splitting_probability = ' + str(round(parameters['s_p'], 2)) + '\nchem_species.fusion_probability = ' + str(round(parameters['f_p'], 2)) + '\nchem_species.max_fusion_separation = ' + str(round(parameters['f_separation'], 2))  + 'e-4' + '\ncell_force.fungi_stiffness = ' + str(parameters['eta']) + '\ncell_force.fungi_wall_stiffness = ' + str(parameters['zeta']) + '\ncell_force.fungi_boundary_width = ' + str(round(parameters['W_A'], 2))  + 'e-4' + '\ncell_force.fungi_wall_boundary_width = ' + str(parameters['W_Z']) + '\ncell_force.bond_strength = ' + str(parameters['f_bond']))