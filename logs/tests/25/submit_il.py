import il_pedagogical as il
import numpy as np
import json


il_path = "home/salman/il-pedagogical"

output = 'logs/g_potential_xx.json'

lattice_size = 150
a = 30. # ~ Potential width, dimensionless
D = 35. # Potential depth, kJ/mol
r_eq = 1.0 # Equilibrium distance of unperturbed lattice, dimensionless
T = 373
sigma = il.morse_std(r_eq, D, a, T)
delta = 0.7*sigma
gauss_pot_std = 0.9

lower = 0.0
upper = 0.5 
# sigma = 0.1
mean = 0.0

displacement_type = 'uniform'
dissociation_type = 1
potential_type = 1

if displacement_type == 'trunc_norm':
    disp_kwargs = {'lower' : lower, 'upper' : upper, 'sigma' : delta, 'mean' : mean}
else:
    disp_kwargs = False

if potential_type == 1:
    potential_kwargs = {'r-eq': r_eq, 'D' : D}
else:
    potential_kwargs = {'D' : D, 'std' : gauss_pot_std}

lattice = il.make_quenched_disorder_lattice(lattice_size, delta, displacement_type, False, lower, upper, delta, mean)
# plot_lattice(lattice)
NN_distances = il.nearest_neighbor_distances(lattice)
Ads_E = il.adsorption_energies(lattice, dissociation_type,potential_type, r_eq=r_eq, D=D, a=a, T=T)

NN_distances = il.nearest_neighbor_distances(lattice)[1:-1, 1:-1]
barrier_distribution = Ads_E #+ 25

### True Average ###
B = 1000/8.314/T
site_avg_E = np.mean(np.exp(-B*barrier_distribution)*(barrier_distribution))/np.mean(np.exp(-B*barrier_distribution))

# Flatten from 2D to 1D arrays
NN_distances = np.reshape(NN_distances, ((lattice_size - 2)**2, 4))
NN_distances.sort()

barrier_distribution = np.ravel(barrier_distribution)
il.histogram(barrier_distribution, T, 35)

### Start Importance Learning ###
n_bins = 20
initial_pool_size = 25

IL = il.importance_learning(barrier_distribution, NN_distances, T, initial_pool_size, 100,1000)

parameters = {
    'lattice-length' : lattice_size,
    'displacement-type' : 'uniform',
    'displacement-kwargs' : disp_kwargs,
    'displacement-amt' : delta,
    'dissociation-type' : dissociation_type,
    'potential-type' : potential_type,
    'potential-kwargs' : potential_kwargs,
    'T' : T,
    'initial-pool-size' : initial_pool_size
}

# Export stuff
results = {
    'parameters' : parameters,
    'NN-distances' : NN_distances.tolist(),
    'barrier-heights' : barrier_distribution.tolist(),
    'True <Ea>k' : site_avg_E,
    'importance-learning' : IL
}

a = json.dumps(results, ensure_ascii=False, indent=2)
with open(output, 'w') as outfile:
    outfile.write(a)    
