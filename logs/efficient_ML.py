import sys
sys.path.append('../')

from il_pedagogical import *
import numpy as np
import timeit
import functools
from pyswarm import pso


lattice_size = 75

###### Gaussian #######
a = 35 # ~ Potential width, dimensionless
D = 60 # Potential depth, kJ/mol
r_eq = 1.0 # Equilibrium distance of unperturbed lattice, dimensionless
T = 300
sigma = morse_std(r_eq, D, a, T)
delta = 0.35
gauss_pot_std = 0.9
dissociation_type = 2
potential_type = 2

####### Morse ########
# lattice_size = 75
# a = 30.
# D = 35.
# r_eq = 1.0 # Equilibrium distance of unperturbed lattice, dimensionless
# T = 373
# sigma = morse_std(r_eq, D, a, T)
# delta = 1.75 * sigma
# potential_type = 1
# dissociation_type = 2

lower = 0.0
upper = 0.5 
# sigma = 0.1
mean = 0.0

n_iterations = 5
initial_pool_size = 50
displacement_type = 'uniform'


if displacement_type == 'trunc_norm':
    disp_kwargs = {'lower' : lower, 'upper' : upper, 'sigma' : delta, 'mean' : mean}
else:
    disp_kwargs = False

if potential_type == 1:
    potential_kwargs = {'r-eq': r_eq, 'D' : D}
else:
    potential_kwargs = {'D' : D, 'std' : gauss_pot_std}

lattice = make_quenched_disorder_lattice(lattice_size, delta, displacement_type, False, lower, upper, delta, mean)
# plot_lattice(lattice)
NN_distances = nearest_neighbor_distances(lattice)
Ads_E = adsorption_energies(lattice, dissociation_type, potential_type, r_eq=r_eq, D=D, std=gauss_pot_std, T=T)
# Ads_E = adsorption_energies(lattice, dissociation_type, potential_type, r_eq=r_eq, D=D, a=a, T=T)


NN_distances = nearest_neighbor_distances(lattice)[1:-1, 1:-1]
barrier_distribution = Ads_E + 50

### True Average ###
B = 1000/8.314/T
site_avg_E = np.mean(np.exp(-B*barrier_distribution)*(barrier_distribution))/np.mean(np.exp(-B*barrier_distribution))

# Flatten from 2D to 1D arrays
NN_distances = np.reshape(NN_distances, ((lattice_size - 2)**2, 4))
NN_distances.sort()

barrier_distribution = np.ravel(barrier_distribution)
fig, axes = plt.subplots(1, 2, sharex=True, figsize=(14,6))
histogram(axes[0], axes[1], barrier_distribution, T, 35)
plt.show()
### Start Importance Learning ###
n_bins = 20

lattice_len = int(np.sqrt(len(barrier_distribution)) + 2)
IL = {}
sampled_sites, sampled_barrier_heights, sampled_NN_distances = inspect_sites(lattice_len, NN_distances, barrier_distribution, initial_pool_size)

M, model_barriers_LOO, residuals = train(sampled_NN_distances, sampled_barrier_heights)

# def check_precision_accuracy(sampled_NN_distances, sampled_barrier_heights, M, NN_distances, lattice_size):
# """
# Check accuracy of model predicted sites computed by:
# (1.) longdouble precision + infinite precision where underflow happens
# (2.) All infinite precision (which is sloooow but exact)

# Returns L2 norm of the difference
# """
# standard_prec_barriers = predicted_adsorption_energies(sampled_NN_distances, sampled_barrier_heights, M, NN_distances, lattice_size)
    # decimal_prec_barriers = predicted_adsorption_energies_d(sampled_NN_distances, sampled_barrier_heights, M, NN_distances, lattice_size)
    
# return(np.linalg.norm(standard_prec_barriers - decimal_prec_barriers))

# def compare_time(sampled_NN_distances, sampled_barrier_heights, M, NN_distances, lattice_size, n_times=3):
#     t_long = timeit.Timer(functools.partial(predicted_adsorption_energies, sampled_NN_distances, sampled_barrier_heights, M, NN_distances, lattice_size))
#     t_decimal = timeit.Timer(functools.partial(predicted_adsorption_energies_d, sampled_NN_distances, sampled_barrier_heights, M, NN_distances, lattice_size))

#     print('Long/arbitrary precision mix (s): {}'.format(t_long.timeit(n_times)))
#     print('Arbitrary precision all sites (s): {}'.format(t_decimal.timeit(n_times)))
#     pass

# accuracy = check_precision_accuracy(sampled_NN_distances, sampled_barrier_heights, M, NN_distances, lattice_size)

# print('||Ea(long) - Ea(arbitrary)|| = {}'.format(accuracy))

# compare_time(sampled_NN_distances, sampled_barrier_heights, M, NN_distances, lattice_size)

# this = []
# def loss_func(x):
#     L = np.zeros((4,4))
#     L[0] = [x[0],0,0,0]
#     L[1] = [x[1],x[2],0,0]
#     L[2] = [x[3],x[4],x[5],0]
#     L[3] = [x[6],x[7],x[8],x[9]]
#     M = np.matmul(L,np.transpose(L))
#     X_train= sampled_NN_distances
#     Y_train = sampled_barrier_heights
#     model_predicted_barriers = []
#     Y_train =  [Decimal(Y_train[i]) for i in range(0,len(Y_train))]
#     for i in range(0,len(Y_train)):
#         test = X_train[i]
#         temp = np.delete(X_train,i,axis=0)
#         delta = temp-test
#         dist_mat = np.diagonal(np.matmul(np.matmul(delta,M),np.transpose(delta)))
#         dist_mat = np.transpose([Decimal(dist_mat[i]) for i in range(len(dist_mat))])
#         k_mat = np.exp(-1*dist_mat)
#         temp2 = np.delete(Y_train,i,axis=0)
#         model_predicted_barriers.append(np.sum(np.multiply(temp2,k_mat))/(np.sum(k_mat)))
#     model_predicted_barriers  = np.asarray(model_predicted_barriers)
#     residuals = np.linalg.norm(Y_train - model_predicted_barriers)
#     residuals = float(residuals)
#     return residuals
# lb = np.zeros(10)
# ub = 100*np.ones(10)
# xopt, fopt = pso(loss_func,lb,ub)

# L_opt = np.zeros((4,4))
# L_opt[0] = [xopt[0],0,0,0]
# L_opt[1] = [xopt[1],xopt[2],0,0]
# L_opt[2] = [xopt[3],xopt[4],xopt[5],0]
# L_opt[3] = [xopt[6],xopt[7],xopt[8],xopt[9]]
# M_opt = np.matmul(L_opt,np.transpose(L_opt))
# M_opt = np.asarray(M_opt)
# # fig, axes = plt.subplots(1,1, sharex=True, figsize=(14,6))
M_opt = M
X_train= sampled_NN_distances
Y_train = sampled_barrier_heights
model_predicted_barriers = []
Y_train =  [Decimal(Y_train[i]) for i in range(0,len(Y_train))]
for i in range(0,len(Y_train)):
    test = X_train[i]
    temp = np.delete(X_train,i,axis=0)
    delta = temp-test
    dist_mat = np.diagonal(np.matmul(np.matmul(delta,M_opt),np.transpose(delta)))
    dist_mat = np.transpose([Decimal(dist_mat[i]) for i in range(len(dist_mat))])
    k_mat = np.exp(-1*dist_mat)
    temp2 = np.delete(Y_train,i,axis=0)
    model_predicted_barriers.append(np.sum(np.multiply(temp2,k_mat))/(np.sum(k_mat)))
model_predicted_barriers  = np.asarray(model_predicted_barriers)
residuals_2 = ((Y_train - model_predicted_barriers)**2).sum()

model_predicted_barriers =  [float(model_predicted_barriers[i]) for i in range(0,len(model_predicted_barriers))]
print(model_predicted_barriers)
# model_barriers_2 = model_predicted_barriers


# plt.rcParams['figure.figsize'] = 13, 13
# fig = plt.figure()
# ax = fig.add_subplot(1, 2, 1)


# plot_trained(ax, model_barriers_LOO, sampled_barrier_heights, 2.5, 50)
# ax = fig.add_subplot(1, 2, 2)
# plot_trained(ax, model_barriers_2, sampled_barrier_heights, 2.5, 50)
# plt.show()


# print(M)
# print(M_opt)