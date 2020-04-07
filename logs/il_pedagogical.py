# Standard libraries
import json

# pip imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.integrate as integrate
from pandas import DataFrame
from mpl_toolkits.mplot3d import Axes3D


# non-pip
from potentials import *
from visualization import *
from lattice import *
from il import *

def adsorption_energies(lattice, dissociation_type=1, potential_type=1, **kwargs):
    """
    Get activation energy of a site based on local environment.
    Based off Morse potential, V(r) = D(1 - exp(-a(r-r_eq)))
    inputs:
    r -  4 x 1 np array of distances to nearest neighbors on the lattice
    dissociation_type - int
        1: takes weakest bond as bond broken
        2: takes ensemble avg over all bonds

    potential_type - in
        1: Morse potential, w(r) = D(1 - exp(-a(r-r_eq)))
            kwargs:
            r_eq - float of non-dimensional "equilibrium" bond length between lattice sites 
            D - well depth
            a - force constant like value

        2: Gaussian potential, w(r) = -D * exp(r**2/(2 * sigma**2))
            kwargs:
            D - potential depth
            std - standard deviation of Gaussian
    """
    # FIXME: Be able to take in NN distances with edges pruned
    # FIXME: Theres some reworking needing to be done in the arguments here
    NN_distances = nearest_neighbor_distances(lattice, None)
    
    if potential_type == 1:
        V = morse_potential(NN_distances, kwargs['r_eq'], kwargs['D'], kwargs['a'])

    elif potential_type == 2:
        V = normal(NN_distances, kwargs['D'], 0, kwargs['std'])

    if dissociation_type == 1:
        adsorption_energies = -np.asarray(np.max(V,axis=2)) -0.2*kwargs['D']
    
    elif dissociation_type == 2:
        B = 1000./(8.314 * kwargs['T'])
        exp_V = np.exp(-1 * B * V)
        adsorption_energies = 1 / B * np.asarray(np.log(np.sum(exp_V,axis=2))) -0.2*kwargs['D']

    # Return non-boundary sites
    return adsorption_energies[1:-1, 1:-1]

if __name__ == "__main__":

    initial_pool_size = 100    
    output = 'logs/g_potential_{}.json'.format(initial_pool_size)
    
    lattice_size = 500
    a = 35 # ~ Potential width, dimensionless
    D = 60 # Potential depth, kJ/mol
    r_eq = 1.0 # Equilibrium distance of unperturbed lattice, dimensionless
    T = 300
    sigma = morse_std(r_eq, D, a, T)
    delta = 0.1
    gauss_pot_std = 0.9

    lower = 0.0
    upper = 0.5 
    # sigma = 0.1
    mean = 0.0

    displacement_type = 'uniform'
    dissociation_type = 2
    potential_type = 2

    if displacement_type == 'trunc_norm':
        disp_kwargs = {'lower' : lower, 'upper' : upper, 'sigma' : delta, 'mean' : mean}
    else:
        disp_kwargs = False

    if potential_type == 1:
        potential_kwargs = {'r-eq': r_eq, 'D' : D}
    else:
        potential_kwargs = {'D' : D, 'std' : gauss_pot_std}

    lattice = make_quenched_disorder_lattice(lattice_size, delta, displacement_type, False, lower, upper, delta, mean)
    decorated_lattice = decorate_lattice(lattice, 0.50, 0.25, 0.25)

    graftable_sites, competing_sites = locate_grafting_sites(decorated_lattice)
    # plot_lattice(decorated_lattice, True, graftable_sites)

    D_MO = 500.
    r_eq_MO = 1.
    a_MO = 2.0

    D_M_O = 120.
    r_eq_M_O = 1.
    a_M_O = 1.3

    # D_MO = 500.
    # r_eq_MO = 1.
    # a_MO = 1.

    # D_M_O = 120.
    # r_eq_M_O = 1.
    # a_M_O = 1.

    # D_MO = 500.
    # r_eq_MO = 1.
    # a_MO = 1.02

    # D_M_O = 150.
    # r_eq_M_O = 1.
    # a_M_O = 1.

    local_coordinates = compute_local_coordinates(decorated_lattice, graftable_sites)


    graft_E,ads_E = grafting_Energies(D_MO, r_eq_MO, a_MO, D_M_O, r_eq_M_O, a_M_O, graftable_sites, decorated_lattice)


    # plt.hist(graft_E,bins=20)
    # plt.show()

    # plt.hist(ads_E,bins=20)
    # plt.show()
    NN_distances = nearest_neighbor_distances(decorated_lattice, graftable_sites)
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(15,6))
    histogram(ax1,ax2,ads_E,T,n_bins=20)
    plt.show()
    plt.hist(graft_E,bins=20)
    plt.show()
    # data1 = np.hstack((np.transpose(graft_E),np.transpose(ads_E)))
    data2 = np.transpose(np.vstack((np.transpose(graft_E),np.transpose(ads_E))))
    data2 = DataFrame(data2)
    sns.pairplot(data2)
    plt.show()

    # Randomly choose sites to train
    n_samples = 70
    sampled_sites = np.random.choice(np.shape(graftable_sites)[0],size =  n_samples,replace=False)
    print(sampled_sites)
    sampled_barrier_heights = ads_E[sampled_sites]
    sampled_local_coordinates = local_coordinates[sampled_sites]

    #TRAINING
    M, model_barriers_LOO, residuals = train(sampled_local_coordinates, sampled_barrier_heights)
    
    # plt.scatter(model_barriers_LOO,sampled_barrier_heights)
    # plt.show()
    fig, ax = plt.subplots(1, 1, figsize=(7,6))
    plot_trained(ax, model_barriers_LOO, sampled_barrier_heights, 2.5, n_samples)
    plt.show()

    # plt.hist(ads_E-predicted_adsorption_energies(sampled_local_coordinates,sampled_barrier_heights,M,local_coordinates,np.shape(graftable_sites)[0]))
    # plt.show()
    # print(np.sqrt(np.var(ads_E-predicted_adsorption_energies(sampled_local_coordinates,sampled_barrier_heights,M,local_coordinates,np.shape(graftable_sites)[0]))))

    # plt.hist2d(ads_E,graft_E,bins=20)
    # plt.show()
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    B = 1000/8.314/T

    x = ads_E * (np.exp(-B * ads_E))/(np.sum(np.exp(-B * ads_E)))
    x = ads_E
    y = graft_E
    hist, xedges, yedges = np.histogram2d(x, y, bins=10)

    # Construct arrays for the anchor positions of the 16 bars.
    xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = 0

    # Construct arrays with the dimensions for the 16 bars.
    dx = dy = 2. * np.ones_like(zpos)
    dz = hist.ravel()
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', zsort='average')


    B = 1000/8.314/T

    # x = ads_E * (np.exp(-B * ads_E))/(np.sum(np.exp(-B * ads_E)))
    # y = graft_E
    # hist, xedges, yedges = np.histogram2d(x, y, bins=4)

    # # Construct arrays for the anchor positions of the 16 bars.
    # xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
    # xpos = xpos.ravel()
    # ypos = ypos.ravel()
    # zpos = 0

    # # Construct arrays with the dimensions for the 16 bars.
    # dx = dy = 0.5 * np.ones_like(zpos)
    # dz = hist.ravel()
    # ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='r', zsort='average')




    plt.show()
   

    # Get model predicted barrier heights for all l^2 sites
    # model_barrier_heights = predicted_adsorption_energies(sampled_local_coordinates, sampled_barrier_heights, M, local_coordinates, np.shape(graftable_sites)[0]) 

    # Ads_E = adsorption_energies(lattice, 2, 2, r_eq=r_eq, D=D, std=gauss_pot_std, T=T)

    # NN_distances = nearest_neighbor_distances(lattice)[1:-1, 1:-1]
    # barrier_distribution = Ads_E + 50

    # ### True Average ###
    # B = 1000/8.314/T
    # site_avg_E = np.mean(np.exp(-B*barrier_distribution)*(barrier_distribution))/np.mean(np.exp(-B*barrier_distribution))

    # # Flatten from 2D to 1D arrays
    # NN_distances = np.reshape(NN_distances, ((lattice_size - 2)**2, 4))
    # NN_distances.sort()
    
    # barrier_distribution = np.ravel(barrier_distribution)
    # # fig, axes = plt.subplots(1, 2, sharex=True, figsize=(14,6))
    # # histogram(axes[0], axes[1], barrier_distribution, T, 35)
    # # plt.show()
    # ### Start Importance Learning ###
    # n_bins = 20

    # IL = importance_learning(barrier_distribution, NN_distances, T, initial_pool_size, 100, plot_every=25)
    
    # parameters = {
    #     'lattice-length' : lattice_size,
    #     'displacement-type' : 'uniform',
    #     'displacement-kwargs' : disp_kwargs,
    #     'displacement-amt' : delta,
    #     'dissociation-type' : dissociation_type,
    #     'potential-type' : potential_type,
    #     'potential-kwargs' : potential_kwargs,
    #     'T' : T,
    #     'initial-pool-size' : initial_pool_size
    # }

    # # Export stuff
    # results = {
    #     'parameters' : parameters,
    #     'NN-distances' : NN_distances.tolist(),
    #     'barrier-heights' : barrier_distribution.tolist(),
    #     'True <Ea>k' : site_avg_E,
    #     'importance-learning' : IL
    # }

    # a = json.dumps(results, ensure_ascii=False, indent=2)
    # with open(output, 'w') as outfile:
    #     outfile.write(a)    

