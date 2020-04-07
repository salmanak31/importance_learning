# Standard libraries
import json

# pip imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.integrate as integrate

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

def histogram_graft_mod(h, b, t, T):
    kb = 1.38064852*10**(-23)
    h_1 = 6.62607004*10**(-34)
    mid_hist = np.asarray([(b[i]+b[i+1])/2 for i in range(np.shape(b)[0]-1)])
    return h*np.exp(-(kb*T/h_1)*np.exp(-(mid_hist*1000)/(8.314*T))*t)

def grafting_population(graft_E, ads_E, n_bins, time):
    
    graft_E = np.reshape(graft_E,(np.shape(graft_E)[0],1))
    ads_E = np.reshape(ads_E,(np.shape(ads_E)[0],1))

    histogram_graft, bins_graft = np.histogram(graft_E,n_bins)

    # bin index of graft_E elements
    bin_index_r = np.digitize(graft_E, bins_graft, right=False)
    bin_index_l = np.digitize(graft_E, bins_graft, right=True)
    diff = bin_index_r - bin_index_l
    sum = bin_index_r + bin_index_l
    diff_sum = diff*sum
    for item in np.where(diff_sum > 1)[0]:
        bin_index_r[item] = np.shape(bins_graft)[0] - 1 
    for item in np.where(diff_sum == 1)[0]:
        bin_index_r[item] = 1
    bin_index = bin_index_r
    bin_index = np.reshape(bin_index,(np.shape(bin_index)[0],1))
    ##END##

    # array combining grafting bin index, ads_E, and grafting energy (sorted using bin index)
    combined = np.hstack((ads_E,graft_E,bin_index))
    combined_sorted = combined[combined[:,2].argsort()]
    
    # np.savetxt("test_101.txt",combined_sorted)
    
    # set up populations = # of bins. This is the starting population i.e. all the sites are ungrafted at the moment
    populations = []
    for i in range(0,int(combined_sorted[np.argmax(combined_sorted[:,2])][2])):
        populations.append(combined_sorted[np.where(combined_sorted[:,2]==float(i+1))])

    # for i in range(0,len(populations)):
    #     np.savetxt(str(i)+".txt",np.asarray(populations[i]))
    
    # Calculate the decay in ungrafted population as a function of time
    # This is a list of length = # of bins and each element is the number of sites that have to be delted
    delta_pop = [int(np.round(histogram_graft - histogram_graft_mod(histogram_graft, bins_graft, time))[i]) for i in range(np.shape(histogram_graft)[0])]

    # generate random numbers to delete sites based on the population decrease
    rand_delete = []
    for i in range(0,len(histogram_graft)):
        # generate random numbers for bins which have a non zero population
        if len(populations[i])>0:
            rand_delete.append(np.random.randint( 0, len(populations[i]), size = delta_pop[i]))
        # append an empty list if the bin population = 0 (since there is nothing left to graft)
        else:
            rand_delete.append([])

    # grafted sites
    populations = np.asarray(populations)

    # list of adsorption energies of grafted sites
    grafted_pop_ads = np.array([])

    for i in range(0,len(populations)):
        # concatenate the list of all grafted sites
        grafted_pop_ads = np.concatenate([grafted_pop_ads, populations[i][rand_delete[i]][:,0]])

    # population change after grafting
    for i in range(len(rand_delete)):
        # Skip if the number of sites in a histogram are zero (nothing left to graft) or if there is no reduction in the number of sites (i.e. the random list for this bin is empty)
        if len(rand_delete[i]) == 0 or np.shape(populations[i])[0] == 0 :
            pass
        else:
            populations[i] = np.delete(populations[i],rand_delete[i],axis = 0)
    # populations = new population

    # population of grafted sites (only grafting energies)
    # this can be used to construct the decay of graftable site populations (as a function of deltaG_graft)
    graft_E_modified = np.array([])
    for i in range(0,len(populations)):
        graft_E_modified = np.concatenate([graft_E_modified, populations[i][:,1]])

    return populations, grafted_pop_ads, graft_E_modified


if __name__ == "__main__":

    initial_pool_size = 50    
    output = 'logs/test_trunc_norm/g_potential_{}.json'.format(initial_pool_size)

    lattice_size =200
    a = 35 # ~ Potential width, dimensionless
    D = 60 # Potential depth, kJ/mol
    r_eq = 1.0 # Equilibrium distance of unperturbed lattice, dimensionless
    T = 300
    delta = 0.15

    # DONT NEED
    # sigma = morse_std(r_eq, D, a, T)
    gauss_pot_std = 0.9
    # DONT NEED

    # DONT NEED
    lower = 0.0
    upper = 0.1
    sigma = 0.11
    mean = 0.0
    # DONT NEED

    displacement_type = 'uniform'

    # DONT NEED
    dissociation_type = 2
    potential_type = 2
    # DONT NEED



#SAVED PARAMS
    # D_MO = 500.
    # r_eq_MO = 1.
    # a_MO = 1.8

    # D_M_O = 110
    # r_eq_M_O = 1.
    # a_M_O = 1.5

    D_MO = 500.
    r_eq_MO = 1.
    a_MO = 1.5

    D_M_O = 300.
    r_eq_M_O = 1.
    a_M_O = 1.3

    empty_fraction = 0.3
    OH_fraction = 0.3
    siloxane_fraction = 0.4

    lattice_fractions = {'empty' : empty_fraction, 'OH' : OH_fraction, 'Siloxane' : siloxane_fraction}
    MO_Morse = {'D' : D_MO, 'a' : a_MO, 'r_eq' : r_eq_MO}
    siloxane_Morse = {'D' : D_M_O, 'a' : a_M_O, 'r_eq' : r_eq_M_O}

    # These don't do anything ... yet
    unstrained_energies = {'M-A': 1, 'M-O' : 1, 'M---O' : 1} 

    if displacement_type == 'trunc_norm':
        disp_kwargs = {'lower' : lower, 'upper' : upper, 'sigma' : delta, 'mean' : mean}
    else:
        disp_kwargs = False

    if potential_type == 1:
        potential_kwargs = {'r-eq': r_eq, 'D' : D}
    else:
        potential_kwargs = {'D' : D, 'std' : gauss_pot_std}

    lattice = make_quenched_disorder_lattice(lattice_size, delta, displacement_type, True, False, lower, upper, sigma, mean)
    decorated_lattice = decorate_lattice(lattice, empty_fraction, OH_fraction, siloxane_fraction)

    graftable_sites, competing_sites = locate_grafting_sites(decorated_lattice)
    # print(np.shape(graftable_sites))
    # plot_lattice(decorated_lattice, True, graftable_sites)

    local_coordinates = compute_local_coordinates(decorated_lattice, graftable_sites)
    local_coordinates_dict = {'OH-OH-distance' : local_coordinates[:, 0].tolist(),
                              'siloxane-distances' : local_coordinates[:, 1].tolist(),
                              'OH-siloxane-angle' : local_coordinates[:, 2].tolist(), 
                              'OH-siloxane-midpoint' : local_coordinates[:, 3].tolist()}

    graft_E, ads_E = grafting_Energies(D_MO, r_eq_MO, a_MO, D_M_O, r_eq_M_O, a_M_O, graftable_sites, decorated_lattice)

    n_bins = 20
    time = 1
    populations, grafted_pop_ads, graft_E_modified = grafting_population(graft_E, ads_E, n_bins, time)
    # loop_log = [ 10**i for i in range(0,10)]
    # h_red = []
    # data = []
    # print(np.shape(mid_hist))
    # for i, item  in enumerate(loop_log,start=0):
    #     h_red.append(h*np.exp(-(kb*T/h_1)*np.exp(-(mid_hist*1000)/(8.314*T))*item))
    #     h_red[i] = [ int(h_red[i][j]) for j in range(len(h_red[i])) ]


    # print(combined[:,1])
    # max_bin = np.max(combined[:,1])
    # rho_bins = []
    # for i in range(1,int(max_bin+1)):
    #     count = (combined_sorted[:,1].tolist()).count(i)
    #     rho_bins.append(combined_sorted[0:count,0])


    # mid_hist = [(b[i]+b[i+1])/2 for i in range(np.shape(b)[0]-1)]
    # print(mid_hist,np.shape(mid_hist))
    # mid_hist = np.asarray(mid_hist)


    # kb = 1.38064852*10**(-23)
    # h_1 = 6.62607004*10**(-34)
    # loop_log = [ 10**i for i in range(0,10)]
    # print(loop_log)
    # h_red = []
    # data = []
    # print(np.shape(mid_hist))
    # for i, item  in enumerate(loop_log,start=0):
    #     h_red.append(h*np.exp(-(kb*T/h_1)*np.exp(-(mid_hist*1000)/(8.314*T))*item))
    #     h_red[i] = [ int(h_red[i][j]) for j in range(len(h_red[i])) ]
    #     data1 = []
    #     for f, v in zip(np.reshape(np.asarray(h_red[i]),(20,)), mid_hist):
    #         data1.extend([v] * f)

    #     data.append(data1)



    # for i in range(0,len(h_red)):
    #     plt.hist(h_red[i],alpha = 0.5)

    # plt.show()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.set_xlabel(r'$E_a$, kJ/mol')
    ax2.set_xlabel(r'$E_a$, kJ/mol')
    histogram(ax1, ax2, graft_E, T, n_bins=20)
    plt.show()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.set_xlabel(r'$E_a$, kJ/mol')
    ax2.set_xlabel(r'$E_a$, kJ/mol')
    histogram(ax1, ax2, ads_E, T, n_bins=20)
    plt.show()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.set_xlabel(r'$E_a$, kJ/mol')
    ax2.set_xlabel(r'$E_a$, kJ/mol')
    histogram(ax1, ax2, grafted_pop_ads, T, n_bins=20)
    plt.show()


    time = 10
    populations, grafted_pop_ads, new_pop = grafting_population(graft_E, ads_E, n_bins, time)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.set_xlabel(r'$E_a$, kJ/mol')
    ax2.set_xlabel(r'$E_a$, kJ/mol')
    histogram(ax1, ax2, grafted_pop_ads, T, n_bins=20)
    plt.show()

    time = 100
    populations, grafted_pop_ads, new_pop = grafting_population(graft_E, ads_E, n_bins, time)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.set_xlabel(r'$E_a$, kJ/mol')
    ax2.set_xlabel(r'$E_a$, kJ/mol')
    histogram(ax1, ax2, grafted_pop_ads, T, n_bins=20)
    plt.show()

    time = 1000
    populations, grafted_pop_ads, new_pop = grafting_population(graft_E, ads_E, n_bins, time)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.set_xlabel(r'$E_a$, kJ/mol')
    ax2.set_xlabel(r'$E_a$, kJ/mol')
    histogram(ax1, ax2, grafted_pop_ads, T, n_bins=20)
    plt.show()

    time = 100000
    populations, grafted_pop_ads, new_pop = grafting_population(graft_E, ads_E, n_bins, time)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.set_xlabel(r'$E_a$, kJ/mol')
    ax2.set_xlabel(r'$E_a$, kJ/mol')
    histogram(ax1, ax2, grafted_pop_ads, T, n_bins=20)
    plt.show()

    time = 1000000
    populations, grafted_pop_ads, new_pop = grafting_population(graft_E, ads_E, n_bins, time)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.set_xlabel(r'$E_a$, kJ/mol')
    ax2.set_xlabel(r'$E_a$, kJ/mol')
    histogram(ax1, ax2, grafted_pop_ads, T, n_bins=20)
    plt.show()
    


    
    time = 100000000
    populations, grafted_pop_ads, new_pop = grafting_population(graft_E, ads_E, n_bins, time)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.set_xlabel(r'$E_a$, kJ/mol')
    ax2.set_xlabel(r'$E_a$, kJ/mol')
    histogram(ax1, ax2, grafted_pop_ads, T, n_bins=20)
    plt.show()
    # site_avg_E = k_weighted_avg_activation_E(ads_E, T)
    # print(site_avg_E)

    # NN_distances = nearest_neighbor_distances(decorated_lattice, graftable_sites)



    n_samples = 70
    sampled_sites = np.random.choice(np.shape(graftable_sites)[0],size =  n_samples,replace=False)
    sampled_barrier_heights = ads_E[sampled_sites]
    sampled_local_coordinates = local_coordinates[sampled_sites]

    #TRAINING
    # M, model_barriers_LOO, residuals = train(sampled_local_coordinates, sampled_barrier_heights)



    # plt.scatter(model_barriers_LOO,sampled_barrier_heights)
    # plt.show()

    # fig, ax = plt.subplots(1, 1, figsize=(7,6))
    # plot_trained(ax, model_barriers_LOO, sampled_barrier_heights, 2.5, n_samples)
    # plt.show()

    # plt.hist(ads_E-predicted_adsorption_energies(sampled_local_coordinates,sampled_barrier_heights,M,local_coordinates,np.shape(graftable_sites)[0]))
    # plt.show()

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
    n_bins = 20

    # IL = importance_learning(ads_E, local_coordinates, T, initial_pool_size, 200, plot_every=1000)
    
    # parameters = {
    #     'lattice-length' : lattice_size,
    #     'displacements' : {'displacement-type' : 'uniform',
    #                        'displacement-kwargs' : disp_kwargs,
    #                        'displacement-amt' : delta},
    #     # 'dissociation-type' : dissociation_type,
    #     'site-fractions' : lattice_fractions,
    #     'potentials' : {'MO-potential' : MO_Morse,
    #                     'siloxane-potential': siloxane_Morse,
    #                     'unstrained-energies' : unstrained_energies},
    #     'number-graftable-sites' : len(graftable_sites),
    #     'number-competing-sites' : len(competing_sites),
    #     # 'potential-type' : potential_type,
    #     # 'potential-kwargs' : potential_kwargs,
    #     'T' : T,
    #     'initial-pool-size' : initial_pool_size
    # }

    # # Export stuff
    # results = {
    #     'parameters' : parameters,
    #     'local-coordinates' : local_coordinates_dict,
    #     'competing-sites' : competing_sites.tolist(),
    #     'grafting-barrier-heights' : graft_E.tolist(),
    #     'barrier-heights' : ads_E.tolist(),
    #     'True <Ea>k' : site_avg_E,
    #     'importance-learning' : IL
    # }

    # a = json.dumps(results, ensure_ascii=False, indent=2)
    # with open("output.json", 'w') as outfile:
    #     outfile.write(a)    

