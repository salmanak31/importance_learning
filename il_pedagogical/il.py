import warnings
import json
from decimal import Decimal

import numpy as np
import matplotlib.pyplot as plt
from metric_learn import MLKR

from visualization import *

def k_weighted_avg_activation_E(barrier_heights, T):
    """
    Computes k weighted mean from a distribtuion

    input -
    barrier_heights - nx1 numpy array of barrier heights in kJ/mol
    T - float of catalyst operation temperature

    returns - 
    k_weight_mean - k weighted mean of barrier heights in kJ/mol
    """
    B = 1000/8.314/T
    
    return np.sum(barrier_heights * np.exp(-B * barrier_heights))/(np.sum(np.exp(-B * barrier_heights)))

def k_weighted_square_avg_activation_E(barrier_heights, T):

    B = 1000/8.314/T
    return np.sum(barrier_heights**2 * np.exp(-B * barrier_heights))/(np.sum(np.exp(-B * barrier_heights)))

def random_sample_sites(barrier_distribution, n_samples):
    """
    Sample random sites for the initial pool
    """
    return np.random.choice(np.arange(len(barrier_distribution)), n_samples, replace=False)


def inspect_sites(local_coordinates, barrier_heights, n_samples):
    """
    Like doing a DFT calculation on a site. 
    Inputs - 
    lattice - numpy array of lattice of sites (ensemble of sites), 
    nearest_neighbor_distances - n x n x 4 array of distances to nearest neighbors. Local coordinates
    barrier_heights - n x n  np array of barrier heights from model chemistry
    n_samples - number of sites to sample

    returns - 
    sampled_sites - n_samples x 2 array of randomly sampled site IDs
    sampled_barrier_heights - n_samples x 1 np array, true barrier heights of sampled sites
    sampled_NN_distances - n_samples x 4 np array, array of nearest neighbor distances of sampled sites
    """
    
    sampled_sites = random_sample_sites(barrier_heights, n_samples)
    sampled_barrier_heights = barrier_heights[sampled_sites]
    sampled_local_coordinates = local_coordinates[sampled_sites]
    return sampled_sites, sampled_barrier_heights, sampled_local_coordinates

def biased_sample(model_barrier_heights, T):
    """
    Samples sites with replacement using biased probabilities (exp(-BE))
    
    input - 
    predicted_adsorption_energies - n_test x 1 numpy array of predicted adsorption energies
    T - catalyst operation temperature

    returns - 
    test site - ID of chosen site
    """
    B = 1000/8.314/T
    exp_E = np.exp(-B * model_barrier_heights) / np.sum(np.exp(-B * model_barrier_heights))
    return np.random.choice(np.arange(len(model_barrier_heights)), replace=True, p = exp_E)

def biased_error(adsorption_energies):
    """
    Calculates the standard error (68.2% CI) of the computed k-weighted average if sites are sampled using the biased
    distribution using t statistics. However, this doesn't take into account the error associated with the model.
    Inputs - 
    predicted_adsorption_energies - numpy array of adsoprtion energies sampled using the biased distribution


    returns - 
    biased sampling error - error at given confidence interval
    """
    from scipy import stats
    DoF = len(adsorption_energies) - 1
    # stats.t.ppf gives 1 sided z-value, z1. The 2-sided z-value, z2 = 0.5*(z1+1)
    return np.std(adsorption_energies)/np.sqrt(len(adsorption_energies)) * stats.t.ppf(0.5*(0.682 + 1), DoF)

def equivalent_random_sampling(barrier_distribution, T, relative_uncertainty):
    """
    Estimate how many random samples would be needed to estimate site average kinetics to same level of uncertainty. 
    Invokes CLT, N + 1 >> <exp(-2 B Ea)> / <exp(-B Ea)>**2
    Averages are estimated with the model predicted distribution. 
    NOTE: N samples will most likely be underestimated in the first few iterations

    inputs:
    barrier_distribution - Exact barrier distribution
    relative_uncertainty - standard error / exact <Ea>k
    """
    T = 300
    B = 1./(8.314/1000 * T)

    kEa = (len(barrier_distribution)**2 * np.var(np.exp(-B * barrier_distribution) * barrier_distribution)) / np.sum(np.exp(-B * barrier_distribution) * barrier_distribution)**2
    k = len(barrier_distribution)**2 * np.var(np.exp(-B * barrier_distribution)) / np.sum(np.exp(-B * barrier_distribution))**2
    
    return int(1. / relative_uncertainty * (kEa + k))

def train(X_train, Y_train, is_verbose=False):
    """
    Trains the metric learning model on the input training sample.
    Inputs - 
    X_train - nx4 numpy array of site parameters
    Y_train - nx4 numpy array of activation barriers of sites
    is_verbose - boolean, whether to print training progress (i.e. cost as a function of training steps)

    returns - 
    M - 4x4 numpy array of the Mahalanobis distance matrix
    model_predicted_barriers - barriers of sites predicted by the metric learning. 
                               The barriers of the training set are predicted by excluding itself
    residuals - numpy array of true barrier - predicted barrier 
    """
    mlkr = MLKR(verbose=is_verbose)
    fit_final = mlkr.fit(X_train, Y_train)
    S = fit_final.metric()
    model_predicted_barriers = []
    train_len = len(Y_train)
    Y_train =  [Decimal(Y_train[i]) for i in range(train_len)]
    for i in range(train_len):
        test = X_train[i]
        temp = np.delete(X_train, i, axis=0)
        delta = temp-test
        dist_mat = np.diagonal(delta @ S @ delta.T)
        dist_mat = np.transpose([Decimal(dist_mat[i]) for i in range(len(dist_mat))])
        k_mat = np.exp(-1*dist_mat)
        temp2 = np.delete(Y_train, i, axis=0)
        model_predicted_barriers.append(np.sum(np.multiply(temp2, k_mat)) / (np.sum(k_mat)))
    
    model_predicted_barriers  = np.asarray([float(i) for i in model_predicted_barriers])
    
    Y_train =  [float(i) for i in Y_train]
    residuals = Y_train - model_predicted_barriers 

    ss_res = np.sum(residuals**2)
    ss_tot = np.sum( (Y_train - np.mean(Y_train))**2 )
    R2 = 1. - ss_res / ss_tot

    return S, model_predicted_barriers, residuals, R2

def predicted_activation_energies(training_local_coordinates, training_barrier_heights, S, local_coordinates, n_sites):
    """
    Prediction of adsorption energy with the trained metric learning model
    Inputs - 
    training_NN_distances - n_train x 4 numpy array of NN distances of sites on which the metric learning model has been trained
    training_barrier_heights - n_train x 1 numpy array of true adsoprtion energies of sites on which the metric learning model has been trained
    M - 4x4 numpy array of the trained Mahalanobis distance
    NN_distances - n_test x 4 numpy array of NN distances of sites for which adsorption energy prediction has to be made
    lattice_size - float of size lattice

    returns - 
    predicted_energies - n_test x 1 numpy array of adsorption energies of the test set predicted by the model
    """
    # This seems sketchy AF but is not needed because the warning is resolved. 
    # If a site is really far from training data in distance space, the Gaussian kernels will all be 0 because of lack of precision
    # In the sum / sum line at the end of the for loop, you get 0/0 = NaN and a raised runtime warning.
    # The NaN cases are addressed in the next for loop, thus the runtime warning is not needed
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    predicted_energies = np.zeros(n_sites)
    training_size = len(training_barrier_heights)
    training_barrier_heights = np.longdouble(training_barrier_heights)

    for i in range(n_sites):
        delta = training_local_coordinates-local_coordinates[i]
        d = np.diagonal(delta @ S @ delta.T)
        k_mat = np.exp(-1 * d)
        predicted_energies[i] = np.sum(np.multiply(training_barrier_heights, k_mat)) / (np.sum(k_mat))

    # If any sites are too far from the training data they give NaN b/c of not enough precision
    # This happens b/c sum(training_barrier_heights @ k_mat)/sum(kmat) = 0/0 = NaN
      
    # Find NaN sites and rerun them with infinite precision via Decimal()
    # This isn't done for every site because it's slow AF
    bad_list = np.argwhere(np.isnan(predicted_energies)).ravel().tolist()
    if len(bad_list) > 0:
        training_decimal = [Decimal(float(training_barrier_heights[i])) for i in range(training_size)]
        
        for i in bad_list:
            delta = training_local_coordinates-local_coordinates[i]
            dist_mat = np.diagonal(np.matmul(np.matmul(delta, S), np.transpose(delta)))
            dist_mat = np.transpose([Decimal(dist_mat[i]) for i in range(len(dist_mat))])
            k_mat = np.exp(-1*dist_mat)
            predicted_energies[i] = np.sum(np.multiply(training_decimal, k_mat)) / (np.sum(k_mat))

    return predicted_energies

def active_site_counting(barrier_distribution, T, n_bins=50):
    hist, bin_edges = np.histogram(barrier_distribution, bins=n_bins, density=True)
    w_hist, w_bin_edges = np.histogram(barrier_distribution, bins=n_bins, density=True, weights=np.exp(-(barrier_distribution*1000)/(8.314*T)))

    dx = bin_edges[1] - bin_edges[0]
    w_dx = bin_edges[1] - bin_edges[0]

    fraction_sites = np.cumsum(hist) * dx
    apparent_activity = np.cumsum(w_hist) * w_dx

    return fraction_sites, apparent_activity 

def importance_learning(barrier_distribution, local_coordinates, T, initial_pool_size, n_iterations, plot_every=5, verbose=True):
    """
    Importance learning loop. 
    
    Inputs:
    barrier_distribution - Barrier height distribution, n x 1 np array
    NN_distances - n x 4 np array of ranked nearest neighbor distances,
    T - Temperature (K)
    initial_pool_size - int, how many sites to use to train the model before importance learning
    n_interations - int, number of importance learning iterations to perform
    verbose - boolean, whether to print and plot results
    plot_every - int, plot model fit and distribution of site estimates every multiple. Will not plot if verbose=False

    Returns: Dictionary of importance learning results
    """
    n_sites = len(barrier_distribution)
    IL = {}
    sampled_sites, sampled_barrier_heights, sampled_local_coordinates = inspect_sites(local_coordinates, barrier_distribution, initial_pool_size)
    random_samples = sampled_barrier_heights
    true_Ea_k = k_weighted_avg_activation_E(barrier_distribution, T)
    if verbose:
        print("###########################################")
        print("#### ENTERING IMPORTANCE LEARNING LOOP ####")
        print("###########################################")
        print('\n')
        print('Number of sites: {}'.format(n_sites))
        print('True <Ea>k: {:02.1f} kJ/mol'.format(true_Ea_k))
        print('\n')

    for i in range(n_iterations):
        # Excluding sites sampled more than once 
        indices_local_coords = np.unique(sampled_local_coordinates,return_index = True, axis = 0)[1]
        sampled_local_coords_unique  = np.asarray([sampled_local_coordinates[index] for index in sorted(indices_local_coords)])

        indices_sampled_barrier = np.unique(sampled_barrier_heights,return_index = True, axis = 0)[1]
        sampled_barrier_heights_unique = np.asarray([sampled_barrier_heights[index] for index in sorted(indices_sampled_barrier)])

        # MACHINE LEARNING
        S, model_barriers_LOO, residuals, R2 = train(sampled_local_coords_unique, sampled_barrier_heights_unique)

        # Get model predicted barrier heights for all sites
        model_barrier_heights = predicted_activation_energies(sampled_local_coords_unique, sampled_barrier_heights_unique, S, local_coordinates, n_sites) 

        # IMPORTANCE  SAMPLING
        test_site = biased_sample(model_barrier_heights, T)
        # Compute <Ea>k by averaging importance sampled barrier heights
        if i == 0:
            avg_Ea_IS = barrier_distribution[test_site]
            importance_sampled_sites = np.asarray([test_site])
            standard_error = 0.
            n_equivalent_random_samples = 0
            
        else:
            importance_sampled_sites = np.append(importance_sampled_sites, test_site)
            avg_Ea_IS = np.mean(barrier_distribution[importance_sampled_sites])
            standard_error = biased_error(barrier_distribution[importance_sampled_sites])
            n_equivalent_random_samples = equivalent_random_sampling(model_barrier_heights, T, standard_error/true_Ea_k)
        
        random_samples = np.append(random_samples, np.random.choice(barrier_distribution))

        if verbose:
            print("####### Iteration {} #######".format(i))
            print("Importance sampled site: {}".format(test_site))
            print("Model Predicted Energy: {:04.1f} kJ/mol".format(model_barrier_heights[test_site]))
            print("True Energy: {:04.1f} kJ/mol".format(barrier_distribution[test_site]))
            print("<Ea>k (Importance sampled average) = {:02.1f} +/- {:02.1f} kJ/mol".format(avg_Ea_IS, standard_error))
            print('\n')

            if i % plot_every == 0:
                plt.rcParams['figure.figsize'] = 5, 4

                # Show training results
                fig, axes = plt.subplots(1, 1, figsize=(7,6))
                plot_trained(axes, model_barriers_LOO, sampled_barrier_heights, initial_pool_size)
                plt.show()
                
                fig, axes = plt.subplots(1, 2, sharex=True, figsize=(14,6))
                histogram(axes[0], axes[1], model_barrier_heights, T, n_bins=20)
                plt.show()

        # Store info in dict
        IL[str(i)] = {
            "Sampled Sites" : sampled_sites.tolist(),
            "Sampled Barrier Heights" : sampled_barrier_heights.tolist(),
            "Model Coefficients" : S.tolist(),
            "R2" : R2,
            "Training predicted barrier" : [float(x) for x in model_barriers_LOO],
            "Next Site" : int(test_site),
            "Predicted Barrier" : model_barrier_heights[test_site],
            "True Barrier" : barrier_distribution[test_site],
            "<Ea>k model" : k_weighted_avg_activation_E(model_barrier_heights,T),
            "<Ea>k importance sampled" : avg_Ea_IS,
            "Standard Error" : standard_error,
            "Equivalent random samples" : n_equivalent_random_samples

        }
        # Append newly selected sites to sampled info
        sampled_sites = np.append(sampled_sites, test_site)
        sampled_barrier_heights = np.append(sampled_barrier_heights, barrier_distribution[test_site])
        sampled_local_coordinates = np.append(sampled_local_coordinates, [local_coordinates[test_site]], axis=0)
    return IL