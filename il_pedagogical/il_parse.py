import json
import numpy as np
import il_pedagogical as ilp

class Parser():
    """
    Parse results from an importance learning run.

    Inputs:
    file_name: str, path of file, .json extension included
    """
    def __init__(self, file_name):
        with open(file_name) as IL_json:
            self.IL_dict = json.load(IL_json)
        self.n_iterations = len(self.IL_dict['importance-learning'])

        pass

    def parameters(self):
        return self.IL_dict['parameters']

    def lattice_length(self):
        """
        Number of sites along the edge of the square lattice. Includes corner points
        """
        return self.IL_dict['parameters']['lattice-length']

    def number_graftable_sites(self):
        return self.IL_dict['parameters']['number-graftable-sites']

    def T(self):
        return self.IL_dict['parameters']['T']

    def displacement_type(self):
        return self.IL_dict['parameters']['displacement-type']

    def displacement_parameters(self):
        return self.IL_dict['parameters']['displacement-kwargs']

    def potential_values(self, potential_type=None):

        if potential_type == None:
            return self.IL_dict['parameters']['potentials']
        else:
            return self.IL_dict['parameters'][str(potential_type)]
        
    def local_coordinates(self):
        """
        Return nearest neighbor distances as 4 x (l)
        """
        return np.asarray(self.IL_dict['local-coordinates'])

    def true_barrier_heights(self):
        """
        Returns the exact (non-model predicted) barrier heights for a lattice
        """
        return np.asarray(self.IL_dict['barrier-heights'])
    
    def true_site_avg_Ea(self):
        """
        Returns <Ea>k averaged over the true barrier height of all sites on the lattice
        """
        return self.IL_dict['True <Ea>k']

    def sampled_sites(self, iteration):
        """
        Sampled sites for an iteration of the importance learning loop
        input: 
        iteration: int, iteration of the importance learning loop

        returns:
        np array of sampled sites
        """
        return np.asarray(self.IL_dict['importance-learning'][str(iteration)]['Sampled Sites'])

    def sampled_barrier_heights(self, iteration):
        """
        True barrier heights of sampled sites for an iteration of the importance learning loop
        input: 
        iteration: int, iteration of the importance learning loop

        returns:
        np array of sampled sites
        """
        return np.asarray(self.IL_dict['importance-learning'][str(iteration)]['Sampled Barrier Heights'])

    def sampled_local_coordinates(self, iteration):
        """
        Ranked nearest neighbor distances of sampled sites for an iteration of the importance learning loop
        input: 
        iteration: int, iteration of the importance learning loop

        returns:
        np array of nearest neighbor distances
        """
        return np.asarray(self.local_coordinates()[self.sampled_sites(iteration)])
    
    def importance_sampled_barrier_heights(self, iteration):
        ini_pool = self.IL_dict['parameters']['initial-pool-size']
        return np.asarray(self.sampled_barrier_heights(iteration)[ini_pool:].tolist() + [self.IL_dict['importance-learning'][str(iteration)]['True Barrier']])

    def importance_sampled_local_coordinates(self, iteration):
        ini_pool = self.IL_dict['parameters']['initial-pool-size']
        next_site = self.IL_dict['importance-learning'][str(iteration)]['Next Site']
        return np.asarray(self.sampled_local_coordinates(iteration)[ini_pool:].tolist()+ [self.IL_dict['local-coordinates'][next_site]])

    def model_coefficients(self, iteration):
        """
        Optimized matrix for metric learning model for an iteration of the importance learning loop
        input: 
        iteration: int, iteration of the importance learning loop

        returns:
        np array model coefficients
        """
        return np.asarray(self.IL_dict['importance-learning'][str(iteration)]['Model Coefficients'])

    def R2(self, iteration):
        """
        Return R**2 value of model at a given iteration
        """
        return self.IL_dict['importance-learning'][str(iteration)]['R2']

    def model_barrier_heights(self, iteration):
        """
        Model predicted barrier heights of all sites for a given iteration as a numpy array
        """
        return ilp.predicted_activation_energies(self.sampled_local_coordinates(iteration), 
                                          self.sampled_barrier_heights(iteration), 
                                          self.model_coefficients(iteration), 
                                          self.local_coordinates(), self.number_graftable_sites())

    def model_training_barrier_heights(self, iteration):
        """
        Returns model predicted barrier heights from the training set, predicted by leave one out
        """
        return np.asarray(self.IL_dict['importance-learning'][str(iteration)]['Training predicted barrier'])

    def all_site_avg_Ea(self):
        """
        Return <Ea>k as np array for all iterations of importance learning loop
        """
        return np.asarray([self.IL_dict['importance-learning'][str(i)]['<Ea>k importance sampled'] for i in range(self.n_iterations)])

    def all_site_Ea_sampling_error(self):
        """
        Return sampling error as np array for all iterations of importance learning loop
        """
        return np.asarray([self.IL_dict['importance-learning'][str(i)]['Standard Error'] for i in range(self.n_iterations)])

    def standard_error_to_CI(self, confidence_lvl):
        """
        Converts standard sampling error (68% CI) to a different CI
        """
        from scipy import stats
        confidence_int = np.zeros((self.n_iterations, 1))
        # Iteration 0 = initial pool
        # Iteration 1 = 1 data points = no standard deviation
        for i in range(1, self.n_iterations):
            barrier_distribution = self.sampled_barrier_heights(i)[self.IL_dict['parameters']['initial-pool-size']:]
            DoF = len(barrier_distribution) - 1
            # stats.t.ppf gives 1 sided z-value, z1. The 2-sided z-value, z2 = 0.5*(z1+1)
            confidence_int[i] = np.std(barrier_distribution)/np.sqrt(len(barrier_distribution)) * stats.t.ppf(0.5*(confidence_lvl + 1), DoF)
        return confidence_int

    def N_equiv_sampling(self):
        """
        Return the equivalent number of random samples needed to estimate <Ea>k with the same accuracy as importance sampling
        """
        return np.asarray([self.IL_dict['importance-learning'][str(i)]["Equivalent random samples"] for i in range(self.n_iterations)])