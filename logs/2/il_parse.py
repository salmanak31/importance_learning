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

    def T(self):
        return self.IL_dict['parameters']['T']

    def displacement_type(self):
        return self.IL_dict['parameters']['displacement-type']

    def dissociation_type(self):
        return self.IL_dict['parameters']['dissociation-type']

    def potential_type(self):
        return self.IL_dict['parameters']['potential-type']

    def NN_distances(self):
        """
        Return nearest neighbor distances as 4 x (l)
        """
        return np.asarray(self.IL_dict['NN-distances'])

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

    def sampled_NN_distances(self, iteration):
        """
        Ranked nearest neighbor distances of sampled sites for an iteration of the importance learning loop
        input: 
        iteration: int, iteration of the importance learning loop

        returns:
        np array of nearest neighbor distances
        """
        return np.asarray(self.NN_distances()[self.sampled_sites(iteration)])

    def model_coefficients(self, iteration):
        """
        Optimized matrix for metric learning model for an iteration of the importance learning loop
        input: 
        iteration: int, iteration of the importance learning loop

        returns:
        np array model coefficients
        """
        return np.asarray(self.IL_dict['importance-learning'][str(iteration)]['Model Coefficients'])

    def model_barrier_heights(self, iteration):
        """
        Model predicted barrier heights of all sites for a given iteration as a numpy array
        """
        return ilp.predicted_adsorption_energies(self.sampled_NN_distances(iteration), 
                                          self.sampled_barrier_heights(iteration), 
                                          self.model_coefficients(iteration), 
                                          self.NN_distances(), self.lattice_length())

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
        return np.asarray([self.IL_dict['importance-learning'][str(i)]['Sampling Error'] for i in range(self.n_iterations)])
