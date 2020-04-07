# from grafting import *
from il import *
from lattice import *
from potentials import *
from visualization import *

if __name__ == "__main__":
    # Standard libraries
    import json

    # pip imports
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    # non-pip
    from visualization import histogram
    from lattice import load_lattice

    from il import importance_learning, k_weighted_avg_activation_E, active_site_counting

    output = 'C:\\Users\\Craig\\Desktop\\repos\\il-pedagogical\\test.json'
    initial_pool_size = 50

    ql_dict = load_lattice('quenched_lattice.json')

    barrier_distribution = np.asarray(ql_dict['rxn-barrier-heights'])
    local_coordinates = np.asarray(ql_dict['local-coordinates'])
    T = ql_dict['parameters']['T']

    local_coordinates_dict = {'OH-OH-distance' : local_coordinates[:, 0],
                              'siloxane-distances' : local_coordinates[:, 1],
                              'OH-siloxane-angle' : local_coordinates[:, 2], 
                              'OH-siloxane-midpoint' : local_coordinates[:, 3]}


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.set_xlabel(r'$E_a$, kJ/mol')
    ax2.set_xlabel(r'$E_a$, kJ/mol')
    histogram(ax1, ax2, barrier_distribution, T, n_bins=20)
    plt.show()
  
    site_avg_E = k_weighted_avg_activation_E(barrier_distribution, T) 
    fraction_sites, apparent_activity_fraction = active_site_counting(barrier_distribution, T)
    # local_coordinates = np.delete(local_coordinates, (0,3), axis=1)

    IL = importance_learning(barrier_distribution, local_coordinates, T, initial_pool_size, 50, plot_every=25)
    
    # Export stuff
    results = {}
    ql_dict.update({'importance-learning' : IL})
    ql_dict.update({'True <Ea>k' : site_avg_E})

    a = json.dumps(ql_dict, ensure_ascii=False, indent=2)
    with open(output, 'w') as outfile:
        outfile.write(a)    

