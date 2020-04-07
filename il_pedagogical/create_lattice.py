"""
Script to create a quenched disordered lattice 
"""

# Standard libraries
import json

# pip imports
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# non-pip
from potentials import optimize_site
from visualization import histogram
from lattice import make_quenched_disorder_lattice, decorate_lattice, locate_grafting_sites, compute_local_coordinates
from il import importance_learning, k_weighted_avg_activation_E, active_site_counting

qd_lattice_fname = 'C:\\Users\\Craig\\Desktop\\repos\\il-pedagogical\\QD_lattice.json'

lattice_size = 1500

T = 300
cov = 0.00022

displacement_type = 'normal'
disp_kwargs = {'covariance' : [[cov, 0], [0, cov]]}

D_MO = 500.
r_eq_MO = 1.
a_MO = 1.9

D_M_O = 120.
r_eq_M_O = 1.16
a_M_O = 2.3

E_MA = 155.

H_ddagger = 65

empty_fraction = 0.3
OH_fraction = 0.3
siloxane_fraction = 0.4

MO_Morse = {'D' : D_MO, 'a' : a_MO, 'r_eq' : r_eq_MO}
siloxane_Morse = {'D' : D_M_O, 'a' : a_M_O, 'r_eq' : r_eq_M_O}
lattice_fractions = {'empty' : empty_fraction, 'OH' : OH_fraction, 'Siloxane' : siloxane_fraction}

lattice = make_quenched_disorder_lattice(lattice_size, cov, True, False)
decorated_lattice = decorate_lattice(lattice, empty_fraction, OH_fraction, siloxane_fraction)

graftable_sites, competing_sites = locate_grafting_sites(decorated_lattice)
# plot_lattice(decorated_lattice, True, graftable_sites)

local_coordinates = compute_local_coordinates(decorated_lattice, graftable_sites)
local_coordinates_dict = {'OH-OH-distance' : local_coordinates[:, 0],
                            'siloxane-distances' : local_coordinates[:, 1],
                            'OH-siloxane-angle' : local_coordinates[:, 2], 
                            'OH-siloxane-midpoint' : local_coordinates[:, 3]}

# local_coordinates = np.delete(local_coordinates, (0, 3), axis=1)
graft_E, ads_H, ads_G = optimize_site(MO_Morse, siloxane_Morse, E_MA, T, graftable_sites, decorated_lattice)

barrier_distribution = ads_H + H_ddagger + (2 * 8.314 * T)/1000

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
ax1.set_xlabel(r'$E_a$, kJ/mol')
ax2.set_xlabel(r'$E_a$, kJ/mol')
histogram(ax1, ax2, barrier_distribution, T, n_bins=20)
plt.show()

parameters = {
    'lattice-length' : lattice_size,
    'displacements' : {'displacement-type' : 'Normal',
                        'displacement-kwargs' : disp_kwargs},
    'site-fractions' : lattice_fractions,
    'potentials' : {'MO-potential' : MO_Morse,
                    'siloxane-potential': siloxane_Morse,
                    'metal-adsorbate-bond' : E_MA},
    'number-graftable-sites' : len(barrier_distribution),
    'number-competing-sites' : len(competing_sites),
    'T' : T,

}

# Export stuff
qd_lattice = {
    'parameters' : parameters,
    'local-coordinates' : local_coordinates.tolist(),
    'grafting-barrier-heights' : graft_E.tolist(),
    'rxn-barrier-heights' : barrier_distribution.tolist()
}

a = json.dumps(qd_lattice, ensure_ascii=False, indent=2)
with open(qd_lattice_fname, 'w') as outfile:
    outfile.write(a)  