import numpy as np
from scipy.optimize import minimize

def morse_potential(r, r_eq, D, a):
    """
    Get activation energy of a site based on local environment.
    Based off Morse potential, V(r) = D(1 - exp(-a(r-r_eq)))
    inputs -
    r -  4 x 1 np array of distances to nearest neighbors on the lattice
    r_eq - float of non-dimensional "equilibrium" bond length between lattice sites 
    D - well depth
    a - force constant like value
    """
    return D*(1-np.exp(-a*(r-r_eq)))**2 - D

def modified_morse(r4, r_eq_M_O, D_M_O, a_M_O):
    """
    Returns the modified morse potential between metal and adsorbate
    Based off Morse potential, V(r) =  D for r < r_eq
                                    = -D(1 - exp(-a(r-r_eq)))+D for r > r_eq
    inputs -
    r -  distance between metal and siloxane oxygen
    r_eq - float of non-dimensional "equilibrium" bond length between metal and siloxane oxygen 
    D - well depth
    a - force constant like value
    """
    if r4 < r_eq_M_O:
        return D_M_O
    else:
        return -morse_potential(r4, r_eq_M_O, D_M_O, a_M_O)


def metal_dist(x1, x2, S):
    """
    Calculates distance between metal and hydroxyl oxygen/siloxane oxygen
    Inputs-
    x1 - x coordinate of metal
    x2 - y coordinate of metal
    S - coordinates of hydroxyl oxygen/siloxane oxygen

    Returns -   
    distance between metal and siloxane

    """
    xm = x1
    ym = x2
    
    xi = S[0]
    yi = S[1]
    return np.sqrt((xm - xi)**2 + (ym - yi)**2)

def bare_metal_pot(M, morse_MO, morse_M_O, S1, S2, S3, S4):
    """
    Potential for the bare metal site, this includes both the hydroxyl oxygens and both the siloxanes
    Inputs -
        M - coordinates of metal
        morse_MO: Dictionary containing
            {
                D : metal-oxygen coordination strength
                r_eq :  metal-oxygen Equilibrium distance
                a : `a` parameter for metal-oxygen bond
            }
        morse_M_O: Dictionary containing
            {
                D : metal-siloxane coordination strength
                r_eq :  metal-siloxane Equilibrium distance
                a : `a` parameter for metal-siloxane coordination
            }
        S1 - 1st hydroxyl oxygen coordinates
        S2 - 2nd hydroxyl oxygen coordinates
        S3 - 1st siloxane oxygen coordinates
        S4 - 2nd siloxane oxygen coordinates
                
        Returns-
        Potential energy of metal
    """

    x1 = M[0]
    x2 = M[1]

    # Calculate distance of metal from S1, S2, S3, and S4
    r1 = metal_dist(x1, x2, S1)
    r2 = metal_dist(x1, x2, S2)
    r3 = metal_dist(x1, x2, S3)
    r4 = metal_dist(x1, x2, S4)

    return morse_potential(r1, morse_MO['r_eq'],  morse_MO['D'],  morse_MO['a'])  + \
           morse_potential(r2, morse_M_O['r_eq'], morse_M_O['D'], morse_M_O['a']) + \
           morse_potential(r3, morse_MO['r_eq'],  morse_MO['D'],  morse_MO['a'])  + \
           morse_potential(r4, morse_M_O['r_eq'], morse_M_O['D'], morse_M_O['a'])

def ads_state_pot(M, morse_MO, morse_M_O, S1, S2, S3, S4):
    """
    Potential for the gas adsorbed on metal, this includes both the hydroxyl oxygens and one of the siloxanes (the one closer to the metal in the optimized bare metal site)
    Inputs -
        M - coordinates of metal
        morse_MO: Dictionary containing
            {
                D : metal-oxygen coordination strength
                r_eq :  metal-oxygen Equilibrium distance
                a : `a` parameter for metal-oxygen bond
            }
        morse_M_O: Dictionary containing
            {
                D : metal-siloxane coordination strength
                r_eq :  metal-siloxane Equilibrium distance
                a : `a` parameter for metal-siloxane coordination
            }
        S1 - 1st hydroxyl oxygen coordinates
        S2 - siloxane oxygen coordinates (which doesn't unbond)
        S3 - 2nd hydroxyl oxygen coordinates
        S4 - siloxane oxygen coordinates (which unbonds)
                
        Returns-
        Potential of adsorbed site
    """
    x1 = M[0]
    x2 = M[1]

    r1 = metal_dist(x1, x2, S1)
    r2 = metal_dist(x1, x2, S2)
    r3 = metal_dist(x1, x2, S3)
    r4 = metal_dist(x1, x2, S4)

    return morse_potential(r1, morse_MO['r_eq'],  morse_MO['D'],  morse_MO['a'])  + \
           morse_potential(r2, morse_M_O['r_eq'], morse_M_O['D'], morse_M_O['a']) + \
           morse_potential(r3, morse_MO['r_eq'],  morse_MO['D'],  morse_MO['a'])  + \
           morse_potential(r4, morse_M_O['r_eq'], morse_M_O['D'], morse_M_O['a']) + \
           modified_morse( r4, morse_M_O['r_eq'], morse_M_O['D'], morse_M_O['a'])

def opt_bare_metal(morse_MO, morse_M_O, S1, S2, S3, S4):
    """
    Optimize the bare metal site
    Inputs -
        morse_MO: Dictionary containing
            {
                D : metal-oxygen coordination strength
                r_eq :  metal-oxygen Equilibrium distance
                a : `a` parameter for metal-oxygen bond
            }
        morse_M_O: Dictionary containing
            {
                D : metal-siloxane coordination strength
                r_eq :  metal-siloxane Equilibrium distance
                a : `a` parameter for metal-siloxane coordination
            }
        S1 - 1st hydroxyl oxygen coordinates
        S2 - 2nd hydroxyl oxygen coordinates
        S3 - 1st siloxane oxygen coordinates
        S4 - 2nd siloxane oxygen coordinates
                
        Returns-
        Optimized potential of bare metal site
    """
    # Generate initial guess for metal position
    M = (S1 + S2 + S3 + S4) / 4
    return minimize(bare_metal_pot, M, method='Nelder-Mead',
                    args=(morse_MO, morse_M_O, S1, S2, S3, S4))

def opt_ads_state(morse_MO, morse_M_O, S1, S2, S3, S4):
    """
    Optimize the adsorbed site
    Inputs -
        morse_MO: Dictionary containing
            {
                D : metal-oxygen coordination strength
                r_eq :  metal-oxygen Equilibrium distance
                a : `a` parameter for metal-oxygen bond
            }
        morse_M_O: Dictionary containing
            {
                D : metal-siloxane coordination strength
                r_eq :  metal-siloxane Equilibrium distance
                a : `a` parameter for metal-siloxane coordination
            }
        S1 - 1st hydroxyl oxygen coordinates
        S2 - siloxane which doens't unbond
        S3 - 2nd hydroxyl oxygen coordinates
        S4 - siloxane which unbonds
                
        Returns-
        Optimized potential of adsorbed site
    """
    # Generate initial guess for metal position
    M = (S1 + S2 + S3 + S4) / 4
    return minimize(ads_state_pot, M, method='Nelder-Mead', 
                    args=(morse_MO, morse_M_O, S1, S2, S3, S4))

def optimize_site(morse_MO, morse_M_O, E_MA, T, graftable_sites, decorated_lattice):
    """
    Calculates the grafting and adsorption energy of a site by optimizing the metal position
    Inputs -
        morse_MO: Dictionary containing
            {
                D : metal-oxygen coordination strength, kJ/mol
                r_eq :  metal-oxygen Equilibrium distance, dimnless
                a : `a` parameter for metal-oxygen bond, dimnless
            }
        morse_M_O: Dictionary containing
            {
                D : metal-siloxane coordination strength, kJ/mol
                r_eq :  metal-siloxane Equilibrium distance, dimnless
                a : `a` parameter for metal-siloxane coordination, dimnless
            }
        E_MA - metal-adsorbate bond strength, kJ/mol
        T - temperature, K
        graftable_sites - Indices of graftable sites as a Nx2 numpy array
        lattice - Tensor of coordinates of sites and type NxNx3
        decorated_lattice - Tensor of lattice with siloxanes, hydroxyls, and graftable sites
        
        Returns-
        graft_E - grafting energy
        ads_E - adsorption energy of gas to metal
    """

    # delta_H for a unperturbed site = 0 = 2E_MO + 2E_M_O + 2E_HL - 2E_ML - 2E_OH, and C = 2E_HL - 2E_ML - 2E_OH
    C = (2 * morse_MO['D'] + 2 * morse_M_O['D'])
    S10 = np.asarray([1,0])
    S20 = np.asarray([0,1])
    S30 = np.asarray([-1,0])
    S40 = np.asarray([0,-1])
    
    # delta_H for a unperturbed site = 70 = 2E_MO + 2E_M_O + 2E_HL - 2E_ML - 2E_OH, and C = 2E_HL - 2E_ML - 2E_OH
    unperturbed_opt = opt_bare_metal(morse_MO, morse_M_O,S10,S30,S20,S40)

    # ADD Translational free energy for a molecule with the weight of HCl
    h = 6.626*(10**(-34))
    kb = 1.38*(10**(-23))
    P = 101325
    R = 8.314/1000.
    T = 300
    m = 6.0536112*(10**(-26))
    m2 = 29./1000/6.02e23
    lam = h/(np.sqrt(2*(np.pi)*m*kb*T))
    lam2 = h/(np.sqrt(2*(np.pi)*m2*kb*T))
    q1 = (kb*T/P)/(lam**3)
    q2 = (kb*T/P)/(lam2**3)
    TS_HL = -((R*T))*np.log(q1)
    TS_ML2 = -((R*T))*np.log(q2)
    C = 70 - (unperturbed_opt.fun + 2*TS_HL  - TS_ML2)

    opt_graft_E = []
    opt_ads_E = []

    for i in range(np.shape(graftable_sites)[0]):
        # Set S1, and S3 to hydroxyl group positions and S2, and S4 to siloxane positions
        if decorated_lattice[graftable_sites[i][0]+1][graftable_sites[i][1]][2] == 1:
            S1 = decorated_lattice[graftable_sites[i][0]+1][graftable_sites[i][1]][0:2]
            S2 = decorated_lattice[graftable_sites[i][0]][graftable_sites[i][1]+1][0:2]
            S3 = decorated_lattice[graftable_sites[i][0]-1][graftable_sites[i][1]][0:2]
            S4 = decorated_lattice[graftable_sites[i][0]][graftable_sites[i][1]-1][0:2]

        # Rotate assignments if siloxanes are on the x-axis and hydroxyls are on the y axis
        else:
            S1 = decorated_lattice[graftable_sites[i][0]][graftable_sites[i][1]+1][0:2]
            S2 = decorated_lattice[graftable_sites[i][0]-1][graftable_sites[i][1]][0:2]
            S3 = decorated_lattice[graftable_sites[i][0]][graftable_sites[i][1]-1][0:2]
            S4 = decorated_lattice[graftable_sites[i][0]+1][graftable_sites[i][1]][0:2]

        # optimize the bare metal site
        bare_metal = opt_bare_metal(morse_MO, morse_M_O, S1, S3, S2, S4)
        opt_graft_E.append(bare_metal.fun)
        
        # compute the distance between the optimized metal site and the the siloxanes and get the index of the siloxane with the larger distance
        index_del = np.argmax((metal_dist(bare_metal.x[0], bare_metal.x[1], S2),
                               metal_dist(bare_metal.x[0], bare_metal.x[1], S4)))

        if index_del == 0 :
            #delete S2 and return opt energy
            ads_state = opt_ads_state(morse_MO, morse_M_O, S1, S4, S3, S2)
            opt_ads_E.append(ads_state.fun)

        else :
            #delete S4 and return opt energy
            ads_state = opt_ads_state(morse_MO, morse_M_O, S1, S2, S3, S4)
            opt_ads_E.append(ads_state.fun)
        
        if not ads_state.success:
            print("WARNING! Site {} failed".format(i))
    grafting_G = C + np.asarray(opt_graft_E) + 2*TS_HL  - TS_ML2

    ### Adsorption energies ###
    adsorption_H = (np.asarray(opt_ads_E) - E_MA) - (np.asarray(opt_graft_E) + R*T)
    # G = H - TS
    adsorption_G = adsorption_H - TS_ML2

    return grafting_G, adsorption_H, adsorption_G



