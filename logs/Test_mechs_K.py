import matplotlib.pyplot as plt
import numpy as np

def f(K1):
    Pa = 1
    k3 = 1
    K2 = 1
    return k3*K2*K1*Pa/(K1*Pa+K1*K2*Pa + 1)

def f2(E):
    Pa = 1
    Pb = 0.5
    K3 = 1
    k2 = 1
    return (k2*np.exp(-E)*Pa*(1-Pb/K3))/(1+(np.exp(-E))*Pa)


E = np.linspace(0,100,10000)


# plt.plot(K1,f(K1))

plt.plot(E,f2(E))

plt.show()