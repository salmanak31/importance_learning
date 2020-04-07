from potentials import *
import numpy as np
import matplotlib.pyplot as plt

# git_path = "C:/Program Files/Git/git-bash.exe"
pot_path = "C:/Users/Salman/Desktop/Research/il-pedagogical/logs/2/potentials.py"

print(np.random.random(1) - 0.25)
S1 = np.asarray([1+ 0.5*np.random.random(1) - 0.25,0.5*np.random.random(1) - 0.25])
S2 = np.asarray([0.5*np.random.random(1) - 0.25,1+0.5*np.random.random(1) - 0.25])
S3 = np.asarray([-1 + 0.5*np.random.random(1) - 0.25,0.5*np.random.random(1) - 0.25])
S4 = np.asarray([0.5*np.random.random(1) - 0.25,-1+0.5*np.random.random(1) - 0.25])


S1 = np.asarray([1,0])
S2 = np.asarray([0,1])
S3 = np.asarray([-1 ,0])
S4 = np.asarray([0,-1])

# S1 = np.asarray([1155.12689812,])

M = np.asarray([0,0])

D_MO = 350
r_MO = 1.
a_MO = 1.7


D_M_O = 300
r_M_O = 1.
a_M_O = 1.3

a = opt_bare_metal(D_MO,r_MO,a_MO,D_M_O,r_M_O,a_M_O,S1,S2,S3,S4)

x = np.linspace(-2, 2, 200)
y = np.linspace(-2, 2, 200)

mesh_g = np.meshgrid(x,y)


lower = -1
upper = 1
print(S1,S2,S3,S4)

plt.contourf(mesh_g[0],mesh_g[1],bare_metal_pot(mesh_g,D_MO,r_MO,a_MO,D_M_O,r_M_O,a_M_O,S1,S2,S3,S4),500)
plt.xlim([-0.5,0.5])
plt.ylim([-0.5,0.5])
plt.colorbar()
plt.scatter(a.x[0],a.x[1])


plt.show()

# b = Final(S1,S2,S3,S4,M)

# plt.contourf(mesh_g[0],mesh_g[1],final_pot(mesh_g,S1,S3,S4),15)
# plt.colorbar()
# plt.scatter(b.x[0],b.x[1])


# plt.show()

# print(a.x)
# print(b.x)