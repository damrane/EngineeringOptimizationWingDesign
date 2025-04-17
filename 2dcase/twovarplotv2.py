import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Objective function
def objective(cr, beff, ct=1):
    return (ct + cr) / (2 * beff)

# Create grid for ct and cr
cr_vals = np.linspace(0.5, 10.0, 100)
beff_vals = np.linspace(33, 80, 100)
Beff, CR = np.meshgrid(beff_vals, cr_vals)
Z = objective(CR, Beff)

# 3D surface plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(Beff, CR, Z, edgecolor='none')
ax.set_xlabel('Beff')
ax.set_ylabel('cr')
ax.set_zlabel('Objective')
ax.set_title('3D Surface of Objective vs. Beff and cr')

# Contour plot
fig2, ax2 = plt.subplots()
contours = ax2.contour(Beff, CR, Z, levels=15)
ax2.clabel(contours, inline=True, fontsize=8)
ax2.set_xlabel('Beff')
ax2.set_ylabel('cr')
ax2.set_title('Contour Plot of Objective vs. Beff and cr')

plt.show()
