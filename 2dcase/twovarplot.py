import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Objective function
def objective(ct, cr, beff=50):
    return (ct + cr) / (2 * beff)

# Create grid for ct and cr
ct_vals = np.linspace(0.5, 10.0, 100)
cr_vals = np.linspace(0.5, 10.0, 100)
CT, CR = np.meshgrid(ct_vals, cr_vals)
Z = objective(CT, CR)

# 3D surface plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(CT, CR, Z, edgecolor='none')
ax.set_xlabel('ct')
ax.set_ylabel('cr')
ax.set_zlabel('Objective')
ax.set_title('3D Surface of Objective vs. ct and cr')

# Contour plot
fig2, ax2 = plt.subplots()
contours = ax2.contour(CT, CR, Z, levels=15)
ax2.clabel(contours, inline=True, fontsize=8)
ax2.set_xlabel('ct')
ax2.set_ylabel('cr')
ax2.set_title('Contour Plot of Objective vs. ct and cr')

plt.show()
