import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Objective function
def objective(ct, cr, beff):
    return (ct + cr) / (2 * beff)

# Sample points
ct_vals = np.linspace(0.5, 10.0, 30)
cr_vals = np.linspace(0.5, 10.0, 30)
beff_vals = np.linspace(33.0, 80.0, 30)

CT, CR, BE = np.meshgrid(ct_vals, cr_vals, beff_vals, indexing='ij')
OBJ = objective(CT, CR, BE)

# Flatten and randomly subsample for visibility
pts = np.vstack([CT.ravel(), CR.ravel(), BE.ravel()]).T
obj_vals = OBJ.ravel()
idx = np.random.choice(len(obj_vals), size=2000, replace=False)

# 3D scatter with color as objective
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(pts[idx, 0], pts[idx, 1], pts[idx, 2], c=obj_vals[idx])
fig.colorbar(sc, label='Objective')
ax.set_xlabel('ct')
ax.set_ylabel('cr')
ax.set_zlabel('beff')
ax.set_title('4D visualization: ct, cr, beff (axes) and objective (color)')

# 2D contour slices at fixed beff values
slice_values = [33.0, 50.0, 80.0]
fig2, axs = plt.subplots(1, len(slice_values), figsize=(15, 5))
for ax, b in zip(axs, slice_values):
    CT2, CR2 = np.meshgrid(ct_vals, cr_vals)
    Z2 = objective(CT2, CR2, b)
    cf = ax.contour(CT2, CR2, Z2, levels=15)
    ax.clabel(cf, inline=True, fontsize=8)
    ax.set_title(f'beff = {b}')
    ax.set_xlabel('ct')
    ax.set_ylabel('cr')
fig2.suptitle('Objective contours at fixed beff slices')

plt.tight_layout()
plt.show()
