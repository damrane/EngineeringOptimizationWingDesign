import numpy as np
import matplotlib.pyplot as plt

# Objective function
def objective(ct, cr, beff):
    return (ct + cr) / (2 * beff)

def sensitivity_analysis():
    # Define base values
    base_ct = 0.5
    base_cr = 0.5
    base_beff = 33.0
    
    # Define ranges with small steps
    delta = 1e-5
    ct_range = np.arange(0.5, 10, delta)
    cr_range = np.arange(0.5, 20, delta)
    beff_range = np.arange(33, 80, delta)
    
    # Calculate objective values - each calculation isolates one variable
    ct_objectives = [objective(ct, base_cr, base_beff) for ct in ct_range]
    cr_objectives = [objective(base_ct, cr, base_beff) for cr in cr_range]
    beff_objectives = [objective(base_ct, base_cr, beff) for beff in beff_range]
    
    # Create subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot for ct (cr and beff constant)
    axs[0].plot(ct_range, ct_objectives)
    axs[0].set_xlabel('ct')
    axs[0].set_ylabel('Objective Value')
    axs[0].set_title(f'Sensitivity to ct (cr={base_cr}, beff={base_beff})')
    
    # Plot for cr (ct and beff constant)
    axs[1].plot(cr_range, cr_objectives)
    axs[1].set_xlabel('cr')
    axs[1].set_ylabel('Objective Value')
    axs[1].set_title(f"Sensitivity to cr (ct={base_ct}, beff={base_beff})")
    
    # Plot for beff (ct and cr constant)
    axs[2].plot(beff_range, beff_objectives)
    axs[2].set_xlabel('beff')
    axs[2].set_ylabel('Objective Value')
    axs[2].set_title(f"Sensitivity to beff (ct={base_ct}, cr={base_cr})")
    
    plt.tight_layout()
    plt.show()

# Run the sensitivity analysis
sensitivity_analysis()