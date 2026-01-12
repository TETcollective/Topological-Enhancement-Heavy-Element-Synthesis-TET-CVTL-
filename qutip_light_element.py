"""
TET--CVTL: QuTiP simulation for light-element fusion enhancement
Simon Soliman, TET Collective, January 2026
License: CC BY-NC-ND 4.0
"""

import qutip as qt
import numpy as np
import matplotlib.pyplot as plt

# Primordial trefoil phase
theta = 6 * np.pi / 5

# Light-element proxy (low Z barrier for baseline comparison)
Z_eff = 3.0  # Effective for light fusion (e.g., p-Li or similar)

# Base Hamiltonian
H0 = Z_eff * qt.tensor(qt.sigmax(), qt.sigmax())

# Anyonic catalysis
phase = np.exp(1j * theta)
phase_op = qt.tensor(qt.qeye(2), qt.qdiags([1.0, phase], 0))

H_eff = H0 + phase_op

# Initial state
psi0 = (qt.tensor(qt.basis(2,0), qt.basis(2,1)) + 
        qt.tensor(qt.basis(2,1), qt.basis(2,0))).unit()

# Fused state proxy
fused = qt.tensor(qt.basis(2,0), qt.basis(2,0))

times = np.linspace(0, 20, 600)

result_with = qt.mesolve(H_eff, psi0, times)
overlap_with = [abs(fused.overlap(state))**2 for state in result_with.states]

result_without = qt.mesolve(H0, psi0, times)
overlap_without = [abs(fused.overlap(state))**2 for state in result_without.states]

enhancement = np.max(overlap_with) / np.max(overlap_without)
print(f"Light-element fusion enhancement factor: {enhancement:.1f}x")

plt.figure(figsize=(10,6))
plt.plot(times, overlap_with, label=f'With trefoil catalysis (enhancement {enhancement:.1f}x)', color='cyan', linewidth=3)
plt.plot(times, overlap_without, '--', label='Standard light-element barrier', color='gray', linewidth=2.5)
plt.title('TET--CVTL Enhancement of Light-Element Fusion (Proxy System)')
plt.xlabel('Time (arbitrary units)')
plt.ylabel('Fusion channel overlap probability')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('light_element_fusion_enhancement.pdf')
plt.savefig('light_element_fusion_enhancement.png', dpi=300)
print("Figure saved: light_element_fusion_enhancement.pdf / .png")
