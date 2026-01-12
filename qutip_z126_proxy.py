"""
TET--CVTL: QuTiP simulation for Z=126 island of stability proxy
Simon Soliman, TET Collective, January 2026
License: CC BY-NC-ND 4.0
"""

import qutip as qt
import numpy as np
import matplotlib.pyplot as plt

# Primordial trefoil phase
theta = 6 * np.pi / 5

# Island of stability proxy (Z=126)
Z_eff = 126.0

# Base Hamiltonian with extreme barrier
H0 = Z_eff * qt.tensor(qt.sigmax(), qt.sigmax())

# Correlated anyonic catalysis with collective scaling
phase = np.exp(1j * theta * np.sqrt(Z_eff))  # Multi-knot collective effect
phase_op = qt.tensor(qt.qeye(2), qt.qdiags([1.0, phase], 0))

H_eff = H0 + phase_op

# Initial state
psi0 = (qt.tensor(qt.basis(2,0), qt.basis(2,1)) + 
        qt.tensor(qt.basis(2,1), qt.basis(2,0))).unit()

# Fused state proxy
fused = qt.tensor(qt.basis(2,0), qt.basis(2,0))

times = np.linspace(0, 6, 400)  # Shorter time due to stronger barrier

result_with = qt.mesolve(H_eff, psi0, times)
overlap_with = [abs(fused.overlap(state))**2 for state in result_with.states]

result_without = qt.mesolve(H0, psi0, times)
overlap_without = [abs(fused.overlap(state))**2 for state in result_without.states]

enhancement = np.max(overlap_with) / np.max(overlap_without) if np.max(overlap_without) > 0 else float('inf')
print(f"Island of stability (Z=126 proxy) enhancement factor: {enhancement:.1f}x")

plt.figure(figsize=(10,6))
plt.plot(times, overlap_with, label=f'With trefoil catalysis (enhancement {enhancement:.1f}x)', color='gold', linewidth=3)
plt.plot(times, overlap_without, '--', label='Standard Z=126 barrier', color='darkred', linewidth=2.5)
plt.title('TET--CVTL Enhancement Toward Island of Stability (Z=126 Proxy)')
plt.xlabel('Time (arbitrary units)')
plt.ylabel('Fusion channel overlap probability')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('island_stability_Z126_enhancement.pdf')
plt.savefig('island_stability_Z126_enhancement.png', dpi=300)
print("Figure saved: island_stability_Z126_enhancement.pdf / .png")
