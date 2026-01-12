"""
TET--CVTL: QuTiP simulation for aneutronic fusion cycles enhancement
Simon Soliman, TET Collective, January 2026
License: CC BY-NC-ND 4.0
"""

import qutip as qt
import numpy as np
import matplotlib.pyplot as plt

# Primordial trefoil phase
theta = 6 * np.pi / 5

# Effective Z for different aneutronic cycles
cycles = {
    'p-11B': 6.0,      # Z=1 (p) + Z=5 (B) effective
    'D-3He': 2.0,      # Z=1 (D) + Z=2 (3He) effective
    'p-7Li': 4.0       # Z=1 (p) + Z=3 (Li) effective
}

fig, axs = plt.subplots(3, 1, figsize=(10, 12))
fig.suptitle('TET--CVTL Enhancement in Aneutronic Fusion Cycles')

for idx, (name, Z_eff) in enumerate(cycles.items()):
    # Base Hamiltonian (repulsive Coulomb proxy)
    H0 = Z_eff * qt.tensor(qt.sigmax(), qt.sigmax())

    # Anyonic catalysis term (collective scaling)
    phase = np.exp(1j * theta * Z_eff**0.5)
    phase_op = qt.tensor(qt.qeye(2), qt.qdiags([1.0, phase], 0))

    H_eff = H0 + phase_op

    # Initial entangled state
    psi0 = (qt.tensor(qt.basis(2,0), qt.basis(2,1)) + 
            qt.tensor(qt.basis(2,1), qt.basis(2,0))).unit()

    # Fused state proxy
    fused = qt.tensor(qt.basis(2,0), qt.basis(2,0))

    times = np.linspace(0, 15, 500)

    result_with = qt.mesolve(H_eff, psi0, times)
    overlap_with = [abs(fused.overlap(state))**2 for state in result_with.states]

    result_without = qt.mesolve(H0, psi0, times)
    overlap_without = [abs(fused.overlap(state))**2 for state in result_without.states]

    enhancement = np.max(overlap_with) / np.max(overlap_without)
    print(f"{name} enhancement factor: {enhancement:.1f}x")

    axs[idx].plot(times, overlap_with, label=f'With catalysis (enhancement {enhancement:.1f}x)', color='gold', linewidth=3)
    axs[idx].plot(times, overlap_without, '--', label='Standard barrier', color='darkred', linewidth=2.5)
    axs[idx].set_title(f'{name} Cycle')
    axs[idx].set_xlabel('Time (arbitrary units)')
    axs[idx].set_ylabel('Fusion overlap probability')
    axs[idx].legend()
    axs[idx].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('aneutronic_cycles_enhancement.pdf')
plt.savefig('aneutronic_cycles_enhancement.png', dpi=300)
print("Figure saved: aneutronic_cycles_enhancement.pdf / .png")
