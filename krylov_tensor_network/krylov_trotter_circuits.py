from TN_circuit import *
from lattice_afi import *
import numpy as np
from scipy.linalg import expm, toeplitz
import scipy as sp

X = np.array([[0,1],[1,0]])
Y = np.array([[0,-1j],[1j,0]])
Z = np.array([[1,0],[0,-1]])
I = np.eye(2)
XX = np.kron(X,X)
YY = np.kron(Y,Y)
ZZ = np.kron(Z,Z)
zero = np.array([1,0])
one = np.array([0,1])

class HeavyHexHeisenbergKrylovTNSim(TNCircuit):

    def __init__(self, nx: int, ny: int, chi_max:int, J: float=1.0, g:float=1.0, mps_order=None):
        '''
        Params:
            nx, ny: Linear dimensions of Heavy Hex lattice
            chi_max: Maximum bond dimensions to truncate to for all computations
            J: Heisenberg coupling constant
            g: Anisotropy in ZZ couplings for XXZ model (=/= 1 away from SU(2) point)
        '''
        self.lattice = lattice_2d(tp='heavy_hex', nx=nx, ny=ny, mps_order=mps_order)
        TNCircuit.__init__(self, mps_order=[0]+[self.lattice.mps_to_qubit[i]+1 for i in range(self.lattice.n_qubits)], chi_max=chi_max)

        #+1 for ancilla
        self.J = J
        self.g = g

        self.pre_measure_states = None
        self.flip_inds = None

        self.layer1_links = [(self.lattice.lat_to_mps_idx(k[0])+1, self.lattice.lat_to_mps_idx(k[1])+1) for k,v in self.lattice.couplings.items() 
                             if self.lattice.lat_to_mps_idx(k[0]) < self.lattice.lat_to_mps_idx(k[1]) and v==1]
        self.layer2_links = [(self.lattice.lat_to_mps_idx(k[0])+1, self.lattice.lat_to_mps_idx(k[1])+1) for k,v in self.lattice.couplings.items() 
                             if self.lattice.lat_to_mps_idx(k[0]) < self.lattice.lat_to_mps_idx(k[1]) and v==2]
        self.layer3_links = [(self.lattice.lat_to_mps_idx(k[0])+1, self.lattice.lat_to_mps_idx(k[1])+1) for k,v in self.lattice.couplings.items() 
                             if self.lattice.lat_to_mps_idx(k[0]) < self.lattice.lat_to_mps_idx(k[1]) and v==3]
        

    def initialize_state(self, initial_state: list | int | str = 'GHZ', coeffs = None, flip_inds = None) -> None:
        self.flip_inds = [self.lattice.lat_to_mps_idx(flip_inds[i]) + 1 for i in range(len(flip_inds))]
        super().initialize_state(initial_state, coeffs, [0]+self.flip_inds)
        return
    
    def apply_trotter_layer(self, t):
        self.apply_two_q_gate_layer([expm(-1j*t*XX) for _ in range(len(self.layer1_links))], self.layer1_links, chi=self.chi_max)
        self.apply_two_q_gate_layer([expm(-1j*t*YY) for _ in range(len(self.layer1_links))], self.layer1_links, chi=self.chi_max)
        self.apply_two_q_gate_layer([expm(-1j*t*ZZ) for _ in range(len(self.layer1_links))], self.layer1_links, chi=self.chi_max)
        self.apply_two_q_gate_layer([expm(-1j*t*XX) for _ in range(len(self.layer2_links))], self.layer2_links, chi=self.chi_max)
        self.apply_two_q_gate_layer([expm(-1j*t*YY) for _ in range(len(self.layer2_links))], self.layer2_links, chi=self.chi_max)
        self.apply_two_q_gate_layer([expm(-1j*t*ZZ) for _ in range(len(self.layer2_links))], self.layer2_links, chi=self.chi_max)
        self.apply_two_q_gate_layer([expm(-1j*t*XX) for _ in range(len(self.layer3_links))], self.layer3_links, chi=self.chi_max)
        self.apply_two_q_gate_layer([expm(-1j*t*YY) for _ in range(len(self.layer3_links))], self.layer3_links, chi=self.chi_max)
        self.apply_two_q_gate_layer([expm(-1j*t*ZZ) for _ in range(len(self.layer3_links))], self.layer3_links, chi=self.chi_max)
        return
    
    #assumes initial state is a layer of controlled unitaries applied to the all |0> state, controlled on the ancilla qubit
    def apply_controlled_state_prep(self, control_val, target_inds = None, Us = None):
        if target_inds is None:
            target_inds = self.flip_inds
        if Us is None:
            Us = [X for _ in range(len(target_inds))]
        if control_val == 1:
            CUs = [np.kron(np.outer(zero,zero), I) + np.kron(np.outer(one,one), U) for U in Us]
        elif control_val == 0:
            CUs = [np.kron(np.outer(one,one), I) + np.kron(np.outer(zero,zero), U) for U in Us]
        self.apply_two_q_gate_layer(CUs, [(0, t) for t in target_inds], chi=self.chi_max)
        return

    def meas_hadamard_test(self, pauli_op_string, ancilla_op):
        op_string = (ancilla_op + pauli_op_string).lower()
        op_list = [*op_string]
        op_list = ['Sigma'+op_list[i] if op_list[i] != 'i' else 'Id' for i in range(len(op_list))]
        self.state.canonical_form()
        return self.state.expectation_value_multi_sites(op_list, 0)

    def get_pre_measure_states(self, krylov_dim, dt, trotter_order):
        t_evolve_step = dt/trotter_order
        if self.pre_measure_states is None:
            unique_t_evolved_states = [self.state.copy()]
            for i in range(krylov_dim-1):
                for n in range(trotter_order):
                    self.apply_trotter_layer(t_evolve_step)
                copy_state = self.state.copy()
                copy_state.canonical_form()
                unique_t_evolved_states.append(copy_state)
        else:
            unique_t_evolved_states = self.pre_measure_states
            self.state = self.pre_measure_states[-1]
            for i in range(krylov_dim-len(self.pre_measure_states)):
                for n in range(trotter_order):
                    self.apply_trotter_layer(t_evolve_step)
                copy_state = self.state.copy()
                copy_state.canonical_form()
                unique_t_evolved_states.append(copy_state)

        self.pre_measure_states = unique_t_evolved_states
        return unique_t_evolved_states

    def heisenberg_pauli_strings(self):
        op_strings = []
        coeffs = []
        for pair in self.layer1_links+self.layer2_links+self.layer3_links:
            char_list = ['X' if i in pair else 'I' for i in range(1,self.L)]
            op_strings.append(''.join(char_list))
            coeffs.append(self.J)
            char_list = ['Y' if i in pair else 'I' for i in range(1,self.L)]
            op_strings.append(''.join(char_list))
            coeffs.append(self.J)
            char_list = ['Z' if i in pair else 'I' for i in range(1,self.L)]
            op_strings.append(''.join(char_list))
            coeffs.append(self.g)
        return op_strings, coeffs

    #Note: 1. This function destroys the internal state in its current form
    #      2.  Can be used for increasing krylov dimensinos without re-initializing the class as long as
    #          dt is not changed
    def krylov_exp_vals(self, krylov_dim, dt, trotter_order, type='H', pauli_ops=None, coeffs=None):
        if type == 'H':
            pauli_ops, coeffs = self.heisenberg_pauli_strings()
        elif type == 'S':
            pauli_ops = ['I'*(self.L-1)]
            coeffs = [1.0]
        if self.pre_measure_states is None:
            self.pre_measure_states = self.get_pre_measure_states(krylov_dim, dt, trotter_order)
        krylov_expvals = np.zeros(krylov_dim, dtype=np.complex128)
        for i in range(krylov_dim):
            self.state = self.pre_measure_states[i].copy()
            self.apply_controlled_state_prep(control_val=0)
            for j in range(len(pauli_ops)):
                krylov_expvals[i] += (self.meas_hadamard_test(pauli_ops[j], ancilla_op = 'X') 
                                       + 1j*self.meas_hadamard_test(pauli_ops[j], ancilla_op = 'Y'))*coeffs[j]        
        return krylov_expvals
    
    def krylov_H(self, krylov_dim=10, dt=0.1, trotter_order=2):    
        first_row_H = self.krylov_exp_vals(krylov_dim, dt, trotter_order, type='H')
        return toeplitz(first_row_H.conj())
    
    def krylov_S(self, krylov_dim=10, dt=0.1, trotter_order=2):    
        first_row_S = self.krylov_exp_vals(krylov_dim, dt, trotter_order, type='S')
        return toeplitz(first_row_S.conj())
    

def solve_regularized_gen_eig(h, s, k=1, threshold=1e-15, return_vecs=False):

    s_vals, s_vecs = sp.linalg.eigh(s)
    s_vecs = s_vecs.T
    good_vecs = [vec for val, vec in zip(s_vals, s_vecs) if val > threshold]
    if not good_vecs:
        raise AssertionError('WHOLE SUBSPACE ILL-CONDITIONED')
    good_vecs = np.array(good_vecs).T

    h_reg = good_vecs.conj().T @ h @ good_vecs
    s_reg = good_vecs.conj().T @ s @ good_vecs
    vals, vecs = sp.linalg.eigh(h_reg, s_reg)
    if return_vecs:
        return vals[0:k], good_vecs @ vecs[:,0:k]
    else:
        return vals[0:k]
    
