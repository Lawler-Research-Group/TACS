
"""
Created on Tue Nov  1 01:16:44 2022
# Gaurav Gyawali
# Cornell/Harvard University
# Code to generate Time Averaged Classical Shadows(TACS) using Exact Diagonalization(ED)
"""


from quspin.operators import hamiltonian # Hamiltonians and operators
from quspin.basis import spin_basis_general # Hilbert space spin basis_1d
from quspin.tools.evolution import ED_state_vs_time
import numpy as np

#eigenstates of Pauli matrices
z_up = np.array([1,0])
z_down = np.array([0,1])
x_up = (1/np.sqrt(2))*np.array([1,1])
x_down = (1/np.sqrt(2))*np.array([1,-1])
y_up = (1/np.sqrt(2))*np.array([1, 1j])
y_down = (1/np.sqrt(2))*np.array([1,-1j]) 


def get_couplings(N, J,hx,hz):
    '''get the sitewise couplings for given J, hx and hz'''
    J_list = []
    hx_list = []
    hz_list = []
    for i in range(N):
        if i<N-1:
            J_list.append([J,i,i+1])
        hx_list.append([hx,i])
        hz_list.append([hz,i])
    return J_list, hx_list, hz_list
    

def get_TFI_hamiltonian(basis,J,hx,hz):
    '''get the quspin Hamiltonian object for given TFI parameters'''
    N = basis.N
    J_int, hx_int,hz_int = get_couplings(N,J,hx,hz)
    static=[['zz',J_int],['x',hx_int],['z',hz_int]]
    no_checks = dict(check_pcon=False,check_symm=False,check_herm=False)
    H_TFI = hamiltonian(static,[],basis=basis,**no_checks) # caution: matrix is NOT sparse (!)                                                                              
    return H_TFI


def get_initial_state(basis,initial_state="Neel"):
    '''returns either the Neel, All_up or the Cat state'''
    N       = basis.N
    if initial_state == "Neel":
        neel_state = ('10' *  N)[0:N]
        index = basis.index(neel_state)
        psi    = np.zeros(basis.Ns,dtype=np.complex128)
        psi[index] = 1/np.sqrt(2)
    elif initial_state =="All_up":
        all_up = ('11' *  N)[0:N] 
        index = basis.index(all_up)
        psi    = np.zeros(basis.Ns,dtype=np.complex128)
        psi[index] = 1
    elif initial_state =="All_down":
         all_up = ('00' *  N)[0:N] 
         index = basis.index(all_up)
         psi    = np.zeros(basis.Ns,dtype=np.complex128)
         psi[index] = 1
    elif initial_state =="Cat":
        all_up = ('11' *  N)[0:N] 
        all_down = ('00' *  N)[0:N] 
        index_up = basis.index(all_up)
        index_down = basis.index(all_down)
        psi    = np.zeros(basis.Ns,dtype=np.complex128)
        psi[index_up] = 1/np.sqrt(2)
        psi[index_down] = 1/np.sqrt(2)
    else:
        raise TypeError("initial_state can only be 'Neel', 'All_up' or 'Cat'")
    return psi



def inner(psi1,psi2):
    return np.dot(psi1.flatten().conj(),psi2.flatten())


def projector(pauli,state):
    '''projector onto an eigenstate of a Pauli matrix
       0 stands for  +1 (|0>,|+>,+i>)and 1 stands for -1 (|1>,|->,-i>)eigenstates'''
    if pauli=="Z" and state == 0:
        psi = z_up
    elif pauli=="Z" and state == 1:
        psi= z_down
    elif pauli=="X" and state == 0:
        psi= x_up
    elif pauli=="X" and state == 1:
        psi= x_down
    elif pauli =="Y" and state ==0:
        psi= y_up
    elif pauli=="Y" and state==1:
        psi = y_down
    return np.outer(psi,psi.conj())



def ApplyGate(U, qubits, psi):
    ''''Multiplies a state psi by gate U acting on qubits'''
    # print(qubits)
    indices = ''.join([chr(97+q) for q in qubits])
    indices += ''.join([chr(65+q) for q in qubits])
    indices += ','
    indices += ''.join([chr(97+i-32*qubits.count(i)) for i in range(len(psi.shape))])
    return np.einsum(indices, U, psi)

def Measure(psi, paulibasis):
    '''project the wavefunctions onto an eigenstate of the given pauli basis
       uses 'qubit by qubit' method i.e. draws and outcome for a qubit, projects 
       the wavefunction,and goes on the calculate the probability for next qubit.
       This method calculates only N probabilities as opposed to 2^N for a naive algorithm.
       See Algorithm 1 in https://arxiv.org/pdf/2112.08499.pdf'''
    psi = psi.reshape((2,)*int(np.log2(len(psi))))
    result = []
    for i in range(len(paulibasis)):
        p0 = np.abs(inner(ApplyGate(projector(paulibasis[i],0), [i], psi),psi)) # probability of collapsing into 0 
        outcome = int(np.random.choice([0,1],1,p=[p0,abs(1-p0)]))               # draws an outcome 0 or 1 with probabilities p0 and 1-p0
        p = [p0,abs(1-p0)][outcome]
        psi = ApplyGate(projector(paulibasis[i],outcome), [i], psi)/np.sqrt(p) #projects the wavefunction onto the chosen basis
        result.append(str(paulibasis[i]) +str(outcome))
    return result

def classical_shadows(n_shots,psi):
    '''generates classical shadows shots from a given state. List with shape (n_shots, n_site)'''
    N_sites = int(np.log2(len(psi)))
    CS = []
    for i in range(n_shots):
        paulibasis = [["X","Y","Z"][np.random.randint(3)] for j in range(N_sites)] # generate a random Pauli basis
        CS.append(Measure(psi,paulibasis)) 
    return CS

def write_to_file(fname,cs_array):
    with open(fname + ".txt","a") as file:
        for cs in cs_array:
            file.write(cs + "\n")
        file.close()
        
def time_evolve_TFI(N_sites,J,hx,fname, N_shots = 100,T = 100, T_start=20, dt = 0.1, batchsize = 100,initial_state = "Cat"):
    '''Performs time evolution of the given initial state and generates TACS under the 1DTFIM Hamiltonian
    defined by J and hx. TACS sampling starts at T_start for a time window T. Since we are sampling at discrete
    time-intervals, a dt is chosen such that it captures the fastest dynamics. Sampling is done in small batches'''
    
    H_TFI = get_TFI_hamiltonian(basis_spin,J,hx,0)
    state_0 = get_initial_state(basis_spin,initial_state=initial_state)
    energies,states = H_TFI.eigh() #diagonalize the Hamiltonian
    t_list   = np.sort(T_start + T*np.random.rand(N_shots))              # Generates random time slices for genrating shots
    n_shots,t_list = np.histogram(t_list,density=False, bins=int(T/dt))  # Bins the random time slices in [T_start, T_start+T]into bins with interval dt
    N_batches = int(len(t_list)/batchsize)    
    T_list = t_list[:-1].reshape(N_batches,batchsize)                    # Divide the discrete time slices into batches
    print("Starting the time evolution")
    CS_list = []
    for j,t_list in enumerate(T_list):
        state_t  = ED_state_vs_time(state_0,energies,states,t_list,iterate=True)
        print("Running batch " + str(j) + "/" + str(N_batches))
        #print("T=", t_list[0], "n_shots = ", n_shots[0])
        for i, state in enumerate(state_t):
            #print("T=", t_list[i], "n_shots = ", n_shots[i])
            CS = classical_shadows(n_shots[j*batchsize+i],state)
            for cs in CS:
                CS_list.append(cs)
        np.save(fname, CS_list,allow_pickle=True)
    print("Finished writing output")
    
'''********************Settings***************************************'''
N_sites = 10
write_path = "" #Please create a new directory and write here
hx_list = np.arange(0.1,3.5,0.1)
T_start = 10      
T = 30
N_shots = 10000
dt = 0.001
basis_spin = spin_basis_general(N_sites,S='1/2')

for hx in hx_list:
    fname = write_path + "TACS_" + str(N_sites) + "_" + str(round(hx,1))
    print("**********************************\n hx=",hx, "fname = ", fname)
    time_evolve_TFI(N_sites,-1,hx,fname,N_shots,T,T_start,dt)
'''*******************************************************************'''
        