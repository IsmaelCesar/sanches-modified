from typing import List
import numpy as np
import pandas as pd
from sanchez_ansatz import SanchezAnsatz
from qiskit import QuantumCircuit, transpile, execute
from qiskit_aer import Aer

def generate_target_state(
        num_qubits: int, 
        complex_state: bool = False, 
        seed: int = 7 
    ) -> np.ndarray:

    rng = np.random.default_rng(seed)
    state = rng.random(2**num_qubits)

    if complex_state:
        state = state.astype(np.complex64)
        state += 1j*np.random.rand(2**num_qubits)
    
    return state / np.linalg.norm(state)

def run_circuit(circuit: QuantumCircuit):
    sv_sim = Aer.get_backend("statevector_simulator")
    job = execute(circuit, sv_sim)
    return job.result().get_statevector()

def compute_depth(
        qubit_range: List[int] = [3, 10], 
        eps: float = 0.01, 
        use_complex_states: bool = True
    ) -> pd.DataFrame:
    qubit_range = list(range(*qubit_range))

    depth_data = { "num_qubits": [], "depth_original": [], "depth_modified": [] }

    # Measuring depth: 
    for num_qubits in qubit_range:
        state = generate_target_state(num_qubits, complex_state = use_complex_states)

        original = SanchezAnsatz(state, eps, name="Original")
        modified = SanchezAnsatz(state, eps, name="Modified", build_modified=True)

        t_original = transpile(original, basis_gates=["u", "cx"])
        t_modified = transpile(modified, basis_gates=["u", "cx"])

        depth_data["num_qubits"] += [num_qubits]
        depth_data["depth_original"] += [t_original.depth()]
        depth_data["depth_modified"] += [t_modified.depth()]

    df_depth = pd.DataFrame(depth_data)
    df_depth.set_index("num_qubits", inplace=True)
    return df_depth

def compute_fidelity_loss(
        qubit_range: List[int] = [3, 10], 
        eps: float = 0.01, 
        use_complex_states: bool = False
    ) -> pd.DataFrame:

    fidloss_data = { "num_qubits": [], "fidloss_original": [], "fidloss_modified": [] }

    qubit_range = list(range(*qubit_range))
    for num_qubits in qubit_range:
        state = generate_target_state(num_qubits, complex_state=use_complex_states)
        original = SanchezAnsatz(state, eps, name="Original")
        modified = SanchezAnsatz(state, eps, name="Modified", build_modified=True)


        # assigning parameters to experiments
        p_original = original.assign_parameters(original.init_params)
        p_modified = modified.assign_parameters(modified.init_params)

        original_sv = run_circuit(p_original)
        original_data = original_sv.data

        modified_sv = run_circuit(p_modified)
        modified_data = modified_sv.data

        # saving data
        fidloss_data["num_qubits"] += [num_qubits]
        fidloss_data["fidloss_original"]  += [1 - np.abs(state @ original_data.T)**2]
        fidloss_data["fidloss_modified"]  += [1 - np.abs(state @ modified_data.T)**2]

    df_fidloss = pd.DataFrame(fidloss_data)
    df_fidloss.set_index("num_qubits", inplace=True)
    return df_fidloss


if __name__ == "__main__":

    qubit_range = [3, 10]
    errors = [0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
    for eps in errors:
        df_depth = compute_depth(qubit_range=qubit_range)
        print(df_depth)
        df_depth.to_csv(f"circ_depth_error-{eps}.csv", sep=",")

        df_fidelity_loss = compute_fidelity_loss(qubit_range=qubit_range, eps=eps)
        print(df_fidelity_loss)
        df_fidelity_loss.to_csv(f"fid_loss_error-{eps}.csv", sep=",")
