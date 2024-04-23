# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# scripts for running tests on cluster
import os
import numpy as np
from sanchez_ansatz import SanchezAnsatz
from qiskit_aer import Aer
from qiskit_algorithms.optimizers import SPSA
from qiskit import QuantumCircuit, transpile
from copy import deepcopy
from experiments import (ExperimentModule, 
                         get_state, 
                         create_dir, 
                         write_row, 
                         load_config_file)
from typing import List

RESULTS_SCRIPT_PATH = "results/script_results"


def get_resulting_state(quantum_circuit: QuantumCircuit, init_params: List[float]) -> np.ndarray:
    t_qc = quantum_circuit.assign_parameters(init_params)
    sv_sim = Aer.get_backend("statevector_simulator")
    job = sv_sim.run(t_qc)
    job_result = job.result()

    return job_result.get_statevector().data


def run_dist(
        dist_type: str,
        dist_params: dict,
        num_qubits: int,
        eps: float,
        eta: float = 4*np.pi,
        build_modified: bool = False,
        maxiter: int = 250,
        n_runs: int = 10) -> None:

    dist_path = os.path.join(RESULTS_SCRIPT_PATH, dist_type)
    create_dir(dist_path)

    target_state = get_state(num_qubits, dist_type, dist_params)

    for run_idx in range(n_runs):
        sanches_ansatz = SanchezAnsatz(
                            target_state,
                            eps=eps,
                            eta=eta,
                            build_modified=build_modified
                        )
        init_params = sanches_ansatz.init_params
        t_sanchez = transpile(sanches_ansatz, basis_gates=["u", "cx"])

        em = ExperimentModule(
                t_sanchez,
                SPSA(maxiter=maxiter),
                target_state,
                init_params)

        result = em.minimize()

        approx_state = get_resulting_state(t_sanchez, result.x)
        approx_state = approx_state.astype(np.float32)

        og_mod_prefix = "modified" if build_modified else "original"
        filename = f"{og_mod_prefix}_plot_{dist_type}_{num_qubits}qb_{eps}eps.csv"

        filepath = os.path.join(dist_path, filename)
        write_row(approx_state.tolist(), filepath)


def main():
    create_dir(RESULTS_SCRIPT_PATH)

    config_data = load_config_file("script_experiment_config.yml")

    eps = config_data["eps"]
    num_qubits = config_data["num_qubits"]
    n_runs = config_data["n_runs"]
    maxiter = config_data["maxiter"]

    for dist_data in config_data["distributions"]:
        dist_type = dist_data["dist_type"]
        dist_params = dist_data["dist_params"]

        run_dist(dist_type, 
                 deepcopy(dist_params), 
                 num_qubits,
                 eps=eps,
                 maxiter=maxiter, 
                 n_runs=n_runs)
        
        run_dist(dist_type,
                 deepcopy(dist_params),
                 num_qubits,
                 eps=eps,
                 build_modified=True,
                 maxiter=maxiter, 
                 n_runs=n_runs)

if __name__ == "__main__":
    main()
