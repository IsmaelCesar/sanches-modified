# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# contains code for executing a single iteration of the 
# experiment

import os
from typing import Union, List
import numpy as np
from qiskit import QuantumCircuit
from qiskit.algorithms.optimizers import (
    Optimizer,
    SPSA
)
from qiskit.result import Result
from qiskit_aer import Aer
from argparse import ArgumentParser
from sanchez_ansatz import SanchezAnsatz
from qiskit import transpile

parser = ArgumentParser()
parser.add_argument("--num_qubits", type=int, help="The number of qubits in the system")
parser.add_argument("--eps", type=float, help="The value indicating the tolerated error")
parser.add_argument(
    "--eta", 
    type=float,
    default=4*np.pi,
    help=("Hyperparameter defining the supremum point"+
          " of the second order derivative of the log"+ 
          " of the function fo be approximated."))
parser.add_argument(
    "--run_modified", 
    action="store_true", 
    default=False, 
    help="define whether or not to run the modified version"
)
parser.add_argument(
    "--result_dir", 
    type=str,
    default="results",
    help="Define the target directory where the results are to be saved"
)
parser.add_argument(
    "--basis_gates",
     type="+", 
     default=["cx", "u"],
     help="Define the basis_gates to be used in the device"
)
args = parser.parse_args()


class ExperimentModule(): 

    def __init__(
        self, 
        ansatz: QuantumCircuit,
        optimizer: Optimizer,
        target_state: np.ndarray,
        init_params: Union[List[float],np.ndarray] = None,
    ):
        self._ansatz = ansatz
        self._optimizer = optimizer
        self._target_state = target_state
        self._init_params = init_params
        self._loss_progression = []

    def _get_statevector(self, x) -> np.ndarray: 
        sv_sim = Aer.get_backend("statevector_simulator")
        param_circ = self._ansatz.assign_parameters(x)
        job = sv_sim.run(param_circ)
        result = job.result()
        statevector = result.get_statevector()
        return statevector.data

    def _callback_fn(self, value) -> None: 
        self._loss_progression += [value]

    def _objectivive_fn(self) -> callable:
        def objective_fid(x):
            state_vec = self._get_statevector(x)
            fid_loss = 1 - np.abs(self._target_state @ state_vec) ** 2
            
            self._callback_fn(fid_loss)
            return fid_loss

        return objective_fid

    def minimize(self) -> Result:

        init_params = self._init_params
        if init_params is None:
            init_params = np.pi / 4 * np.random.rand(self.ansatz.num_parameters)

        return self._optimizer.minimize(self._objectivive_fn(), x0=init_params)

def experiments(
    num_qubits: int,
    eps: float,
    eta: float,
    run_modified: bool,
    result_dir: str,
    basis_gates: ["cx", "u"]
):

    state = np.random.rand(2**num_qubits)
    state = state / np.linalg.norm(state)

    ansatz = SanchezAnsatz(target_state=state, eps=eps, eta=eta)
    
    init_params = ansatz.init_params
    t_ansatz  = transpile(ansatz, basis_gates=basis_gates)

    em = ExperimentModule(
            t_ansatz, 
            SPSA(maxiter=100),
            target_state=state,
            init_params=init_params
        )

    res = em.minimize()

if __name__ == "__main__":
    arg_map = dict(args._get_kwargs())
    experiments(**arg_map)
