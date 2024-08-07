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

import numpy as np
from qiskit_algorithms.optimizers import SPSA
from argparse import ArgumentParser
from sanchez_ansatz import SanchezAnsatz
from qiskit import transpile
from experiments import ExperimentModule
from typing import List

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
     type=list,
     nargs="+",
     default=["cx", "u"],
     help="Define the basis_gates to be used in the device"
)
args = parser.parse_args()

def experiments(
    num_qubits: int,
    eps: float,
    eta: float,
    run_modified: bool,
    result_dir: str,
    basis_gates: List[str] = ["cx", "u"]
):

    state = np.random.rand(2**num_qubits)
    state = state / np.linalg.norm(state)

    ansatz = SanchezAnsatz(target_state=state, eps=eps, eta=eta, build_modified=run_modified)
    
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
