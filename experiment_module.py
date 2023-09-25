# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# contains module used to perform the optimizations

from typing import Union, List
import numpy as np
from qiskit import QuantumCircuit
from qiskit_algorithms.optimizers import Optimizer
from qiskit.result import Result
from qiskit_aer import Aer

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
