# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
from qiskit import QuantumCircuit
from qiskit import transpile
from qiskit_algorithms.optimizers import Optimizer, SPSA
from experiments.experiment_module import ExperimentModule
from typing import Union, List

class QuFitnessCalculator: 

    def __init__(
            self,
            ansatz: QuantumCircuit,
            init_params: Union[List[float], np.array],
            target_state: Union[List[float], np.array],
            optimizer: Optimizer = None,
    ):
        self._ansatz = ansatz
        self._init_params = init_params
        self._optimizer = optimizer
        self._target_state = target_state
        self._individual_params = []

        if not self._optimizer :
            self._optimizer = SPSA(maxiter=200)

    def _save_individual_params(self, x0) -> None:
        """
        Saves the optimized parameters of each individual evaluated by the fitness function
        """
        self._individual_params += [x0]

    def reset_individual_params(self) -> None:
        self._individual_params = []
    

    def inidividual_params(self) -> np.array:
        return np.array(self._individual_params)

    def compute_fitness(self, individual):
        t_ansatz = transpile(self._ansatz, initial_layout=individual.tolist())
        e_module = ExperimentModule(
                    t_ansatz, 
                    optimizer=self._optimizer, 
                    target_state=self._target_state,
                    init_params=self._init_params)
        minimization_result = e_module.minimize()
        
        self._save_individual_params(minimization_result.x)

        return minimization_result.fun