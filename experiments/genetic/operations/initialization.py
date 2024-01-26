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


class Initialization:

    def __init__(self, individual_size: int, pop_size: int = 200, origin=None):
        self.individual_size = individual_size
        self.pop_size = pop_size
        self.origin = origin

    def random(self) -> np.ndarray:
        pop = np.empty((self.pop_size, self.individual_size), dtype=int)
        for el_idx in range(len(pop)):
            pop[el_idx] = np.random.permutation(self.individual_size)

        #pop = self.remove_origin(pop)

        return pop

    def _remove_origin_from_individual(self, individual: np.ndarray) -> np.ndarray:
        # if there is an origin, removeit from 
        # recently added permutation
        if self.origin is not None: 
            origin_idx = np.where(individual == self.origin)[0][0]
            individual = np.delete(individual, origin_idx)

        return individual

    def remove_origin(self, population: np.ndarray) -> np.ndarray:
        return np.apply_along_axis(self._remove_origin_from_individual, 1, population)
