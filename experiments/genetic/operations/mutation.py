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
import copy


def _apply_swap(inidividual: np.ndarray) -> np.ndarray:
    """
    Effectively apply swapped mutaiton to the indiviudal
    """
    index_range = list(range(len(inidividual)))
    choices = np.random.choice(index_range, 2)
    
    #swap
    temp = inidividual[choices[0]]
    inidividual[choices[0]] = inidividual[choices[1]]
    inidividual[choices[1]] = temp

    return inidividual

def _apply_scramble(individual: np.ndarray) -> np.ndarray:
    """
    Effectively applies scramble mutation to individual
    """
    start = np.random.randint(0, len(individual)-2)
    end = np.random.randint(start+1, len(individual)-1)

    np.random.shuffle(individual[start: end + 1])

    return individual

def _apply_inverse(individual: np.ndarray) -> np.ndarray:
    """
    Effectively aplies inverse mutation
    """

    start = np.random.randint(0, len(individual)-2)
    end = np.random.randint(start+1, len(individual)-1)

    individual[start: end + 1] = list(reversed(individual[start: end + 1]))

    return individual

def _apply_insert(individual: np.ndarray) -> np.ndarray:
    """
    Effectivelly applies the insert mutations
    """
    start = np.random.randint(0, len(individual)-2)
    end = np.random.randint(start+1, len(individual)-1)

    temp_end = individual[end]

    for segment_index in range(end, start+1, -1):
        temp_item = individual[segment_index-1]
        individual[segment_index] = temp_item

    individual[start + 1] = temp_end

    return individual

class PermutationMut:

    def __init__(self, probability: float = 0.5, mutation_type: str = "swap"):
        
        self.mutation_optioins = ["scramble", "inverse", "insert", "swap"]

        assert mutation_type in self.mutation_optioins

        self.probability = probability
        self.mutation_type = mutation_type

    def swap(self, individual: np.ndarray) -> np.ndarray: 
        rand_prob = np.random.rand()
        if rand_prob < self.probability: 
            return _apply_swap(individual)
        return copy.deepcopy(individual)
    
    def scramble(self, individual: np.ndarray) -> np.ndarray:
        rand_prob = np.random.rand()
        if rand_prob < self.probability:
            return _apply_scramble(individual)
        return copy.deepcopy(individual)

    def inverse(self, individual: np.ndarray) -> np.ndarray:
        rand_prob = np.random.rand()
        if rand_prob < self.probability:
            return _apply_inverse(individual)
        return copy.deepcopy(individual)

    def insert(self, individual: np.ndarray) -> np.ndarray:
        rand_prob = np.random.rand()
        if rand_prob < self.probability:
            return _apply_insert(individual)
        return copy.deepcopy(individual)

    def apply(self, individual: np.ndarray) -> np.ndarray:

        if self.mutation_type == "swap":
            return self.swap(individual)
        elif self.mutation_type == "scramble":
            return self.scramble(individual)
        elif self.mutation_type == "inverse":
            return self.inverse(individual)
        elif self.mutation_type == "insert":
            return self.insert(individual)
