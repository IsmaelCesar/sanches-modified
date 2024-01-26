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
from typing import Tuple, List

def _scale_fitness(fitness: np.ndarray) -> np.ndarray:
    # scales fitness values to they all sum up to one
    return (fitness - fitness.min())/ (fitness.max() - fitness.min())


def _apply_roulette_wheel(individuals: np.ndarray, fitness: np.ndarray , num_individuals: int )-> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Effectivelly applies the roulette wheel algorithms
    """
    chosen = [None] * num_individuals
    chosen_indices = []
    parent_idx = 0
    pool_range = list(range(len(individuals)))

    while parent_idx < num_individuals:

        random_prob = np.random.uniform(low=0, high=1)
        #indiv_count = 0 
        #while fitness[pool_range[indiv_count]] > random_prob: 
        #    indiv_count += 1
        chosen_pr = -1
        for pr in pool_range:
            if fitness[pr] < random_prob:
                chosen_pr = pr
                break
        
        chosen_pr = np.random.choice(pool_range) if chosen_pr == -1 else chosen_pr

        chosen[parent_idx] = individuals[chosen_pr]
        chosen_indices += [chosen_pr]
        pool_range.remove(chosen_pr)

        parent_idx += 1
    
    return chosen, chosen_indices

class SelectIndividuals:

    def __init__(
            self, 
            num_individuals: int =  2, 
            selection_type: str = "tournament",
            scale_fitness: bool = True):

        assert selection_type in ["random", "tournament", "roulette"]

        self.num_individuals = num_individuals
        self.selection_type = selection_type
        self.scale_fitness = scale_fitness

    def random(self, individuals: np.ndarray) -> List[np.ndarray]:
        """
        Randomly selects two individuals from the individuals array for reproduction.
        """
        pool_range = list(range(len(individuals)))
        parents = [None] * self.num_individuals
        
        for  p_idx, selected_p_idx in enumerate(np.random.choice(pool_range, self.num_individuals)):
            parents[p_idx] = individuals[selected_p_idx]
        
        return parents

    def tournament(self, individuals: np.ndarray, fitness: np.ndarray) -> List[np.ndarray]:
        """
        Select individuals from individuals array for reproduction based on tournament algorithm
        """
        parent_idx = 0
        parents = [None] * self.num_individuals
        pool_range = list(range(len(individuals)))

        while parent_idx < len(parents):
            individuals_indices = np.random.choice(pool_range, 3)
            best_one = individuals_indices[0]
            for indiv_idx in individuals_indices[1:]:
                if fitness[best_one] < fitness[indiv_idx]:
                    best_one = indiv_idx
                    pool_range.remove(indiv_idx)
            
            parents[parent_idx] = individuals[best_one]
            parent_idx += 1
        
        return parents

    def roulette_wheel(
            self, 
            individuals: np.ndarray, 
            fitness: np.ndarray) -> List[np.ndarray]:
        """
        Applies roulette wheel algorithm for selecting parents.
        """
        chosen_individuals, _ = _apply_roulette_wheel(individuals, fitness, self.num_individuals)
        return chosen_individuals
       

    def apply(self, individuals: np.ndarray, fitness: np.ndarray) -> List[np.ndarray]:

        if self.scale_fitness: 
            temp_fit = _scale_fitness(fitness)
        else: 
            temp_fit = fitness

        if self.selection_type == "random": 
            return self.random(individuals)
        elif self.selection_type == "tournament":
            return self.tournament(individuals, temp_fit)
        elif self.selection_type == "roulette":
            return self.roulette_wheel(individuals, temp_fit)

class KElitism:
    """
    Esta classe implementa o k-elitismo pars single
    traveling salesman problem,
    onde a mesma garante que os k-melhores individuos
    da população anterior passem para geração seguinte
    """
    def __init__(self, k: int = 1):
        self.k = k

    def apply(
            self, 
            old_population: np.ndarray, 
            old_fitness: np.ndarray, 
            new_population: np.ndarray, 
            new_fitness: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Effectivelly applies K-elitism
        """

        pop_size = len(old_population)
        num_var = len(old_population[0])        

        # getting best k of old population
        best_k = old_fitness.argsort()[:self.k]
        
        # eliminating the worst k of new population
        rest_of_new = new_fitness.argsort()[::-1][self.k:]

        updated_pop = np.empty((pop_size, num_var), dtype=int)
        updated_fit = np.empty((pop_size,))
        
        updated_pop[:self.k] = old_population[best_k]
        updated_fit[:self.k] = old_fitness[best_k]

        updated_pop[self.k:] = new_population[rest_of_new]
        updated_fit[self.k:] = new_fitness[rest_of_new]

        return updated_pop, updated_fit
    
    def apply_with_params(
            self,
            old_population: np.array, 
            old_fitness: np.array, 
            old_params: np.array,
            new_population: np.array, 
            new_fitness: np.array, 
            new_params: np.array) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Effectivelly applies K-elitism
        """

        pop_size = len(old_population)
        num_var = len(old_population[0])
        num_params = len(old_params[0])

        # getting best k of old population
        best_k = old_fitness.argsort()[:self.k]
        
        # eliminating the worst k of new population
        rest_of_new = new_fitness.argsort()[::-1][self.k:]

        updated_pop = np.empty((pop_size, num_var), dtype=int)
        updated_params = np.empty((pop_size, num_params))
        updated_fit = np.empty((pop_size,))
        
        updated_pop[:self.k] = old_population[best_k]
        updated_fit[:self.k] = old_fitness[best_k]
        updated_params[:self.k] = old_params[best_k]

        updated_pop[self.k:] = new_population[rest_of_new]
        updated_fit[self.k:] = new_fitness[rest_of_new]
        updated_params[self.k:] = new_params[rest_of_new]

        return updated_pop, updated_fit, updated_params

class FitnessProportional:

    def __init__(self, pop_size: int, num_cidades: int):

        self.pop_size = pop_size
        self.num_cidades = num_cidades
        self.num_individuals = pop_size //2

    def apply(
            self, 
            old_population: np.ndarray,
            old_fitness: np.ndarray, 
            new_population: np.ndarray, 
            new_fitness: np.ndarray) -> np.ndarray:
        
        scaled_old_fit = _scale_fitness(old_fitness)
        scaled_new_fit = _scale_fitness(new_fitness)

        survivors = np.zeros((self.pop_size, self.num_cidades), dtype=int) -1
        survivors_fitness = np.zeros((self.pop_size,)) -1
        
        old_survivors, chosen_old_indices = _apply_roulette_wheel(old_population, scaled_old_fit, self.num_individuals)
        old_survivors = np.array(old_survivors)
        old_survivors_fitness = old_fitness[chosen_old_indices]
        
        survivors[:self.num_individuals] = old_survivors
        survivors_fitness[:self.num_individuals] = old_survivors_fitness

        new_survivors, chosen_new_indices =  _apply_roulette_wheel(new_population, scaled_new_fit, self.num_individuals)
        new_survivors = np.array(new_survivors)
        new_survivors_fitness = new_fitness[chosen_new_indices]

        survivors[self.num_individuals:] = new_survivors
        survivors_fitness[self.num_individuals:] = new_survivors_fitness

        return survivors, survivors_fitness
