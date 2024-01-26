# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import numpy as np
from experiments.util import create_dir, write_row
from .operations.initialization import Initialization
from .operations.crossover import PermutationX
from .operations.mutation import PermutationMut
from .operations.selection import SelectIndividuals, KElitism
from .operations.fitness import QuFitnessCalculator
from typing import List
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger("sanchez-genetic")

class _GeneticResultsHandler:

    def __init__(self, results_dir: str):
        self._results_dir = results_dir
        
        #create_dir(self._results_dir)
        #self.write_csv(statistics_header, filename, "w+")
    
    def write_csv(self, data, file_name: str, mode: str):
        file_path = os.path.join(f"{self._results_dir}/csv", file_name)
        write_row(data, file_path, mode)
    
    def save_plots(self, statistics: dict, filename: str):
        assert "mean_fitness" in statistics
        assert "best_fitness" in statistics

        plt.title("Fitness over  the generations")
        plt.plot(statistics["mean_fitness"], "--", color="gray", label="mean_fitness")
        plt.plot(statistics["best_fitness"], "-.", color="red", label="best_fitness")
        plt.legend(loc="best")
        plt.ylabel("Fitness")
        plt.xlabel("Generations")
        plt.grid()

        plot_file = os.path.join(self._results_dir, "plots", filename)
        plt.savefig(plot_file)
        plt.clf()
        plt.close()

class SanchezGenetic:

    def __init__(
            self, 
            n_gen: int,
            num_qubits: int,
            eps: int,
            results_dir: str = "results"
            ):
        
        self._n_gen = n_gen
        self._num_qubits = num_qubits
        self._eps = eps
        self._statistics = { 
            "mean_fitness": [],
            "std_fitness": [],
            "best_fitness": [],
        }

        self._best_individual = []
        self._best_individual_params = []
        self._results_handler = _GeneticResultsHandler(results_dir)

        # writing statistics  header:
        self._statistics_filename = f"statistics_{self._num_qubits}qb_{self._eps}eps.csv"
        self._results_handler.write_csv(list(self._statistics.keys()), mode="w+", file_name=self._statistics_filename)

        self._best_individual_fname = f"best_individual_{self._num_qubits}qb_{self._eps}eps.csv"
        self._best_individual_params_fname = f"best_individual_params_{self._num_qubits}qb_{self._eps}eps.csv"
        self._plot_fname = f"plot_fitness_over_generations_{self._num_qubits}qb_{self._eps}eps.pdf"
        

    def save_statistics(self, population, fitness, individual_params): 

        best_idx = fitness.argmin()
        mean_fitness = np.mean(fitness)
        std_fitness = np.std(fitness)

        self._statistics["mean_fitness"] += [mean_fitness]
        self._statistics["std_fitness"] += [std_fitness]
        self._statistics["best_fitness"] += [fitness[best_idx]]

        self._best_individual += [population[best_idx]]
        self._best_individual_params += [individual_params[best_idx]]

        self._results_handler.write_csv([mean_fitness, std_fitness, fitness[best_idx]], self._statistics_filename, mode="a")
        self._results_handler.write_csv([self._best_individual[-1]], self._best_individual_fname, mode="a+")
        self._results_handler.write_csv([self._best_individual_params[-1]], self._best_individual_params_fname, mode="a+")


    def evolve(
            self,
            pop_initializer: Initialization,
            crossover_op: PermutationX, 
            mutation_op: PermutationMut,
            fitness_calculator: QuFitnessCalculator,
            selection_op: SelectIndividuals, 
            k_elitism: KElitism):
        
        pop_size = pop_initializer.pop_size

        population = pop_initializer.random()
        fitness = np.apply_along_axis(fitness_calculator.compute_fitness, 1, population)
        individual_params = fitness_calculator.individual_params()
        
        fitness_calculator.reset_individual_params()

        self.save_statistics(population, fitness, individual_params)

        for gen_idx in range(self._n_gen): 

            best_fitness = self._statistics["best_fitness"][-1]
            best_individual = self._best_individual[-1]

            logger.info(f"N gen: {gen_idx + 1}; Best fitness: {np.round(best_fitness, 4)}; Best individual: {best_individual}")

            # creating new population
            new_population = []

            for _ in range(pop_size // 2):
                parent_1, parent_2 = selection_op.apply(population, fitness)
                child_1, child_2 = crossover_op.apply(parent_1, parent_2)
                mchild_1 = mutation_op.apply(child_1)
                mchild_2 = mutation_op.apply(child_2)

                new_population += [mchild_1, mchild_2]
            
            # computing new fitness
            new_population = np.array(new_population, dtype=int)
            new_fitness = np.apply_along_axis(fitness_calculator.compute_fitness, 1, new_population)
            new_individual_params = fitness_calculator.individual_params()
            fitness_calculator.reset_individual_params()

            # select survivors
            population, fitness, individual_params = k_elitism.apply_with_params(
                                                        population, fitness, individual_params,
                                                        new_population, new_fitness, new_individual_params)

            self.save_statistics(population, fitness, individual_params)
        
        self._results_handler.save_plots(self._statistics, self._plot_fname)
