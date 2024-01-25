from experiments import get_state
from experiments.genetic import SanchezGenetic
from experiments.genetic.operations.initialization import Initialization
from experiments.genetic.operations.crossover import PermutationX
from experiments.genetic.operations.mutation import PermutationMut
from experiments.genetic.operations.fitness import QuFitnessCalculator
from experiments.genetic.operations.selection import SelectIndividuals, KElitism
from qiskit import transpile
from qiskit_algorithms.optimizers import SPSA
from sanchez_ansatz import SanchezAnsatz
import logging

logger = logging.getLogger("sanchez-genetic")
logger.setLevel(logging.INFO)

stream_handler = logging.StreamHandler()
logger.addHandler(stream_handler)

def main():

    num_qubits = 4
    target_state = get_state(num_qubits, "random-sparse", {"density": .5 })

    sanchez_circuit = SanchezAnsatz(target_state, 0.05, build_modified=True)
    init_params = sanchez_circuit.init_params

    t_sanchez = transpile(sanchez_circuit, basis_gates=["cx", "u"])

    
    genetic = SanchezGenetic(10)
    genetic.evolve(
        pop_initializer=Initialization(individual_size=num_qubits, pop_size=10),
        crossover_op=PermutationX(probability=.5, crossover_type="pmx"),
        mutation_op=PermutationMut(probability=.2, mutation_type="scramble"),
        fitness_calculator=QuFitnessCalculator(t_sanchez, init_params, target_state, SPSA(100)),
        selection_op=SelectIndividuals(num_individuals=2),
        k_elitism=KElitism(k=4)
    )


if __name__ == "__main__":
    main()
