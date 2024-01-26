import os
import numpy as np
from experiments import get_state, ParseKvAction
from experiments.genetic import SanchezGenetic
from experiments.genetic.operations.initialization import Initialization
from experiments.genetic.operations.crossover import PermutationX
from experiments.genetic.operations.mutation import PermutationMut
from experiments.genetic.operations.fitness import QuFitnessCalculator
from experiments.genetic.operations.selection import SelectIndividuals, KElitism
from qiskit import transpile
from experiments.util import create_dir
from qiskit_algorithms.optimizers import SPSA
from sanchez_ansatz import SanchezAnsatz
import argparse
import logging

logger = logging.getLogger("sanchez-genetic")
logger.setLevel(logging.INFO)

stream_handler = logging.StreamHandler()
logger.addHandler(stream_handler)

parser = argparse.ArgumentParser()
parser.add_argument("--results-dir", 
                    type=str,
                    required=True,
                    default="results", 
                    help="The directory where the results are to be saved")
parser.add_argument("--num-qubits",
                    type=int,
                    required=True, 
                    help="total number of qubits in the circuit")
parser.add_argument("--state-type",
                    type=str,
                    required=True,
                    choices=["random",
                             "random-sparse",
                             "lognormal",
                             "laplace",
                             "triangular",
                             "normal",
                             "bimodal"])
parser.add_argument("--eps",
                    type=float,
                    required=True,
                    help=("(Epsilon) tolerated error used for defining the approximation "+
                          "and at which level the tree will be truncated in the original "+
                          "article"))
parser.add_argument("--eta", 
                    type=float,
                    required=False,
                    default=4*np.pi,
                    help=("Hyperparameter defining the supremum point of the second order "+
                          "derivative of the log of the function fo be approximated." +
                          "Default value is 4*π"))
parser.add_argument("--state-params",
                    nargs="+", 
                    action=ParseKvAction,
                    required=False,
                    default=None,
                    help="The parameters of the state")
parser.add_argument("--run-idx",
                    type=int,
                    default=0,
                    required=False,
                    help="The index of the current-execution")
parser.add_argument("--n-gen", 
                    type=int,
                    required=True,
                    help="Total number of generations for the genetic algorithm")
parser.add_argument("--pop-size", 
                    type=int,
                    required=True,
                    help="Population size")
parser.add_argument("--crossover-type",
                    type=str,
                    choices=["pmx", "edge", "cycle", "order" ],
                    required=True,
                    help="Population size")
parser.add_argument("--mutation-type",
                    type=str,
                    choices=["scramble", "inverse", "insert", "swap"],
                    required=True,
                    help="Population size")
parser.add_argument("--crossover-prob", 
                    type=float,
                    default=.8,
                    required=False,
                    help="Defines the probability of the crossover operation to occur")
parser.add_argument("--mutation-prob", 
                    type=float,
                    default=.3, 
                    required=False,
                    help="Defines the probability of the mutation to occur")
parser.add_argument("--device", 
                    type=str,
                    default="CPU", 
                    required=False,
                    help="Defines the probability of the mutation to occur")
args = parser.parse_args()

def main(
    results_dir: str, 
    num_qubits: int,
    state_type: str, 
    eps: float,
    eta: float, 
    state_params: dict,
    run_idx: int,
    n_gen: int,
    pop_size: int,
    crossover_type: str,
    mutation_type: str,
    crossover_prob: float,
    mutation_prob: float,
    device: str
):
    
    #creating run dir
    run_dir = os.path.join(results_dir, f"run_{run_idx}")
    create_dir(run_dir)

    target_state = get_state(num_qubits, state_type=state_type, state_params=state_params)
    

    if "density" in state_params:
        density_value = state_params["density"]
        run_dir = os.path.join(run_dir, f"density_{density_value}")
        create_dir(run_dir)
    
    create_dir(f"{run_dir}/plots")
    create_dir(f"{run_dir}/csv")
    create_dir(f"{run_dir}/circuits")


    # Running original circuit
    build_modified = False
    modified_opts = [False, True]

    for build_modified in modified_opts:
        sanchez_ansatz = SanchezAnsatz(target_state, eps=eps, eta=eta, build_modified=build_modified)
        init_params = sanchez_ansatz.init_params

        t_sanchez = transpile(sanchez_ansatz, basis_gates=["cx", "u"])

        genetic = SanchezGenetic(n_gen=n_gen, num_qubits=num_qubits, eps=eps, build_modified=build_modified, results_dir=run_dir)
        genetic.evolve(
            pop_initializer=Initialization(individual_size=num_qubits, pop_size=pop_size),
            crossover_op=PermutationX(probability=crossover_prob, crossover_type=crossover_type),
            mutation_op=PermutationMut(probability=mutation_prob, mutation_type=mutation_type),
            fitness_calculator=QuFitnessCalculator(t_sanchez, init_params, target_state, SPSA(250), device=device),
            selection_op=SelectIndividuals(num_individuals=2),
            k_elitism=KElitism(k=1)
        )

if __name__ == "__main__":
    args_dict = dict(args._get_kwargs())
    main(**args_dict)
