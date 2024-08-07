# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import re
import pandas as pd
import os
import numpy as np
from experiments import ExperimentModule
from experiments import (
    save_plots, 
    write_row, 
    write_opcounts,
    save_qasm_circuit,
    create_dir,
    get_state,
    ParseKvAction
)
from sanchez_ansatz import SanchezAnsatz
from qiskit import transpile
from qiskit_algorithms.optimizers import SPSA
from itertools import product
from experiments.densities import get_probability_freqs
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import logging

parser = ArgumentParser()
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
parser.add_argument("--use-entanglement",
                    required=False,
                    default=False,
                    action='store_true', 
                    help=("Define if the circuit definition should compose and entanglement layer"+
                          " at the end"))
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
parser.add_argument("--device", 
                    type=str,
                    default="GPU",
                    required=False,
                    help="Tell the QiskitAer whether to user CPU or GPU")
parser.add_argument("--verbose", default=False, action="store_true")
args = parser.parse_args()

def run(results_dir: str,
        num_qubits: int, 
        state_type: str,
        eps: float,
        eta: float,
        use_entanglement: bool,
        state_params: dict,
        run_idx: int,
        device: str,
        verbose: bool):
    create_dir(results_dir)

    run_dir = os.path.join(f"{results_dir}", f"run_{run_idx}")

    # creating directories
    create_dir(run_dir)

    if "density" in state_params: 
       density_value = state_params["density"]
       run_dir = os.path.join(run_dir, f"density_{density_value}")
       create_dir(run_dir)

    create_dir(f"{run_dir}/plots")
    create_dir(f"{run_dir}/csv")
    create_dir(f"{run_dir}/op_counts")
    create_dir(f"{run_dir}/circuits")

    # setting up logging
    logger = logging.getLogger("run-logger")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    if verbose:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(f"{results_dir}/run-{run_idx}.log", mode="a+")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    #===================
    logger.info("--"*50)
    logger.info(f"\t\t Experiments for {num_qubits} qubits and {eps} error")
    logger.info("--"*50)

    logger.info("--"*50)
    logger.info("\t\t Running ORIGINAL Method")
    logger.info("--"*50)

    state = get_state(num_qubits, state_type, state_params)

    sanchez = SanchezAnsatz(state, eps, eta, use_entanglement=use_entanglement)

    init_params = sanchez.init_params

    t_sanchez = transpile(sanchez, basis_gates=["cx", "u"])

    em_original = ExperimentModule(
                    t_sanchez,
                    SPSA(maxiter=3000),
                    target_state=state,
                    init_params=init_params,
                    device=device
                )
    em_original.minimize()

    logger.info("--"*50)
    logger.info("\t\t Running MODIFIED Method")
    logger.info("--"*50)

    m_sanchez = SanchezAnsatz(state, eps, build_modified=True)

    init_params = m_sanchez.init_params
    tm_sanchez = transpile(m_sanchez, basis_gates=["cx", "u"])

    em_modified = ExperimentModule(
                    tm_sanchez,
                    SPSA(maxiter=3000),
                    target_state=state,
                    init_params=init_params
                )

    result_modified = em_modified.minimize()
    
    #=======================================
    

    save_plots(em_original, em_modified, f"{run_dir}/plots/plot_{num_qubits}qb_{eps}eps.pdf")

    # writing csv loss progression
    write_row(em_original._loss_progression, file=f"{run_dir}/csv/original_fidloss_{num_qubits}qb_{eps}eps.csv")
    write_row(em_modified._loss_progression, file=f"{run_dir}/csv/modified_fidloss_{num_qubits}qb_{eps}eps.csv")

    # writing csv best point
    best_point_original = em_original.result.x
    best_point_modified = em_modified.result.x
    write_row(best_point_original, file=f"{run_dir}/csv/original_xbest_{num_qubits}qb_{eps}eps.csv")
    write_row(best_point_modified, file=f"{run_dir}/csv/modified_xbest_{num_qubits}qb_{eps}eps.csv")

    # writing op_counts
    write_opcounts(em_original, em_modified, file=f"{run_dir}/op_counts/counts_{num_qubits}qb_{eps}eps.txt")

    # saving circuits
    circuit_save_original = em_original._ansatz.assign_parameters(best_point_original)
    circuit_save_modified = em_modified._ansatz.assign_parameters(best_point_modified)

    save_qasm_circuit(circuit_save_original, file=f"{run_dir}/circuits/original_curcuit_{num_qubits}qb_{eps}eps.qasm")
    save_qasm_circuit(circuit_save_modified, file=f"{run_dir}/circuits/modified_curcuit_{num_qubits}qb_{eps}eps.qasm")

if __name__ == "__main__":
    args_dict = dict(args._get_kwargs())
    run(**args_dict)
