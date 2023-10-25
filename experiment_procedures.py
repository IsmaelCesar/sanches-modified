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
import numpy as np
from experiments import ExperimentModule
from experiments import (
    save_plots, 
    write_row, 
    write_opcounts,
    save_circuit,
    create_dir,
    get_state
)
from sanchez_ansatz import SanchezAnsatz
from qiskit import transpile
from qiskit_algorithms.optimizers import SPSA
from itertools import product
from experiments.densities import get_probability_freqs
from argparse import ArgumentParser, Action
import matplotlib.pyplot as plt
import logging

class ParseKvAction(Action):
   """
   Based on the parsin procedure available in:
   https://gist.github.com/vadimkantorov/37518ff88808af840884355c845049ea
   WHere the last comment has an extended version of the code
   """
   
   def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())

        for v in values:
           try: 
              key, value = v.split("=")
              getattr(namespace, self.dest)[key] = float(value)
           except Exception as e:
              print("Invalid Input: ", v)
              print(e)

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
parser.add_argument("--verbose", default=False, action="store_true")
args = parser.parse_args()

def run(results_dir: str,
        num_qubits: int, 
        state_type: str,
        eps: float,
        eta: float,
        state_params: dict,
        run_idx: int,
        verbose: bool):
    #create_dir(results_dir)

    # setting up logging
    #logger = logging.getLogger("run-logger")
    #formatter = logging.Formatter('%(asctime)s - %(message)s')
    #stream_handler = logging.StreamHandler()
    #stream_handler.setFormatter(formatter)
    #file_handler = logging.FileHandler(f"{results_dir}/run-{run_idx}.log", mode="a+")
    #file_handler.setFormatter(formatter)

    #logger.addHandler(stream_handler)
    #logger.addHandler(file_handler)
    #===================
    print("Results dir: ", results_dir)
    print("Num qubits: ", num_qubits)
    print("State Type: ", state_type)
    print("Eps: ", eps)
    print("Eta: ", eta)
    print("State Params: ", state_params)
    print("Run idx: ", run_idx)
    print("Verbose: ", verbose)

    


if __name__ == "__main__":
    #args_dict = dict(args._get_kwargs())
    #run(**args_dict)
    state = _get_state(7, "lognormal", {"x_points": (0, 20), "s": 1, "loc": 1, "scale": 0.5})
