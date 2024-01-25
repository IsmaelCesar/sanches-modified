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
import os
import csv
import pickle as pkl
from qiskit import QuantumCircuit
from scipy import sparse
from .experiment_module import ExperimentModule
import matplotlib.pyplot as plt
from argparse import Action

# -- procedures for writing data

def save_plots(
        em_original: ExperimentModule,
        em_modified: ExperimentModule,
        file: str = "results/plot.pdf"
    ):
    
    plt.plot(em_original._loss_progression, "--", marker="o", markevery=500, label="Original")
    plt.plot(em_modified._loss_progression, "-.", marker="+", markevery=500, label="Modified")
    plt.ylabel("fid loss")
    plt.xlabel("iterations")
    plt.legend(loc="best")
    plt.savefig(file)
    plt.clf()
    plt.close()

def write_row(data, file="results/data.csv", mode: str ="w+") -> None:

    if os.path.exists(file):
        mode = "a+"

    with open(file, mode) as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(data)

def write_opcounts(
        em_oiginal: ExperimentModule, 
        em_modified: ExperimentModule, 
        file="count_opts.txt", mode="w+"
    ):

    line = "--"*20

    with open(file, mode) as txt_file:
        txt_file.write(line + "\n")
        txt_file.write("\t\t Operation Count \n")
        txt_file.write(line+"\n")
        txt_file.write(f"Op count ORIGINAL: {em_oiginal._ansatz.count_ops()}\n")
        txt_file.write(f"Op count MODIFIED: {em_modified._ansatz.count_ops()}\n")

        txt_file.write(line+"\n")
        txt_file.write("\t\t Depth \n")
        txt_file.write(line+"\n")
        txt_file.write(f"Op count ORIGINAL: {em_oiginal._ansatz.depth()}\n")
        txt_file.write(f"Op count MODIFIED: {em_modified._ansatz.depth()}\n")

def save_circuit(circuit: QuantumCircuit, file="circuit.pkl"): 
    with open(file, "wb") as f: 
        pkl.dump(circuit, f)

def load_circuit(file: str) -> QuantumCircuit:
    with open(file, "rb") as f:
        circuit = pkl.load(f)
    return circuit

def create_dir(dir_name: str): 
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

def get_random_state(num_qubits, seed=7, complex_state=False):
    rng = np.random.default_rng(seed)
    state = rng.random(2**num_qubits)

    if complex_state:
        state = rng.random(2**num_qubits) + 1j*rng.random(2**num_qubits)

    return state /np.linalg.norm(state)

def _get_sparse_array(num_qubits: int, density: float, rng: np.random.Generator, complex_state=False): 
    dok_matrix = sparse.random(2**num_qubits, 
                               1, 
                               density=density, 
                               random_state=rng,
                               format="dok")
    dtype = np.float32
    if complex_state:
        dtype = np.complex64
    
    sparse_arr = np.zeros((2**num_qubits,), dtype=dtype)
    for key, value in dok_matrix.items():
        if not complex_state:
            sparse_arr[key[0]] = value
        else:
            sparse_arr[key[0]] = value + 1j*rng.random()
            
    return sparse_arr

def get_sparse_random(num_qubits, density, seed=7, complex_state=False):
    rng = np.random.default_rng(seed)
    state = _get_sparse_array(num_qubits, density, rng, complex_state=complex_state)
    return state / np.linalg.norm(state)

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
              if key == "complex_state": 
                getattr(namespace, self.dest)[key] = bool(value)
              elif key == "x_points":
                 getattr(namespace, self.dest)[key] = tuple([int(digits) for digits in re.findall(r"\d+", value) ])
              else:               
                getattr(namespace, self.dest)[key] = float(value)
           except Exception as e:
              print("Invalid Input: ", v)
              print(e)
