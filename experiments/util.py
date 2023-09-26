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
import csv
import pickle as pkl
from qiskit import QuantumCircuit
from .experiment_module import ExperimentModule
import matplotlib.pyplot as plt

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