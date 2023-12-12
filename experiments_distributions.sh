#!/bin/bash 

eps_arr=(0.5 0.1 0.05 0.01)
run=(1 2 3 4 5 6 7 8 9 10)
qubits_array=(3 4 5 6 7 8 9 10 11 12 13 14 15)

# random dense state
for run_idx in ${run[@]}
do
    for eps_val in ${eps_arr[@]}
    do
        for qubit_idx in ${qubits_array[@]}
        do
            python -m experiment_procedures --results-dir results/runs_shell/lognormal --num-qubits $qubit_idx --state-type lognormal --state-params x_points="(0, 1)" s=1 loc=1 scale=2 --eps $eps_val --run-idx $run_idx --verbose --device CPU
            python -m experiment_procedures --results-dir results/runs_shell/bimodal --num-qubits $qubit_idx --state-type bimodal --state-params x_points="(0, 1)" loc_bim1=5 scale_bim1=0.5 loc_bim2=15 scale_bim2=2 --eps $eps_val --run-idx $run_idx --verbose --device CPU
            python -m experiment_procedures --results-dir results/runs_shell/triangular --num-qubits $qubit_idx --state-type triangular --state-params x_points="(0, 1)" c=1 loc=10 scale=5 --eps $eps_val --run-idx $run_idx --verbose --device CPU
            python -m experiment_procedures --results-dir results/runs_shell/normal --num-qubits $qubit_idx --state-type normal --state-params x_points="(0, 1)" loc=10 scale=3.5 --eps $eps_val --run-idx $run_idx --verbose --device CPU
            python -m experiment_procedures --results-dir results/runs_shell/laplace --num-qubits $qubit_idx --state-type laplace --state-params x_points="(0, 1)" loc=10 scale=2 --eps $eps_val --run-idx $run_idx --verbose --device CPU
        done
    done
done