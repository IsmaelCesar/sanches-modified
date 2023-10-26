#!/bin/bash 

eps_arr=(0.5 0.1 0.05 0.01)
run=(1 2 3 4 5 6 7 8 9 10)
qubits_array=(3 4 5 6 7 8)


# random dense state
for run_idx in ${run[@]}
do
    for eps_val in ${eps_arr[@]}
    do
        for qubit_idx in ${qubits_array[@]}
        do
            python -m experiment_procedures --results-dir results/runs_shell/random_state --num-qubits $qubit_idx --state-type random --state-params complex_state=False --eps $eps_val --run-idx $run_idx --verbose
            python -m experiment_procedures --results-dir results/runs_shell/random_complex_state --num-qubits $qubit_idx --state-type random --state-params complex_state=True --eps $eps_val --run-idx $run_idx --verbose
        done
    done
done

# random sparse state
densities=(0.1 0.2 0.3 0.4 0.5)
for run_idx in ${run[@]}
do
    for eps_val in ${eps_arr[@]}
    do
        for qubit_idx in ${qubits_array[@]}
        do
            for density in ${densities[@]}
            do
                python -m experiment_procedures --results-dir results/runs_shell/random_sparse_state --num-qubits $qubit_idx --state-type random-sparse --state-params complex_state=False density=$density --eps $eps_val --run-idx $run_idx --verbose
                python -m experiment_procedures --results-dir results/runs_shell/random_complex_sparse_state --num-qubits $qubit_idx --state-type random-sparse --state-params complex_state=True density=$density --eps $eps_val --run-idx $run_idx --verbose
            done
        done
    done
done