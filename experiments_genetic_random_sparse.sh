#!/bin/bash

density_opts=(0.2 0.3 0.5)
crossover_operations=("pmx" "cycle" "order")
mutation_operations=("scramble" "inverse" "insert")

num_qubits=4
device=GPU
n_gen=1
pop_size=4

for cx_op in ${crossover_operations[@]}
do
    for mut_op in ${mutation_operations[@]}
    do
        for density in ${density_opts[@]}
        do
            python -m experiments_genetic --results-dir results/genetic/random_sparse_${cx_op}_${mut_op} --num-qubits $num_qubits --eps 0.05 --state-type random-sparse --state-params complex_state=False density=$density --run-idx 1 --n-gen $n_gen --pop-size $pop_size --crossover-type $cx_op --mutation-type $mut_op --device $device
        done
    done
done
