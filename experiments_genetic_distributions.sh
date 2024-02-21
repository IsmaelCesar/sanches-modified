#!/bin/bash


crossover_operations=("pmx" "cycle" "order")
mutation_operations=("scramble" "inverse" "insert")

num_qubits=15
device=GPU
n_gen=1
pop_size=4

for cx_op in ${crossover_operations[@]}
do
    for mut_op in ${mutation_operations[@]}
    do
        python -m experiments_genetic --results-dir results/genetic/distributions/lognormal_${cx_op}_${mut_op} --num-qubits $num_qubits --eps 0.05 --state-type lognormal --state-params x_points="(0, 1)" s=1 loc=0.1 scale=0.3 --run-idx 1 --n-gen $n_gen --pop-size $pop_size --crossover-type $cx_op --mutation-type $mut_op --device $device
        python -m experiments_genetic --results-dir results/genetic/distributions/bimodal_${cx_op}_${mut_op} --num-qubits $num_qubits --eps 0.05   --state-type bimodal   --state-params x_points="(0, 1)" loc_bim1=0.25 scale_bim1=0.1 loc_bim2=0.75 scale_bim2=0.15 --run-idx 1 --n-gen $n_gen --pop-size $pop_size --crossover-type $cx_op --mutation-type $mut_op --device $device
        python -m experiments_genetic --results-dir results/genetic/distributions/normal_${cx_op}_${mut_op} --num-qubits $num_qubits --eps 0.05    --state-type normal    --state-params x_points="(0, 1)" loc=0.5 scale=3 --run-idx 1 --n-gen $n_gen --pop-size $pop_size --crossover-type $cx_op --mutation-type $mut_op --device $device
        python -m experiments_genetic --results-dir results/genetic/distributions/laplace__${cx_op}_${mut_op} --num-qubits $num_qubits --eps 0.05  --state-type laplace   --state-params x_points="(0, 1)" loc=0.5 scale=0.2 --run-idx 1 --n-gen $n_gen --pop-size $pop_size --crossover-type $cx_op --mutation-type $mut_op --device $device
    done
done
