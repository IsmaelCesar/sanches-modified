#!/bin/bash

density_opts=(0.2 0.3 0.5)

for density in ${density_opts}
do
    python -m experiments_genetic --results-dir results/genetic --num-qubits 12 --eps 0.05 --state-type random-sparse --state-params complex_state=False density=$density --eps 0.05 --run-idx 1 --n-gen 10 --pop-size 10 --crossover-type cycle --mutation-type scramble --device GPU
done
