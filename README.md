# sanches-modified
This repo contains a modification to the ansatz proposed by MARIN-SANCHEZ et al. (2021) DOI: 10.1103/PhysRevA.107.022421

The result data can be found [here](https://drive.google.com/file/d/1cryh-JSzXGnotny56w4wosTiR1Sso-R2/view?usp=sharing), and shoud be extracted 
inside the `results` directory.


When running the `script_experiment.py` the config file it utilizes `script_experiment_config.yml` must follow the format as in:
```yml
eps: 0.05
num_qubits: 15
n_runs: 2
maxiter: 2
distributions:
  - dist_type: "normal"
    dist_params: 
      x_points: [0, 1]
      loc: 0.5
      scale: 0.3
  - dist_type: "lognormal"
    dist_params: 
      x_points: [0, 1]
      s: 1
      loc: 0.1
      scale: 0.3
  - dist_type: "bimodal"
    dist_params: 
      x_points: [0, 1]
      loc_bim1: 0.25
      scale_bim1: 0.1
      loc_bim2: 0.75
      scale_bim2: 0.15
  - dist_type: "laplace"
    dist_params: 
      x_points: [0, 1]
      loc: 0.5
      scale: 0.2
```
