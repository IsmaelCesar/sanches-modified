from typing import Optional, List
import numpy  as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Qubit, Parameter
from qiskit.circuit.library import BlueprintCircuit
from qclib.state_preparation.util.state_tree_preparation import (
    Amplitude,
    state_decomposition
)
from qclib.state_preparation.util.angle_tree_preparation import create_angles_tree
from qclib.state_preparation.util import tree_utils
from qclib.state_preparation.util.angle_tree_preparation import NodeAngleTree

class SanchezAnsatz(BlueprintCircuit):
    """
    This class contains the modified version of the 
    variational quantum circuit for approximated function loading
    proposed by MARIN-SANCHEZ et al. (2021) 
    DOI: 10.1103/PhysRevA.107.022421
    """

    def __init__(
            self, 
            target_state: np.ndarray,
            eps: float,
            eta: float = 4*np.pi,
            name: Optional[str] = "SanchezAnsatz",
            global_phase: Optional[bool] = False
        ):
        """
        Parameters
        ----------
        target_state: The state we desire to approximate
        
        eps: (Epsilon) tolerated error used for defining the approximation
             and at which level the tree will be truncated in the original 
             article

        eta: Hyperparameter defining the supremum point of the second order
             derivative of the log of the function fo be approximated.
             Default value is 4*π
        """
        super().__init__(name=name)
        self.num_qubits = int(np.log2(len(target_state)))
        self.k0 = self._compute_k0(eps, eta)
        self.target_state = target_state
        self.global_phase = global_phase
        self.init_params = []
    
    def _check_configuration(self, raise_on_failure: bool = True) -> bool: 
        log2_params = np.log2(len(self.target_state))
        log2_params_ceil = np.ceil(np.log2(len(self.target_state)))
        valid = True
        if log2_params < log2_params_ceil:
            valid = False
            if(raise_on_failure):
                raise ValueError("target_state size must be a power of 2")
            return valid
        return valid

    @property
    def num_qubits(self) -> int: 
        return self._num_qubits

    @num_qubits.setter
    def num_qubits(self, value: int) -> None: 
        self._num_qubits = value
        self.qregs = [QuantumRegister(value)]

    def _compute_k0(self, eps: float, eta: float = 4*np.pi) -> int:
        """
        Computes the qubit from which the angles are to be clustered
        according to equation 4 of MARIN-SANCHEZ et al. (2021) 
        DOI: 10.1103/PhysRevA.107.022421

        Parameters
        ----------
        eps: (Epsilon) tolerated error used for defining the approximation
            and at which level the tree will be truncated in the original 
            article

        eta: Hyperparameter defining the supremum point of the second order
             derivative of the log of the function fo be approximated.
             Default value is 4*π
        """

        internal_log = 4**(-self.num_qubits) - 96/eta**2 * np.log(1 - eps)
        arg_1 = np.ceil(-1/2 * np.log2(internal_log))

        return int(np.max([arg_1, 2]))

    def _build(self) -> None:
        super()._build()
        circuit = QuantumCircuit(*self.qregs)

        self._build_define(circuit)

        try:
            operation = circuit.to_gate(label=self.name)

        except:
            operation = circuit.to_instruction(label=self.name)

        self.append(operation, [*self.qubits], [])

    def _build_define(self, circuit: QuantumCircuit) -> QuantumCircuit:

        amps = [Amplitude(amp_idx, amp_value) for amp_idx, amp_value in enumerate(self.target_state)]
        state_tree = state_decomposition(self.num_qubits, amps)
        angle_tree = create_angles_tree(state_tree)

        angle_levels = list(range(0, self.num_qubits))

        top_levels = angle_levels[0:self.k0 - 1]

        for level_idx in top_levels:
            level_nodes = []
            tree_utils.subtree_level_nodes(angle_tree, level_idx, level_nodes)
            parameters = self._define_parameters_for_node_list(level_idx, len(level_nodes))
            self._multiplex_parameters(circuit, parameters, circuit.qubits[:level_idx+1])
            
            # saving initial parameters
            self.init_params += [node.angle_y for node in level_nodes]
        
        cluster_levels = angle_levels[self.k0-1:]
        self._clusterize_angles(cluster_levels, angle_tree, circuit)
        
    def _clusterize_angles(self, cluster_levels: List[int], angle_tree: NodeAngleTree, circuit: QuantumCircuit):

        for c_lvl in cluster_levels:
            level_nodes = []
            tree_utils.subtree_level_nodes(angle_tree, c_lvl, level_nodes)

            level_yvalues = [node.angle_y for node in level_nodes]
            c_level = np.mean(level_yvalues)
            
            self.init_params += [c_level]
            circuit.ry(Parameter(name=f"cluster[{c_lvl}]"), c_lvl)

    def _multiplex_parameters(
            self,
            circuit: QuantumCircuit,
            parameters: List[Parameter],
            qubits: List[Qubit],
            reverse: bool = False
        ) -> None:

        mat = [[0.5, 0.5],[0.5, -0.5]]

        if len(parameters) == 1:
            return circuit.ry(parameters[0], qubits[0])
        elif len(parameters) == 2:
            mux_param = np.dot(parameters, mat)
    
            sub_circ = QuantumCircuit(2)
            sub_circ.ry(mux_param[0], 1)
            sub_circ.cx(0, 1)
            sub_circ.ry(mux_param[0], 1)
            sub_circ.cx(0, 1)
            if reverse: 
                sub_circ = sub_circ.reverse_ops()
            circuit.compose(sub_circ, qubits, inplace=True)

            return None

        params_length = len(parameters)
        eye_dim = int(np.log2(params_length))

        block_matrix = np.kron(mat, np.eye(2**(eye_dim-1)))
        mux_param = np.dot(parameters, block_matrix).tolist()

        self._multiplex_parameters(circuit, mux_param[0:params_length//2], qubits[1:])
        circuit.cx(qubits[0], qubits[-1])
        self._multiplex_parameters(circuit, mux_param[params_length//2:], qubits[1:], reverse=True)
        circuit.cx(qubits[0], qubits[-1])

        return None

    def _define_parameters_for_node_list(self, level_idx: int, total_angles: int, label="ry") -> List[Parameter]:
        """
        Creates a parameter list for the list of the nodes in the current level being explored

        Parameters
        ----------
        level_idx: The index of the current tree level

        total_angles: The total number of angles in the current level
        """
        parameters =  [Parameter(name=f"{label}[{level_idx}, {param_idx}]") for param_idx in range(total_angles)]
        return parameters

