from typing import Optional, Union, List
import numpy  as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
from qiskit.circuit.library import BlueprintCircuit
from qclib.state_preparation.util.state_tree_preparation import (
    Amplitude,
    state_decomposition
)
from qclib.state_preparation.util.angle_tree_preparation import create_angles_tree
from qclib.state_preparation.util import tree_utils
from qclib.state_preparation.util.angle_tree_preparation import NodeAngleTree
from qiskit.circuit.library import RZGate, RYGate, CXGate, CZGate, CYGate

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
            eta: Optional[float] = 4*np.pi,
            name: Optional[str] = "SanchezAnsatz",
            global_phase: Optional[bool] = False,
            build_modified: Optional[bool] = False
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
        self.build_modified = build_modified
    
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

            # saving initial parameters
            init_params_y = [node.angle_y for node in level_nodes]
            init_params_z = [node.angle_z for node in level_nodes]

            if any(init_params_y):
                self.init_params += init_params_y
                parameters_y = self._define_parameters_for_node_list(level_idx, len(level_nodes))

                mux_circuit = self._multiplex_parameters(RYGate, parameters_y)
                circuit.compose(mux_circuit, circuit.qubits[:level_idx+1], inplace=True)

            if any(init_params_z):
                self.init_params += init_params_z
                parameters_z = self._define_parameters_for_node_list(level_idx, len(level_nodes), label="rz")
                mux_circuit = self._multiplex_parameters(RZGate, parameters_z)
                circuit.compose(mux_circuit, circuit.qubits[:level_idx+1], inplace=True)

        cluster_levels = angle_levels[self.k0-1:]
        
        if self.build_modified:
            self._cluster_modified(cluster_levels, angle_tree, circuit)
        else: 
            self._clusterize_angles(cluster_levels, angle_tree, circuit)

    def _clusterize_angles(self, cluster_levels: List[int], angle_tree: NodeAngleTree, circuit: QuantumCircuit) -> None:
        """
        Computes the representative angles for each level by computing centroid from the 
        angles in each level

        Parameters
        ----------
        cluster_levels: Level of the angle tree from which the angles are to be clustered

        angle_tree: Tree containing the angles to be used in the rotations

        circuit: The quantum circuit
        """

        for c_lvl in cluster_levels:
            level_nodes = []
            tree_utils.subtree_level_nodes(angle_tree, c_lvl, level_nodes)

            level_yvalues = [node.angle_y for node in level_nodes]
            level_zvalues = [node.angle_z for node in level_nodes]

            yc_level = np.mean(level_yvalues)
            zc_level = np.mean(level_zvalues)
            
            self.init_params += [yc_level]
            circuit.ry(Parameter(name=f"cluster_y[{c_lvl}]"), c_lvl)

            self.init_params += [zc_level]
            circuit.rz(Parameter(name=f"cluster_z[{c_lvl}]"), c_lvl)

    def _cluster_modified(self, cluster_levels: List[int], angle_tree: NodeAngleTree, circuit: QuantumCircuit) -> None:

        for level_idx, c_lvl in enumerate(cluster_levels):

            level_nodes = []
            tree_utils.subtree_level_nodes(angle_tree, c_lvl, level_nodes)

            level_yvalues = [node.angle_y for node in level_nodes]
            level_zvalues = [node.angle_z for node in level_nodes]

            if any(level_yvalues):
                slice_y = self._slice_angle_list(level_yvalues, c_lvl, level_idx)
                clusters_y = np.mean(slice_y, axis=1)
                clusters_yparams = self._define_parameters_for_node_list(level_idx, len(clusters_y), label="clu_ry")

                qubit_step = int(np.log2(len(clusters_yparams)))
                # multiplexing ry values
                mux_circuit = self._multiplex_parameters(RYGate, clusters_yparams)
                circuit.compose(mux_circuit, circuit.qubits[cluster_levels[0]:c_lvl + 1], inplace=True)

                # saving init params
                self.init_params += clusters_y.tolist()

            if any(level_yvalues):
                slice_z = self._slice_angle_list(level_zvalues, c_lvl, level_idx)
                clusters_z = np.mean(slice_z, axis=1)
                clusters_zparams = self._define_parameters_for_node_list(level_idx, len(clusters_z), label="clu_rz")
                
                qubit_step = int(np.log2(len(clusters_zparams)))
                # multiplexing rz values
                mux_circuit = self._multiplex_parameters(RZGate, clusters_zparams)
                circuit.compose(mux_circuit, circuit.qubits[cluster_levels[0]:c_lvl + 1], inplace=True)

                # saving init params
                self.init_params += clusters_z.tolist()

    def _slice_angle_list(self, level_values: List[float], current_lvl: int, level_idx: int) -> List[List[float]]:
        start = 0
        slice_step = 2**(current_lvl - level_idx)
        slices = []
        while level_values[start:slice_step]:
            slices += [level_values[start:slice_step]]

            start = slice_step
            slice_step += 2**(current_lvl - level_idx)
        
        return slices

    def _multiplex_parameters(
            self,
            operation: Union[RYGate, RZGate],
            parameters: List[Parameter],
            c_operation: Union[CXGate, CZGate, CYGate] = CXGate
        ) -> None:

        mat = [[0.5, 0.5],[0.5, -0.5]]

        if len(parameters) == 1:
            circuit = QuantumCircuit(1)
            circuit.append(operation(parameters[0]), [0])
            return circuit
        elif len(parameters) == 2:
            mux_param = np.dot(parameters, mat)

            circuit = QuantumCircuit(2)
            circuit.append(operation(mux_param[0]), [1])
            circuit.append(c_operation(), [0, 1])
            circuit.append(operation(mux_param[1]), [1])
            circuit.append(c_operation(), [0, 1])
            return circuit

        num_qubits = int(np.log2(len(parameters))) + 1
        circuit = QuantumCircuit(num_qubits)
        q_index = list(range(num_qubits))

        control = q_index[0]
        target = q_index[num_qubits-1]

        params_length = len(parameters)
        eye_dim = int(np.log2(params_length))

        block_matrix = np.kron(mat, np.eye(2**(eye_dim-1)))
        mux_param = np.dot(parameters, block_matrix).tolist()

        #multiplexor
        mux_circ = self._multiplex_parameters(operation, mux_param[:params_length//2])
        circuit.compose(mux_circ, q_index[0:-1], inplace=True)

        circuit.append(c_operation(), [control, target])

        mux_circ = self._multiplex_parameters(operation, mux_param[params_length//2:])
        circuit.compose(mux_circ.reverse_ops(), q_index[0:-1], inplace=True)

        circuit.append(c_operation(), [control, target])

        return circuit

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

