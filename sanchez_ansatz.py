from typing import Optional
import numpy  as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Qubit, Parameter
from qiskit.circuit.library import BlueprintCircuit
from qclib.state_preparation.util.state_tree_preparation import state_decomposition
from qclib.state_preparation.util.angle_tree_preparation import create_angles_tree
from qclib.state_preparation.util import tree_utils

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
        self.params = target_state
        self.global_phase = global_phase
    
    def _check_configuration(self, raise_on_failure: bool = True) -> bool: 
        log2_params = np.log2(len(self.params))
        log2_params_ceil = np.ceil(np.log2(len(self.params)))
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
        circuit = QuantumCircuit(self.num_qubits)

        try: 
            operation = circuit.to_gate(label=self.name)
        
        except:
            operation = circuit.to_instruction(label=self.name)

        self.append(operation, [*self.qubits], [])

