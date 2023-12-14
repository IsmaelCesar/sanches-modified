# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
from .util import get_random_state, get_sparse_random
from .densities import get_probability_freqs

def get_state(num_qubits: int, state_type: str, state_params: dict) -> np.ndarray:

    if state_type == "random":
        return get_random_state(num_qubits, **state_params)
    elif state_type == "random-sparse": 
        return get_sparse_random(num_qubits, **state_params)
    else:
        # temp value to avoid division by zero
        temp_avoid = 1e-7
        x = np.linspace(*state_params["x_points"], num=2**num_qubits)
        state_params.pop("x_points")
        freqs = get_probability_freqs(x, 
                                      num_qubits, 
                                      state_type,
                                      state_params)
        return freqs / np.linalg.norm(freqs + temp_avoid)