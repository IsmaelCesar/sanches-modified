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
from scipy.stats import (
    lognorm, 
    norm, 
    triang,
    laplace,
)

def get_probability_freqs(
    x_points: np.ndarray,
    num_qubits: int,
    density: str,
    density_params: dict = None):
    assert density in ["lognormal", "laplace", "triangular", "normal", "bimodal"]

    distance = (x_points.max() - x_points.min()) / (2**num_qubits - 1)
    delta = distance * .5

    # temp value to avoid division by zero
    temp_avoid = 1e-7
    if density == "lognormal":
        density_params = {"s": 1, "loc": 1, "scale": 1} if not density_params else density_params
        
        x_values = lognorm.pdf(x_points, **density_params)
        x_values = x_values / np.linalg.norm(x_values + temp_avoid)
        #xcdf_plus = lognorm.cdf(x_points + delta, **density_params)
        #xcdf_minus = lognorm.cdf(x_points - delta, **density_params)
        #x_pmf = xcdf_plus - xcdf_minus
        #x_pmf += 1e-10

    elif density == "normal":
        density_params = {"loc": .5, "scale": 1} if not density_params else density_params
        x_values = norm.pdf(x_points, **density_params)
        x_values = x_values / np.linalg.norm(x_values + temp_avoid)

        #xcdf_plus = norm.cdf(x_points + delta, **density_params)
        #xcdf_minus = norm.cdf(x_points - delta, **density_params)
        #x_pmf = xcdf_plus - xcdf_minus
        #x_pmf += 1e-10

    elif density == "bimodal":
        density_params = { "loc_bim1": 5, "scale_bim1": 2, 
                           "loc_bim2": 15, "scale_bim2": 2} \
                            if not density_params else density_params
        
        x_values_bim1 = norm.pdf(x_points, loc=density_params["loc_bim1"], scale=density_params["scale_bim1"])
        x_values_bim2 = norm.pdf(x_points, loc=density_params["loc_bim2"], scale=density_params["scale_bim2"])
        x_values  = x_values_bim1 + x_values_bim2
        x_values = x_values / np.linalg.norm(x_values + temp_avoid)

        #xcdf_plus_bim1 = norm.cdf(x_points + delta, 
        #                          loc=density_params["loc_bim1"], 
        #                          scale=density_params["scale_bim1"])
        #xcdf_minus_bim1 = norm.cdf(x_points - delta, 
        #                           loc=density_params["loc_bim1"],
        #                           scale=density_params["scale_bim1"])
        #x_pmf_bim1 = xcdf_plus_bim1 - xcdf_minus_bim1

        #xcdf_plus_bim2 = norm.cdf(x_points + delta, 
        #                          loc=density_params["loc_bim2"],
        #                          scale=density_params["scale_bim2"])
        #xcdf_minus_bim2 = norm.cdf(x_points - delta, 
        #                           loc=density_params["loc_bim2"],
        #                           scale=density_params["scale_bim2"])
        #x_pmf_bim2 = xcdf_plus_bim2 - xcdf_minus_bim2

        #x_pmf = x_pmf_bim1 + x_pmf_bim2
        #x_pmf += 1e-10

    elif density == "triangular": 
        density_params = {"c": 1, "loc": 10, "scale": 5} if not density_params else density_params

        x_values = triang.pdf(x_points, **density_params)
        x_values = x_values / np.linalg.norm(x_values + temp_avoid)

        #xcdf_plus = triang.cdf(x_points + delta, **density_params)
        #xcdf_minus = triang.cdf(x_points - delta, **density_params)
        #x_pmf_bim1 = xcdf_plus - xcdf_minus

        #x_pmf = xcdf_plus - xcdf_minus
        #x_pmf += 1e-10

    elif density == "laplace":
        density_params = {"loc": 1, "scale": 1} if not density_params else density_params

        x_values = laplace.pdf(x_points, **density_params)
        x_values = x_values / np.linalg.norm(x_values + temp_avoid)

        #xcdf_plus = laplace.cdf(x_points + delta, **density_params)
        #xcdf_minus =laplace.cdf(x_points - delta, **density_params)
        #x_pmf = xcdf_plus - xcdf_minus
        #x_pmf += 1e-10

    return x_values
