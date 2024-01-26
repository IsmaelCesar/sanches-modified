
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
import copy 
from typing import Tuple, List, Dict, Union

def _combine_second_parent_segment(child: np.ndarray, parent: np.ndarray, end: int):
    """
    auxiliary method of the _apply_order_1_x for combining the second parent segment
    """
    
    parent_idx = child_idx = end+1

    if parent_idx > len(parent) - 1:
        parent_idx = 0

    if child_idx > len(child) - 1 : 
        child_idx = 0

    for _ in range(len(child)):

        if parent[parent_idx] not in child:
            child[child_idx] = parent[parent_idx]
            child_idx += 1

            if child_idx > len(child) - 1:
                child_idx = 0

        parent_idx += 1

        if parent_idx > len(parent) - 1:
            parent_idx = 0

    return child

def _apply_order_1_x(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Effectively applies order 1 crossover on the cromossomes of the parents
    """
    cromossome_size = len(parent1)
    start = np.random.randint(0, cromossome_size-2)
    end = np.random.randint(start+1, cromossome_size-1)

    child_1 = np.zeros(cromossome_size, dtype=int) -1
    child_2 = np.zeros(cromossome_size, dtype=int) -1
    
    # first child
    child_1[start: end+1] = copy.deepcopy(parent1[start:end+1])
    child_1 = _combine_second_parent_segment(child_1, parent2, end)

    #second child:
    child_2[start: end+1] = copy.deepcopy(parent2[start:end+1])
    child_2 = _combine_second_parent_segment(child_2, parent1, end)

    return child_1, child_2

def _combine_child_parent_pmx(child: np.ndarray, parent1: np.ndarray, parent2: np.ndarray, start: int , end: int) -> np.ndarray:
    """
    Auxiliary procedure for the _apply_pmx method
    """

    #check whith element have not been copied
    no_child = []
    for seg_idx in range(start, end + 1): 
        if parent2[seg_idx] not in child: 
            no_child += [(seg_idx, parent2[seg_idx])]
    
    for p2_idx, p2_value  in no_child:
        p1_in_p2_idx = np.where(parent2 == parent1[p2_idx])[0][0]
        if child[p1_in_p2_idx] == -1:
                child[p1_in_p2_idx] = p2_value
        else:
            # look for another empty element on the child individual
            while child[p1_in_p2_idx] != p2_value:
                p1_in_p2_idx = np.where(parent2 == parent1[p1_in_p2_idx])[0][0]
                if child[p1_in_p2_idx] == -1:
                    child[p1_in_p2_idx] = p2_value

    for p2_idx, p2_element in enumerate(parent2):
       if p2_element not in child:
           # get the first index marked as empty
           empty_element_idx = np.where(child == -1)[0][0]
           child[empty_element_idx] = p2_element

    return child

def _apply_pmx(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]: 
    """
    Effectively apply pmx crossover on the cromossome of the parents
    """
    cromossome_size = len(parent1)
    start = np.random.randint(0, cromossome_size - 2)
    end = np.random.randint(start + 1, cromossome_size - 1)

    child_1 = np.zeros(cromossome_size, dtype=int) -1
    child_2 = np.zeros(cromossome_size, dtype=int) -1

    child_1[start: end + 1] = copy.deepcopy(parent1[start: end + 1])
    child_1 = _combine_child_parent_pmx(child_1, parent1, parent2, start, end)

    child_2[start: end + 1] = copy.deepcopy(parent2[start: end + 1])
    child_2 = _combine_child_parent_pmx(child_2, parent2, parent1, start, end)

    return child_1, child_2

def _cycle_child(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray: 
    """
    auxiliary procedure for creating one child of the procedure _apply_cycle_x
    """
    child = np.zeros(len(parent1), dtype=int) -1

    for p1_idx, _ in enumerate(parent1):
        if parent2[p1_idx] not in child:
            # first element of the cycle
            cycle_idx = p1_idx
            init_cycle = child[cycle_idx] = parent2[cycle_idx]
            
            #creating subsequent cycle
            while parent2[cycle_idx] != init_cycle:
                # position of the current element of the cycle of parent1 in parent2
                cycle_idx = np.where(parent1 == parent2[cycle_idx])[0][0]
                child[cycle_idx] = parent2[cycle_idx]

    return child


def _apply_cycle_x(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Effectively applies the cycle crossover
    """
    child_1 = _cycle_child(parent1, parent2)
    child_2 = _cycle_child(parent2, parent1)
    return child_1, child_2

def _build_edge(edge_idx: int, chromosome_size: int) -> List[int]:

    edge = [0, 0]
    edge[0] = edge_idx - 1
    edge[1] = edge_idx + 1 if edge_idx + 1 < chromosome_size - 1 else 0
    return edge

def _construct_edge_table(parent1: np.ndarray, parent2: np.ndarray) -> Dict[int, List[int]]:
    table = {}

    for p1_el in parent1:
        element_p1_idx = np.where(parent1 == p1_el)[0][0]
        edge_p1 = _build_edge(element_p1_idx, len(parent1))
        elements_edge_p1 = parent1[edge_p1].tolist()

        element_p2_idx = np.where(parent2 == p1_el)[0][0]
        edge_p2 = _build_edge(element_p2_idx, len(parent1))
        elements_edge_p2 = parent2[edge_p2].tolist()

        table[p1_el] = sorted(elements_edge_p1 + elements_edge_p2)

    return table

def _remove_item_references(current_item: int, edge_table:  Dict[int, List[int]]) -> Dict[int, List[int]]:
    """
    removes all current item references from edge table and returns an updated edge table
    """
    edge_table.pop(current_item)

    for k in edge_table.keys(): 
        if current_item in edge_table[k]:
            item_count = edge_table[k].count(current_item)
            # remove each of the occurences of current item on the edge list
            for _ in range(item_count):
                edge_table[k].remove(current_item)

    return edge_table

def _find_common_edges(current_item_edges: List[int]) -> Union[int, None]:
    """
    If the current_item_edges has duplicate items it means it has a common edge.
    Returns integer if it finds common edge items and None otherwise
    """
    # count items
    item_counts = {}
    for ci_edge in current_item_edges:
        item_counts[ci_edge] = 0
        for cj_edge in current_item_edges:
            # count repetitions only on different indices
            if ci_edge == cj_edge:
                item_counts[ci_edge] += 1

    # checks if there is a common edge in item counts
    current_value = None
    for k, v in item_counts.items():
        if v >= 2: 
            current_value = k
            break
    return current_value

def _find_shortest_list(current_item_edges: List[int], edge_table: Dict[int, List[int]]) -> Union[int, None]:
    edge_size = {}
    # for each element in edge list get the edge count
    for ci_edge_el in current_item_edges:
        edge_size[ci_edge_el] = len(edge_table[ci_edge_el])

    current_item = None

    #verifica empates
    sorted_size = dict(sorted(edge_size.items(), key = lambda x : x[1]))
    values = list(sorted_size.values())
    if np.all(values == values[0]):
        current_item = np.random.choice(list(sorted_size.keys()))
    else:
        sorted_size_keys = list(sorted_size.keys())
        current_item = sorted_size_keys[0]

    return current_item

def _choose_next_item(current_item_edges: List[int], edge_table: Dict[int, List[int]]) -> int:
    
    if current_item_edges: 
        current_item = _find_common_edges(current_item_edges)
        if current_item is None:
            current_item = _find_shortest_list(current_item_edges, edge_table)

        # randomly chose another element from edge table if nothing works
        if current_item is None: 
            edge_table_keys = list(edge_table.keys())
            current_item = np.random.choice(edge_table_keys)
    else:
        # ramdomly pick another key from edge table if 
        # the current_item_edges is an empty list
        edge_table_keys = list(edge_table.keys())
        current_item = np.random.choice(edge_table_keys)
    return current_item

def _construct_child_edge_x(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    
    edge_table = _construct_edge_table(parent1, parent2)
    child = np.zeros(len(parent1), dtype=int) -1

    current_item = np.random.choice(list(edge_table.keys()))
    
    for child_idx in range(len(child)): 
        child[child_idx] = current_item
        current_item_edges = edge_table[current_item]
        edge_table = _remove_item_references(current_item, edge_table)
        if edge_table:
            current_item = _choose_next_item(current_item_edges, edge_table)

    return child


def _apply_edge_x(parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Effectively applies edge crossover
    """
    child_1 = _construct_child_edge_x(parent1, parent2)
    child_2 = _construct_child_edge_x(parent2, parent1)

    return child_1, child_2

class PermutationX:

    def __init__(self, probability: float = 0.5, crossover_type: str = "order"):
        self.crossover_options = ["pmx", "edge", "cycle", "order" ]

        assert crossover_type in self.crossover_options

        self.probability = probability
        self.crossover_type = crossover_type

    def order_1(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rand_prob = np.random.rand()
        if rand_prob <= self.probability:
          return _apply_order_1_x(parent1, parent2)

        return copy.deepcopy(parent1), copy.deepcopy(parent2)

    def pmx(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]: 
        rand_prob = np.random.rand()
        if rand_prob <= self.probability:
            return _apply_pmx(parent1, parent2)
        return copy.deepcopy(parent1), copy.deepcopy(parent2)

    def cycle(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rand_prob = np.random.rand()
        if rand_prob <= self.probability:
            return _apply_cycle_x(parent1, parent2)
        return copy.deepcopy(parent1), copy.deepcopy(parent2)


    def edge(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rand_prob = np.random.rand()
        if rand_prob <= self.probability:
            return _apply_edge_x(parent1, parent2)
        return copy.deepcopy(parent1), copy.deepcopy(parent2)

    def apply(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        if self.crossover_type == "order":
            return self.order_1(parent1, parent2)
        elif self.crossover_type == "pmx":
            return self.pmx(parent1, parent2)
        elif self.crossover_type == "cycle":
            return self.cycle(parent1, parent2)
        elif self.crossover_type == "edge": 
            return self.edge(parent1, parent2)
