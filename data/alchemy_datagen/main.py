from dm_alchemy.types import graphs
from dm_alchemy.types.stones_and_potions import *
from dm_alchemy.ideal_observer import precomputed_maps
from dm_alchemy.types.graphs import Graph
import numpy as np
import torch
import frozendict
from tqdm import tqdm
from itertools import product
import os
import os
from typing import Tuple, Union
import numpy as np
import torch
from torch.utils.data import Dataset
import gdown
import random

def rotations_list_to_index(rotation_list):
    indices = []
    for i, r in enumerate(possible_rotations()):
        for r_ in rotation_list:
            if np.array_equal(r, r_):
                indices.append(i)

    return indices

class AlchemyEnv:
    def __init__(
        self,
        rotation: np.ndarray,
        potion_map: PotionMap,
        stone_map: StoneMap,
        graph: Graph,
    ) -> None:
        self.stone_map = stone_map
        self.potion_map = potion_map
        self.graph = graph
        self.rotation = rotation

    def possible_perceived_stones(self):
        aligned_stones = self.possible_aligned_stones()
        perceived_stones = [unalign(stone, self.rotation) for stone in aligned_stones]

        return perceived_stones
    
    def possible_aligned_stones(self):
        aligned_stones = [
            self.stone_map.apply_inverse(stone) for stone in possible_latent_stones()
        ]

        return aligned_stones

    def possible_perceived_potions(self):
        return possible_perceived_potions()        
        

    def apply_potion(
        self, perceived_stone: PerceivedStone, perceived_potion: PerceivedPotion
    ) -> PerceivedStone:
        # Compute latent stone and potion
        aligned_stone = align(perceived_stone, self.rotation)
        latent_stone = self.stone_map.apply(aligned_stone)
        latent_potion = self.potion_map.apply(perceived_potion)

        # Follow associated edge in graph
        start_node = self.graph.node_list.get_node_by_coords(
            list(latent_stone.latent_coords)
        )

        end_node = None
        for node, v in self.graph.edge_list.edges[start_node].items():
            if v[1].latent_potion() == latent_potion:
                end_node = node
                break

        # If it exist, compute resulting perceived stone
        if end_node != None:
            end_latent_stone = LatentStone(np.array(end_node.coords))
            end_aligned_stone = self.stone_map.apply_inverse(end_latent_stone)
            end_stone = unalign(end_aligned_stone, self.rotation)
        else:
            end_stone = perceived_stone

        return end_stone


class AlchemyFactory:
    def __init__(self) -> None:
        self.stone_maps = possible_stone_maps()
        self.potion_maps = possible_potion_maps(
            precomputed_maps.get_perm_index_conversion()[1]
        )
        self.graphs = self._possible_graphs()
        self.rotations = possible_rotations()

    def _possible_graphs(self):
        constraints = graphs.possible_constraints()
        graphs_distr = graphs.graph_distr(constraints)
        graphs_distr_as_list = list(graphs_distr.items())
        graphs_distr_constraints = [
            graphs.constraint_from_graph(k) for k, _ in graphs_distr_as_list
        ]
        graphs_distr_num_constraints = graphs.get_num_constraints(
            graphs_distr_constraints
        )
        graphs_distr_sorted = sorted(
            zip(
                graphs_distr_as_list,
                graphs_distr_num_constraints,
                graphs_distr_constraints,
            ),
            key=lambda x: (x[2], str(x[1])),
        )
        graphs_list = np.frompyfunc(graphs.Graph, 2, 1)(
            np.array([g[0].node_list for g, _, _ in graphs_distr_sorted], dtype=object),
            np.array([g[0].edge_list for g, _, _ in graphs_distr_sorted], dtype=object),
        )

        return graphs_list

    def unravel_id(self, id):
        return np.unravel_index(
            id,
            (
                len(self.rotations),
                len(self.potion_maps),
                len(self.stone_maps),
                len(self.graphs),
            ),
        )
    
    def get_indices(self, alchemy: AlchemyEnv):
        rot_id = rotations_list_to_index([alchemy.rotation])[0]
        pot_map_id = alchemy.potion_map.index(precomputed_maps.get_perm_index_conversion()[0])
        stone_map_id = alchemy.stone_map.index()
        graph_id = np.where(self.graphs == alchemy.graph)[0].item()

        return (rot_id, pot_map_id, stone_map_id, graph_id)


    def __getitem__(self, id):

        r, p, s, g = self.unravel_id(id)

        return AlchemyEnv(
            self.rotations[r], self.potion_maps[p], self.stone_maps[s], self.graphs[g]
        )


def generate_dataset():
    alchemies = AlchemyFactory()
    rotations = alchemies.rotations
    potion_maps = alchemies.potion_maps
    stone_maps = alchemies.stone_maps
    graphs = alchemies.graphs

    transitions = np.zeros(
        shape=(len(rotations), len(potion_maps), len(stone_maps), len(graphs), 48, 9)
    )

    for i, rotation in tqdm(enumerate(rotations)):
        for j, potion_map in enumerate(potion_maps):
            for k, stone_map in enumerate(stone_maps):
                for l, graph in enumerate(graphs):
                    alchemy = AlchemyEnv(rotation, potion_map, stone_map, graph)

                    context = []
                    potions = alchemy.possible_perceived_potions()
                    stones = alchemy.possible_perceived_stones()

                    for p, s in product(potions, stones):
                        s_ = alchemy.apply_potion(s, p)

                        context += [
                            np.concatenate(
                                [
                                    s.perceived_coords,
                                    [s.reward],
                                    [p.index()],
                                    s_.perceived_coords,
                                    [s_.reward],
                                ],
                                dtype=int,
                                casting="unsafe",
                            )
                        ]

                    transitions[i, j, k, l] = np.stack(context)
    transitions = torch.from_numpy(transitions).flatten(0, 3).int()
    torch.save(transitions, 'transitions.pt')


if __name__ == "__main__":
    generate_dataset()