from enum import Enum

import networkx as nx
import numpy as np

from cores.core.values.constants import GraphFramework


class GraphType(Enum):
    RANDOM = "random"
    CITIES = "cities"


class MockGraphGenerator:
    @staticmethod
    def random(num_nodes: int, num_edges: int, framework: GraphFramework, seed: int = 0):
        if framework == GraphFramework.NETWORKX:
            return MockGraphGenerator.random_networkx(num_nodes, num_edges, seed)
        else:
            raise ValueError(f"Unknown framework: {framework}")

    @staticmethod
    def random_networkx(num_nodes: int, num_edges: int, seed: int = 0):
        # Set the seed
        np.random.seed(seed)

        # Create random graph
        nodes = list(range(num_nodes))
        G = nx.DiGraph()
        G.add_nodes_from(nodes)

        # Add index attribute to nodes
        for node_id in G.nodes():
            G.nodes[node_id]["_id_node"] = node_id
        i = 0
        while i < num_edges:
            head = np.random.choice(nodes)
            tail = np.random.choice(nodes)
            # Check if edge exists
            if G.has_edge(head, tail):
                continue
            G.add_edge(head, tail, _id=i)
            i += 1

        return G

    @staticmethod
    def cities(num_cities: int, framework: GraphFramework, seed: int = 0):
        if not isinstance(framework, GraphFramework):
            framework = GraphFramework(framework)
        if framework == GraphFramework.NETWORKX:
            return MockGraphGenerator.cities_networkx(num_cities, seed)
        else:
            raise ValueError(f"Unknown framework: {framework}")

    @staticmethod
    def cities_networkx(num_cities: int, seed: int = 0):
        G = nx.DiGraph()

        # Add cities as nodes

        np.random.seed(seed)

        # Get alphabet
        alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        assert num_cities < len(alphabet), f"num_cities must be less than {len(alphabet)}"

        cities = {}

        for letter in alphabet[:num_cities]:
            x = np.random.random() * 20
            y = np.random.random() * 20
            cities[letter] = (x, y)

        for city, coords in cities.items():
            G.add_node(city, pos=coords)

        # Add distances between cities as directed edges
        for city1 in cities:
            for city2 in cities:
                if city1 != city2:
                    distance = (
                        (cities[city1][0] - cities[city2][0]) ** 2
                        + (cities[city1][1] - cities[city2][1]) ** 2
                    ) ** 0.5
                    G.add_edge(city1, city2, weight=distance)

        return G
