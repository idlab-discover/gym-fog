from collections import defaultdict


def dijsktra(graph, initial, end):
    # shortest paths is a dict of nodes
    # whose value is a tuple of (previous node, weight)
    shortest_paths = {initial: (None, 0)}
    current_node = initial
    visited = set()

    while current_node != end:
        visited.add(current_node)
        destinations = graph.edges[current_node]
        weight_to_current_node = shortest_paths[current_node][1]

        for next_node in destinations:
            weight = graph.weights[(current_node, next_node)] + weight_to_current_node
            if next_node not in shortest_paths:
                shortest_paths[next_node] = (current_node, weight)
            else:
                current_shortest_weight = shortest_paths[next_node][1]
                if current_shortest_weight > weight:
                    shortest_paths[next_node] = (current_node, weight)

        next_destinations = {node: shortest_paths[node] for node in shortest_paths if node not in visited}
        if not next_destinations:
            return "Route Not Possible"
        # next node is the destination with the lowest weight
        current_node = min(next_destinations, key=lambda k: next_destinations[k][1])

    # Work back through destinations in shortest path
    path = []
    weight = 0
    while current_node is not None:
        path.append(current_node)

        next_node = shortest_paths[current_node][0]
        next_weight = shortest_paths[current_node][1]

        if weight < next_weight:
            weight = next_weight

        current_node = next_node

    # Reverse path
    path = path[::-1]
    return path, weight


class Graph():
    def __init__(self):
        """
        self.edges is a dict of all possible next nodes
        e.g. {'X': ['A', 'B', 'C', 'E'], ...}
        self.weights has all the weights between two nodes,
        with the two nodes as a tuple as the key
        e.g. {('X', 'A'): 7, ('X', 'B'): 2, ...}
        """
        self.edges = defaultdict(list)
        self.weights = {}

    def add_edge(self, from_node, to_node, weight):
        # Note: assumes edges are bi-directional
        self.edges[from_node].append(to_node)
        self.edges[to_node].append(from_node)
        self.weights[(from_node, to_node)] = weight
        self.weights[(to_node, from_node)] = weight

'''
if __name__ == '__main__':
    g = Graph()

    g.add_edge('worker1', 'sw-Brugges', 2)
    g.add_edge('worker2', 'sw-Brugges', 2)
    g.add_edge('sw-Brugges', 'sw-Antwerp', 10)
    g.add_edge('worker3', 'sw-Antwerp', 3)
    g.add_edge('worker4', 'sw-Antwerp', 3)
    g.add_edge('sw-Antwerp', 'sw-Ghent', 6)
    g.add_edge('worker5', 'sw-Ghent', 3)
    g.add_edge('worker6', 'sw-Ghent', 3)
    g.add_edge('sw-Bruges', 'sw-Ghent', 15)
    g.add_edge('sw-Bruges', 'sw-Brussels', 10)
    g.add_edge('sw-Brussels', 'sw-Ghent', 12)
    g.add_edge('master', 'sw-Brussels', 1)

    #print(dijsktra(g, 'worker3', 'worker6'))

    path, weight = dijsktra(g, 'worker5', 'master')
    print(path)
    print(weight)
'''