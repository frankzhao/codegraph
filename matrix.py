import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

a = [[1,2],
     [3,4]]

b = [[2],[3]]

out = [[0],[0]]

dxg = None

# Graph classes
class Graph:
    def __init__(self, nodes=[], relations=[]):
        self.nodes = nodes
        self.relations = relations

class Node:
    def __init__(self, value=None, name=None):
        self.value = value
        self.name = name

    def __str__(self):
        if self.value is None:
            return str(self.name)
        elif self.name is None:
            return str(self.value)
        else:
            return str(self.name)

class Relation:
    def __init__(self, method=None, in_nodes=[], out_nodes=[]):
        self.method = method
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes

graph = Graph()

def matrix(a,b):
    for arow in range(len(a)):
        for bcol in range(len(b[0])):
            for brow in range(len(b)):

                # Nodes for initial values
                n1 = Node(a[arow][brow], "a"+str(arow)+str(brow))
                n2 = Node(b[brow][bcol], "b"+str(brow)+str(bcol))

                out_node = Node(a[arow][brow] * b[brow][bcol])

                # Relation for multiplication
                e1 = Relation('mul', [n1,n2],[out_node])

                # Add to graph
                graph.nodes += [n1]
                graph.nodes += [n2]
                graph.relations += [e1]

                out[arow][bcol] += a[arow][brow] * b[brow][bcol]

                # Nodes for addition values
                o = Node(out[arow][bcol], "x"+str(arow)+str(bcol))
                o_new = Node(out[arow][bcol], "x."+str(arow)+str(bcol))
                e2 = Relation('add', [o,out_node],[o_new])

                # Add addition nodes
                graph.nodes += [o]
                graph.relations += [e2]

    print("Resulting matrix:")
    for e in out:
        print(e)

    G = generate_graph(graph)
    reconstruct(G)

# Conversion from my graph model to networkx
def generate_graph(graph):
    G = nx.DiGraph()
    for node in graph.nodes:
        G.add_node(node)

    for relation in graph.relations:
        for inNode in relation.in_nodes:
            for outNode in relation.out_nodes:
                G.add_edge(inNode, outNode)
                G.edge[inNode][outNode]['method']=str(relation.method)

    edgelabels = nx.get_edge_attributes(G, 'method')
    # swap key values for rendering
    dict( zip(edgelabels.values(),edgelabels.keys()) )
    pos = nx.graphviz_layout(G, prog = 'dot')
    nx.draw(G, pos, node_size=1000)
    nx.draw_networkx_edge_labels(G, pos, edge_labels = edgelabels)
    nx.draw_networkx_labels(G, pos)
    plt.show() # Plot graph
    global dxg
    dxg = G
    return G

# Reconstruct input parameters from networkx graph
def reconstruct(graph):
    final_nodes = [] # Nodes with no outedges
    initial_nodes = []
    for node in graph.nodes_iter():
        if len(graph.out_edges(node)) == 0:
            final_nodes += [node]

    # Trace final nodes back to origin
    initial_nodes = []
    all_paths = []
    print("Final nodes: " + str(rmap(str, final_nodes)))
    for node in final_nodes:
        path = []
        find_input_nodes(graph, [node], initial_nodes, path, all_paths)

    # Sort by output node to try and align input memory
    all_paths.sort()
    print("Paths found: " + str(len(all_paths)))
    parray(rmap(str, all_paths))

# DFS
def find_input_nodes(graph, startNodes, outarray=[], path=[], all_paths=[]):
    print("Looking at nodes " + str(rmap(str, startNodes)))
    for node in startNodes:
        if not graph.predecessors(node):
            if node.name not in nodes_to_names(outarray):
                print(node.name)
                outarray += [node]
                all_paths += [[node, path[::-1]]]
        else:
            path += graph.predecessors(node)
            # Get the operation for the edge to this node and store it
            edges = graph.in_edges(node)
            methods = []
            for edge in edges:
                method = graph.edge[edge[0]][node]["method"]
                methods += [method]
            path += list(np.unique(methods))
            find_input_nodes(graph, graph.predecessors(node), outarray, path, all_paths)

### UTILITY ###

# Recursive map
def rmap(f, array):
    if hasattr(array, "__iter__"):
        return [rmap(f, elem) for elem in array]
    else:
        return f(array)

# Print array with newlines
def parray(array):
    for e in array:
        print(e)

# Convert node array to their values
def nodes_to_names(nodes):
    outarray = []
    for i in range(len(nodes)):
        outarray += [nodes[i].name]
    return outarray

#matrix(a,b)

