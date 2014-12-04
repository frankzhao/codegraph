import networkx as nx
import matplotlib.pyplot as plt

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

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

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
                graph.nodes.append(n1)
                graph.nodes.append(n2)
                graph.relations.append(e1)

                out[arow][bcol] += a[arow][brow] * b[brow][bcol]

                # Nodes for addition values
                o = Node(out[arow][bcol], "x"+str(arow)+str(bcol))
                o_new = Node(out[arow][bcol], "x."+str(arow)+str(bcol))
                e2 = Relation('add', [o,out_node],[o_new])

                # Add addition nodes
                graph.nodes.append(o)
                graph.relations.append(e2)

    print("Resulting matrix:")
    for e in out:
        print(e)

    #print("Graph:")
    #print("\nNodes:")
    #for node in range(len(graph.nodes)):
    #    print("Node " + str(graph.nodes[node].value))
    #print("\nRelations:")
    #for relation in range(len(graph.relations)):
    #    r = graph.relations[relation]
    #    print("Relation from " + str(map(str, r.in_nodes))
    #          + " to " + str(map(str, r.out_nodes)) + " using " + str(r.method))

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
    #plt.show() # Plot graph
    global dxg
    dxg = G
    return G

# Reconstruct input parameters from networkx graph
def reconstruct(graph):
    final_nodes = [] # Nodes with no outedges
    initial_nodes = []
    for node in graph.nodes_iter():
        #print(str(node) + " " + str(len(graph.out_edges(node))))
        if len(graph.out_edges(node)) == 0:
            final_nodes.append(node)

    # Trace final nodes back to origin
    initial_nodes = []
    find_input_nodes(graph, final_nodes, initial_nodes)
    initial_node_values = nodes_to_values(initial_nodes)
    print(initial_node_values)

# DFS
def find_input_nodes(graph, startNodes, outarray = []):
    for node in startNodes:
        if not graph.predecessors(node):
            if node not in outarray:
                outarray.append(node)
        else:
            # Get the operation for the edge to this node and store it
            find_input_nodes(graph, graph.predecessors(node), outarray)

# Convert node array to their values
def nodes_to_values(nodes):
    for i in range(len(nodes)):
        nodes[i] = nodes[i].name
    return nodes
