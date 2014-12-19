# Frank Zhao 2014
# Code generation from algorithmic graphs
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random as rand
import string
#from collections import deque

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
            o = Node(0, rand_hash(5)+"x_init")
            for brow in range(len(b)):

                # Nodes for initial values
                n1 = Node(a[arow][brow], rand_hash(5)+"a"+str(arow)+str(brow))
                n2 = Node(b[brow][bcol], rand_hash(5)+"b"+str(brow)+str(bcol))

                out_node = Node(a[arow][brow] * b[brow][bcol],
                    rand_hash(5) + "v" + str(a[arow][brow] * b[brow][bcol]))

                # Relation for multiplication
                e1 = Relation('mul', [n1,n2],[out_node])

                # Add to graph
                graph.nodes += [n1]
                graph.nodes += [n2]
                graph.relations += [e1]

                out[arow][bcol] += a[arow][brow] * b[brow][bcol]

                # Nodes for addition values
                o_new = Node(out[arow][bcol], rand_hash(5)+"x."+str(arow)+str(bcol))
                e2 = Relation('add', [o,out_node],[o_new])
                o = o_new

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
    global dxg
    dxg = G
    return G

# Reconstruct input parameters from networkx graph
def reconstruct(graph):
    final_nodes = [] # Nodes with no outedges
    initial_nodes = []
    for node in graph.nodes_iter():
        if (len(graph.out_edges(node)) == 0):
            final_nodes += [node]

    # Trace final nodes back to origin
    initial_nodes = []
    all_paths = []
    #print("Final nodes: " + str(rmap(str, final_nodes)))
    for node in final_nodes:
        path = []
        find_paths(graph, [node], initial_nodes, path, all_paths)

    # Sort by output node to try and align input memory
    #all_paths.sort()
    all_paths.sort(key=lambda p: len(p[1])) # sort by path length
    print("Paths found: " + str(len(all_paths)))
    #parray(rmap(str, all_paths))
    
    # CUDA Code
    print(cudagen(all_paths, graph))

# DFS
def find_paths(graph, startNodes, outarray=[], path=[], all_paths=[]):
    #print("Looking at nodes " + str(rmap(str, startNodes)))
    for node in startNodes:
        if not graph.predecessors(node):
            if node not in outarray:
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
            find_paths(graph, graph.predecessors(node), outarray, path, all_paths)

testedge = None
testnode = None
def cudagen(paths, graph):
    code = """
/* CODEGRAPH GENERATED CODE BEGIN */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

"""

    # Magic happens here
    # Create initial values
    initial_dict = get_initial_values(paths, graph)
    
    # generate initial_node -> array_pos dict
    init_values = []
    init_array_dict = {}
    counter = 0
    for d in initial_dict.keys():
        init_values.append(initial_dict[d])
        init_array_dict[d] = counter
        counter += 1
    
    code += "float *initmem = " + str(init_values) + ";\n"
    
    # Find out how to generate final nodes
    finalnodes = []
    for node in graph.nodes():
        if not graph.out_edges(node):
            finalnodes.append(node)

    paths_from_final = []
    for node in finalnodes:
        path = rmap_nodes_args(get_path_for_node(node, graph), get_path_for_node, graph)
        paths_from_final.append(path)
        print(str(rmap(str, path))) # This generates prefix notation

    for i in range(len(paths_from_final)):
        p = paths_from_final[i]
        out = []
        flattened_path = flatten(p)

        print("Flattened: " + str(rmap(str, flattened_path)))
        print("!!!")
        code += "\n float " + finalnodes[i].name + " = " + string.join(flatten(rpn_to_path(flattened_path))) + ";\n"

    # for node in finalnodes:
    #     code += "float " + node.name + " = "
    #     inedges = graph.in_edges(node)
    #     for i in range(len(inedges)):
    #         edge = inedges[i]
            
    #         global testnode, testedge
    #         testnode = node
    #         testedge = edge
            
    #         if i > 0:
    #             # Don't add op sign for first
    #             op = graph.edge[edge[0]][edge[1]]["method"]
    #             if op == "mul":
    #                 code += " * "
    #             elif op == "add":
    #                 code += " + "
    #             else:
    #                 print("Operation: " + op + " not found")
    #         code += edge[0].name
    #     code += ";\n"
            
    print("=== DEBUG ===")
    print(str(rmap(str, finalnodes)))
    
    code += "\n/* CODEGRAPH GENERATED CODE END */\n"
    return code
    
def get_initial_values(paths, graph):
    values = {}
    for path in paths:
        node = path[0]
        if not graph.predecessors(node):
            values[node] = node.value
    return values
    
def get_path_for_node(node, graph):
    if node in ["add", "mul"]:
        return node
    path = []
    #print("Finding path to generate node " + node.name)
    if not graph.in_edges(node):
        return node
    for i in range(len(graph.in_edges(node))):
        edge = graph.in_edges(node)[i]
        if i == 0:
            method = graph.edge[edge[0]][edge[1]]["method"]
            path.append(method)
        path.append(edge[0])
    
    #print(str(rmap(str,path)))
    return path

### UTILITY ###

# Recursive map
def rmap(f, array):
    if hasattr(array, "__iter__"):
        return [rmap(f, elem) for elem in array]
    else:
        return f(array)
        
def rfmap(array, *f):
    if hasattr(array, "__iter__"):
        return [rmap(elem, f) for elem in array]
    else:
        for i in range(len(f)):
            array[i] = f(array[i])
        return array

def rmap_args(array, f, *args):
    if hasattr(array, "__iter__"):
        return [rmap_args(elem, f, *args) for elem in array]
    else:
        return f(array, *args)
        
def rmap_nodes_args(array, f, graph):
    if hasattr(array, "__iter__"):
        return [rmap_nodes_args(elem, f, graph) for elem in array]
    elif isinstance(array, Node) and graph.in_edges(array):
        return rmap_nodes_args(get_path_for_node(array, graph), get_path_for_node, graph)
    else:
        return f(array, graph)

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

def rand_str():
    return str(rand.getrandbits(16))

def rand_hash(n):
    return ''.join(rand.choice(string.ascii_lowercase) for i in range(n))
    
def method_to_op(s):
    if s == "mul":
        return " * "
    elif s == "add":
        return " + "
    else:
        print("Operation: " + op + " not found")
        return " (!) "

def rpn_to_path(rpn):
    #print("! " + out_string)
    #print(str(rpn) + " " + out_string)
    token = rpn.pop(0)
    if token == "add":
        return [rpn_to_path(rpn), " + ", rpn_to_path(rpn)]
    elif token == "mul":
        return [rpn_to_path(rpn), " * ", rpn_to_path(rpn)]
    else:
        return token.name

def parse_rpn(rpn):
    token = rpn.pop(0)
    if token == "add":
        return parse_rpn(rpn) + parse_rpn(rpn)
    if token == "mul":
        return parse_rpn(rpn) * parse_rpn(rpn)
    else:
        return int(token)

def flatten(S):
    if S == []:
        return S
    if isinstance(S[0], list):
        return flatten(S[0]) + flatten(S[1:])
    return S[:1] + flatten(S[1:])

matrix(a,b)
plt.show() # Plot graph