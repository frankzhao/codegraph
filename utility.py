# Frank Zhao 2014
# Code generation from algorithmic graphs

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random as rand
import string

from graph import *

# Print graph
def print_graph(graph):
    for node in graph.nodes:
        print(node)
    for edge in graph.relations:
        print(str(map(str,edge.in_nodes)) + str(edge) + str(map(str,edge.out_nodes)))

# Get initial values of graphs (nodes  with no predecessors)
# Sort by depth first, then by value
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
    
    #print(str(rmap(str,path)))
    return path

# Depth of node from root
def depth(node, graph):
    depth = 0
    if graph.successors(node):
        depth += 1
        depth(graph.successors(node)[0])
    else:
        return depth

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
     
def rlength(array, count = 0):
    if hasattr(array, "__iter__"):
        return sum([rlength(elem) for elem in array])
    else:
        count += 1
        return count

def rpathsort(array):
    if not isinstance(array, list) or len(array) == 0:
        return array
    else:
        return [array.pop(0)] + map(rpathsort,sorted(array))

# Print array with newlines
def parray(array):
    for e in array:
        print(e)

# Convert node array to their names
def nodes_to_names(nodes):
    outarray = []
    for i in range(len(nodes)):
        outarray += [nodes[i].name]
    return outarray
    
# Convert node array to their values
def nodes_to_values(nodes):
    outarray = []
    for i in range(len(nodes)):
        outarray += [nodes[i].value]
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
        return token

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

def array_to_c(a, name):
    array_str = str(a).replace('[', "{\n        ").replace(']',"\n    }").replace('\'', '')
    out = "float " + str(name) + "[" + str(len(a)) + "] = " + array_str + ";\n\n"
    return out

def get_final_nodes(graph):
    final_nodes = [] # Nodes with no outedges
    for node in graph.nodes_iter():
        if (len(graph.out_edges(node)) == 0):
            final_nodes += [node]
    return final_nodes

# Takes a graph, returns a list of disconnect graphs
def flood_fill(graph):
    graphs = []
    final_nodes = get_final_nodes(graph)
    coloured_nodes = []
    for node in final_nodes:
        flood_fill_recursion([node], graph, coloured_nodes)
        graphs.append(coloured_nodes)
        coloured_nodes = []
    return graphs

def flood_fill_recursion(nodes, graph, coloured):
    for node in nodes:
        coloured += [node]
        flood_fill_recursion(graph.predecessors(node), graph, coloured)