import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random as rand
import string

from graph import *

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

# def parse_rpn(rpn, out_string):
#     for symbol in rpn:
#         if symbol == "add"
#             return parse_rpn(1::len(rpn))

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