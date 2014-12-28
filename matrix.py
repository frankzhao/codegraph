# Frank Zhao 2014
# Code generation from algorithmic graphs
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random as rand
import string
#from collections import deque

from utility import *
from graph import *

a = [[1,2],
     [3,4]]

b = [[2],[3]]

out = [[0],[0]]

dxg = None

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
                o_new = Node(out[arow][bcol], rand_hash(5)+"x_"+str(arow)+str(bcol))
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
    code = """/* CODEGRAPH GENERATED CODE BEGIN */

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

    for i in range(len(init_values)):
        init_values[i] = "(float) " + str(init_values[i])
    
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
    
    final_node_code = []
    initmem_array = []
    chunkSize = 0
    for i in range(len(paths_from_final)):
        p = paths_from_final[i]
        out = []
        flattened_path = flatten(p)
        
        # Generate initial array for this path
        path_init = []
        for e in flattened_path:
            if e not in ["add", "mul"]:
                path_init.append(e)
        chunkSize = len(path_init)

        print("Initial values: " + str(map(str, path_init)))
        
        path_init_nodes = path_init[:]
        for j in range(len(path_init)):
            initmem_array += ["(float) " + str(path_init[j].value)]
            path_init[j] = "(float) " + str(path_init[j].value)
        #path_values_str = str(path_init).replace('[', "{\n    ").replace(']',"\n}").replace('\'', '')
        #code += "float initmem[" + str(len(path_init)) + "] = " + path_values_str + ";\n\n"

        print("Flattened: " + str(rmap(str, flattened_path)))
        print("Path init: " + str(path_init_nodes))
        
        flattened_rpn = flatten(rpn_to_path(flattened_path))[:]
        
        # Convert RPN flattened path to initmem indexes
        reconstruction_ids = []
        for node in flattened_rpn:
            if node not in [" + ", " * "]:
                reconstruction_ids.append("a[chunkidx + " + str(path_init_nodes.index(node)) + "]")
            else:
                reconstruction_ids.append(node)
        
        final_node_code += ["    c[chunkidx]" + " = " + string.join(reconstruction_ids) + ";\n"]
    
    # Kernel method
    code +="""__global__ void codegraphKernel(float* a, float* c, const int chunkSize) {
    int threadid = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
	// Don't calculate for elements outside of matrix
	if (threadid >= chunkSize)
		return;

    int chunkidx = threadid * chunkSize;
    
    // Calculate
"""
    code += string.join(final_node_code) + "\n"
    code += "}\n"
    
    # Main method
    code += "int main() {\n"
    code += "    const int chunkSize = " + str(chunkSize) + ";\n"
    code += "    const int initSize = " + str(len(initmem_array)) + ";\n"
    code += "    " + array_to_c(initmem_array, "initmem")
    
    # TODO out size should be out_array.length / chunksize
    # TODO specify grid and block size
    code += """
    // Copy to device
	float* dev_initmem = 0;
	float* dev_out = 0;
	float out[2] = {0.f, 0.f};
	cudaMalloc(&dev_initmem, initSize * sizeof(float));
	cudaMalloc(&dev_out, 2 * sizeof(float));
	cudaMemcpy(dev_initmem, initmem, initSize * sizeof(float), cudaMemcpyHostToDevice);

    // Run on device
    codegraphKernel<<<1,initSize>>>(dev_initmem, dev_out, chunkSize);

	// Copy results
	cudaMemcpy(out, dev_out, 2 * sizeof(float), cudaMemcpyDeviceToHost);

    /*
     *Do something with results here
     */

    // Free
 	cudaFree(dev_initmem);
 	cudaFree(dev_out);
"""
            
    print("=== DEBUG ===")
    print(str(rmap(str, finalnodes)))
    
    code += "}" #end main
    code += "\n/* CODEGRAPH GENERATED CODE END */\n"
    
    # write to file
    f = open("main.cu", 'w')
    f.write(code)
    f.close()
    
    return code

matrix(a,b)
#plt.show() # Plot graph
