# Frank Zhao 2014
# Code generation from algorithmic graphs

import networkx as nx
import pygraphviz as gv
import matplotlib.pyplot as plt
import numpy as np
import random as rand
import string
from collections import OrderedDict

from utility import *
from graph import *

a = []
b = []
out = []

width = 20
size = width * width
# Insert random values
for i in range(width):
    rowa = []
    rowb = []
    rowout = []
    for j in range(width):
        rowa.append(rand.randint(0,100))
        rowb.append(rand.randint(0,100))
        rowout.append(0)
    a.append(rowa)
    b.append(rowb)
    out.append(rowout)

dxg = None

graph = Graph()

def matrix(a,b):
    print("Generating graph...")
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

    #print("Resulting matrix:")
    #for e in out:
    #    print(e)
        
    #print_graph(graph)
    G = generate_graph(graph)
    reconstruct(G)

# Conversion from my graph model to networkx
def generate_graph(graph):
    print("Generating networkx...")
    G = OrderedDiGraph()
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
    #pos = nx.graphviz_layout(G, prog = 'dot')
    #nx.draw(G, pos, node_size=1000)
    #nx.draw_networkx_edge_labels(G, pos, edge_labels = edgelabels)
    #nx.draw_networkx_labels(G, pos)
    global dxg
    dxg = G
    return G

# Reconstruct input parameters from networkx graph
def reconstruct(graph):
    print("Reconstructing graph...")
    final_nodes = [] # Nodes with no outedges
    initial_nodes = []
    for node in graph.nodes_iter():
        if (len(graph.out_edges(node)) == 0):
            final_nodes += [node]

    # Trace final nodes back to origin
    initial_nodes = []
    all_paths = []
    #print("Final nodes: " + str(rmap(str, final_nodes)))
    print("Finding paths...")
    for node in final_nodes:
        path = []
        find_paths(graph, [node], initial_nodes, path, all_paths)

    # Sort by output node to try and align input memory
    #all_paths.sort(key=lambda p: len(p[1])) # sort by path length
    #print("Paths found: " + str(len(all_paths)))
    
    # CUDA Code
    #print(cudagen(all_paths, graph))
    cudagen(all_paths, graph)

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

def cudagen(paths, graph):
    # Memory for output
    global out
    outlen = str(rlength(out))

    # Flood fill to detect disconnected graphs
    print("Performing flood fill...")
    disconnected_graphs = []
    for n in flood_fill(graph):
        # Create subgraph from each node group
        dg = OrderedDiGraph()
        dg.add_nodes_from(sorted(n))
        for node in sorted(n):
            for edge in sorted(graph.edges(node)):
                dg.add_edge(edge[0], edge[1])
                dg.edge[edge[0]][edge[1]]["method"] = graph.edge[edge[0]][edge[1]]["method"]
        disconnected_graphs += [dg]
    #print(disconnected_graphs)

    # Group disconected graph into similar kernels
    print("Grouping kernels...")
    kernel_groups = {}
    for g in disconnected_graphs:
        found = False
        for k in kernel_groups.keys():
            if nx.is_isomorphic(k, g):
                kernel_groups[k] += [g]
                found = True
        if not found:
            kernel_groups[g] = [g]
    kernel_groups = kernel_groups.values()
    
    code = """/* CODEGRAPH GENERATED CODE BEGIN */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

"""
    seen_paths = []
    chunkSize = 0
    
    print("Generating CUDA...")
    for i in range(len(kernel_groups)):
        #print(str(rmap(str,seen_paths)))
        # Create initial values
        initmem_array = []
        kernel_graphs = kernel_groups[i]
        initial_dict = get_initial_values(paths, graph)
        
        # Iterate over each disconnected graph for a group
        # of isomorphic kernels
        #print(str(rmap(str, kernel_graphs)))
        for kernel_graph in kernel_graphs:
            
            #plt.clf()
            #pos = nx.graphviz_layout(kernel_graph, prog = 'dot')
            #nx.draw(kernel_graph, pos, node_size=1000)
            #nx.draw_networkx_labels(kernel_graph, pos)
            #plt.show()
            
            # Magic happens here
            # generate initial_node -> array_pos dict
            init_values = []
            init_array_dict = {}
            counter = 0
            for d in initial_dict.keys():
                init_values.append(initial_dict[d])
                init_array_dict[d] = counter
                counter += 1

            for n in range(len(init_values)):
                init_values[n] = "(float) " + str(init_values[n])

            # Find out how to generate final nodes
            finalnodes = []
            for node in kernel_graph.nodes():
                if not kernel_graph.out_edges(node):
                    finalnodes.append(node)
            
            paths_from_final = []
            for node in finalnodes:
                path = rmap_nodes_args(get_path_for_node(node, kernel_graph), get_path_for_node, kernel_graph)
                if path not in seen_paths:
                    seen_paths.append(i)
                    paths_from_final.append(path)
            #paths_from_final.sort()

            final_node_code = []
            chunkSize = 0
            
            if len(paths_from_final) > 0:
                p = rpathsort(paths_from_final[i])
                out = []
                #print(p)
                flattened_path = flatten([p.pop(0)] + sorted(p))
        
                # Generate initial array for this path
                path_init = []
                for e in flattened_path:
                    if e not in ["add", "mul"]:
                        path_init.append(e)
                chunkSize = len(path_init)

                #print("Initial values: " + str(map(str, path_init)))
        
                path_init_nodes = path_init[:]
                for j in range(len(path_init)):
                    initmem_array += ["(float) " + str(path_init[j].value)]
                    path_init[j] = "(float) " + str(path_init[j].value)

                #print("Flattened: " + str(rmap(str,flattened_path)))
                #print("Path init: " + str(path_init_nodes))
        
                flattened_rpn = flatten(rpn_to_path(flattened_path))[:]
        
                # Convert RPN flattened path to initmem indexes
                reconstruction_ids = []
                for node in flattened_rpn:
                    if node not in [" + ", " * "]:
                        reconstruction_ids.append("a[chunkidx + " + str(path_init_nodes.index(node)) + "]")
                    else:
                        reconstruction_ids.append(node)
        
            final_node_code += ["c[threadid]" + " = " + string.join(reconstruction_ids) + ";\n"]

    # Kernel methods
    print("Generating kernels...")
    for kernel in kernel_groups:
        code +="""__global__ void codegraphKernel(float* a, float* c, const int chunkSize, const int limit) {
    int threadid = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    // Don't calculate for elements outside of matrix
    if (threadid >= limit)
    	return;

    int chunkidx = threadid * chunkSize;
    
    // Calculate
    """
        code += string.join(final_node_code)
        code += "}\n"
    
    # Main method
    code += "int main() {\n"
    code += "    const int chunkSize = " + str(chunkSize) + ";\n"
    code += "    const int initSize = " + str(len(initmem_array)) + ";\n"
    code += "    const int limit = (int) initSize/chunkSize;\n" # Or kernel isomorphic group length
    code += "    " + array_to_c(initmem_array, "initmem")

    # TODO specify grid and block size
    code += """
    // Copy to device
	  float* dev_initmem = 0;
	  float* dev_out = 0;
"""
    code += "    float out[" + outlen + "];\n"
    code += "    cudaMalloc(&dev_initmem, initSize * sizeof(float));\n"
    code += "    cudaMalloc(&dev_out, " + outlen + " * sizeof(float));\n"
    code += """
    cudaMemcpy(dev_initmem, initmem, initSize * sizeof(float), cudaMemcpyHostToDevice);

    // Run on device
    codegraphKernel<<<1,initSize>>>(dev_initmem, dev_out, chunkSize, limit);

    // Copy results
"""
    code += "    cudaMemcpy(out, dev_out, " + outlen + " * sizeof(float), cudaMemcpyDeviceToHost);\n"
    code += """
    /*
     *Do something with results here
     */

    // Free
 	  cudaFree(dev_initmem);
 	  cudaFree(dev_out);
"""
    
    code += "}" #end main
    code += "\n/* CODEGRAPH GENERATED CODE END */\n"
    
    # write to file
    f = open("main.cu", 'w')
    f.write(code)
    f.close()
    
    return code

matrix(a,b)
#plt.show() # Plot graph
