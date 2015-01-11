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
        
    #print_graph(graph)
    G = generate_graph(graph)
    reconstruct(G)

# Conversion from my graph model to networkx
def generate_graph(graph):
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
    all_paths.sort(key=lambda p: len(p[1])) # sort by path length
    print("Paths found: " + str(len(all_paths)))
    
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
test = None
def cudagen(paths, graph):
    # Memory for output
    global out
    outlen = str(len(out))

    # Flood fill to detect disconnected graphs
    disconnected_graphs = []
    for n in flood_fill(graph):
        # Create subgraph from each node group
        dg = graph.subgraph(n)
        disconnected_graphs += [dg]

    # Group disconected graph into similar kernels
    kernel_groups = []
    seen = []
    for i in range(len(disconnected_graphs)):
        current_kernel_group = []
        for j in range(len(disconnected_graphs)):
            if (disconnected_graphs[i] != disconnected_graphs[j]) and (disconnected_graphs[j] not in seen):
                if nx.is_isomorphic(disconnected_graphs[i], disconnected_graphs[j]):
                    current_kernel_group.append(disconnected_graphs[i])
                    current_kernel_group.append(disconnected_graphs[j])
                    seen.append(disconnected_graphs[i])
                    seen.append(disconnected_graphs[j])
            print(current_kernel_group)
        if len(current_kernel_group) > 0:
            kernel_groups += [current_kernel_group]
    print("===========")
    print(kernel_groups)
    global test
    test = kernel_groups
    
    code = """/* CODEGRAPH GENERATED CODE BEGIN */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

"""
    for i in range(len(kernel_groups)):
        # Create initial values
        initmem_array = []
        initial_dict = get_initial_values(paths, graph)
        kernel_graphs = kernel_groups[i]
        
        # Iterate over each disconnected graph for a group
        # of isomorphic kernels
        for kernel_graph in kernel_graphs:
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
                paths_from_final.append(path)
                print(str(rmap(str, path))) # This generates prefix notation
            paths_from_final.sort()

            final_node_code = []
            chunkSize = 0
        
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
        
            final_node_code += ["c[chunkidx]" + " = " + string.join(reconstruction_ids) + ";\n"]

    # Kernel methods
    for kernel in kernel_groups:
        code +="""__global__ void codegraphKernel(float* a, float* c, const int chunkSize) {
    int threadid = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    // Don't calculate for elements outside of matrix
    if (threadid >= chunkSize)
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
    codegraphKernel<<<1,initSize>>>(dev_initmem, dev_out, chunkSize);

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
    
# Create an OrderedDiGraph to preserve operation order
# Inherits from nx.Digraph
class OrderedDiGraph(nx.DiGraph):
    def __init__(self, data=None, **attr):
        self.node_dict_factory = OrderedDict
        self.adjlist_dict_factory = OrderedDict
        self.edge_attr_dict_factory = OrderedDict

        self.graph = {} # dictionary for graph attributes
        self.node = OrderedDict() # dictionary for node attributes
        # We store two adjacency lists:
        # the  predecessors of node n are stored in the dict self.pred
        # the successors of node n are stored in the dict self.succ=self.adj
        self.adj = OrderedDict()  # empty adjacency dictionary
        self.pred = OrderedDict()  # predecessor
        self.succ = self.adj  # successor

        # attempt to load graph with data
        if data is not None:
            convert.to_networkx_graph(data,create_using=self)
        # load graph attributes (must be after convert)
        self.graph.update(attr)
        self.edge=self.adj

matrix(a,b)
plt.show() # Plot graph
