Codegraph
===

Codegraph is an experiment in generating useable CUDA code from generic algorithms by preprocessing. Read the paper here. An example of the generated code can be found in `main.cu`.

How does it work?
===

The key idea behind Codegraph is that a computational algorithm can be represented as a graph where the nodes represent calculation values (or synonymously memory pointers) and the edges represent operations between these values. Translating an algorithm into a graphical representation allows for the use of graph analysis techniques to be used. Codegraph uses these techniques to identify possible kernels and generate parallelised CUDA code from an original Python implementation.

Hooks can be inserted into a simplified Python algorithm to generate the graph. The graph is then analysed for disconnected graphs (representing separate threads) and isomorphisms (representing similar kernels).

Dependencies
===

- graphviz
- pygraphviz
- networkx
- matplotlib
- numpy

Running
===

Currently the file `matrix.py` implements code generation for matrix multiplication. This can be run with

```
python matrix.py
```

The generated CUDA code will be created in `main.cu`.

The graph can be shown by uncommenting the last line `plt.show()`.

Algorithm Assumptions
===

To reduce the complexity of the program in its early stages, the following assumptions are made about the algorithm and thereby the generated graph.

- Graph has no loops or backedges
- Algorithm is ultimately a reduction
- The program always achieves termination
- Operations in the algorithm can be represented using only addition and multiplication. This means subtraction needs to be interpreted as the addition of a negative number, and division is interpreted as multiplication of the inverse.

Notes
===

- Names assigned to nodes during creation will become variable names in the generated code. It is important to ensure node names are C compliant and unique.
- Codegraph runs on the CPU. Do not include your entire dataset in the graph generation.
- Code for processing output values can be inserted in the comment block that looks like

```
/*
 *Do something with results here
 */
```