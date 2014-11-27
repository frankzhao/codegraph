import networkx as nx
import matplotlib.pyplot as plt

def draw_graph(graph, labels=None, graph_layout='shell',
               node_size=1600, node_color='blue', node_alpha=0.3,
               node_text_size=12,
               edge_color='blue', edge_alpha=0.3, edge_tickness=1,
               edge_text_pos=0.3,
               text_font='sans-serif'):

    # create networkx graph
    G=nx.Graph()

    # add edges
    for edge in graph:
        G.add_edge(edge[0], edge[1])

    if graph_layout == 'spring':
        graph_pos=nx.spring_layout(G)
    elif graph_layout == 'spectral':
        graph_pos=nx.spectral_layout(G)
    elif graph_layout == 'random':
        graph_pos=nx.random_layout(G)
    else:
        graph_pos=nx.shell_layout(G)

    # draw graph
    nx.draw_networkx_nodes(G,graph_pos,node_size=node_size,
                           alpha=node_alpha, node_color=node_color)
    nx.draw_networkx_edges(G,graph_pos,width=edge_tickness,
                           alpha=edge_alpha,edge_color=edge_color)
    nx.draw_networkx_labels(G, graph_pos,font_size=node_text_size,
                            font_family=text_font)

    if labels is None:
        labels = range(len(graph))

    edge_labels = dict(zip(graph, labels))
    nx.draw_networkx_edge_labels(G, graph_pos, edge_labels=edge_labels,
                                 label_pos=edge_text_pos)

    # show graph
    plt.show()

# Matrix multiplication
a = [[1,2],
     [4,3]]

b = [[2],[3]]

out = [[0],[0]]

labels = []
graph = []

def matrix(a,b):
    for arow in range(len(a)):
        for bcol in range(len(b[0])):
            for brow in range(len(b)):

                # Nodes for initial values
                graph.append( ("a"+str(arow)+str(brow), a[arow][brow] * b[brow][bcol]) )
                graph.append( ("b"+str(brow)+str(bcol), a[arow][brow] * b[brow][bcol]) )

                # Relation for multiplication for the previous 2 nodes
                labels.append("mul")
                labels.append("mul")

                # Nodes for addition values
                graph.append( (a[arow][brow] * b[brow][bcol],
                                "x"+str(arow)+str(bcol)) )
                graph.append( (out[arow][bcol],
                                "x"+str(arow)+str(bcol)) )
                labels.append("add")
                labels.append("add")

                out[arow][bcol] += a[arow][brow] * b[brow][bcol]

    # draw
    print(graph)
    print(labels)
    draw_graph(graph, labels, graph_layout='shell')
