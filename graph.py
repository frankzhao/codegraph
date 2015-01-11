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
        
    def __str__(self):
        return str(self.method)
        
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