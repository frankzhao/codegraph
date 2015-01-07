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