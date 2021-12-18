"""Network restraints for protein graph sampling."""

import os
import itertools
import numpy as np
import networkx as nx
import pandas as pd
from collections import OrderedDict

# default cutoff crosslink (typical value for DSS/BS3 linker)
XL_LINKER_CUTOFF = 30.0

# discrete Kronecker delta
kronecker = lambda x, y: float(x.rb == y.rb)

class Restraint:
    """ Base class for network restraints"""
    
    def __init__(self, graph, weight=1.0, label="", **kwargs):
        """
        Constructor.

        Args:
        graph (networkx Graph): protein_graph.Graph object. 
        
        weight (float, optional): Weighing factor for a restraint
        relative to other restraints. Defaults to 1.0.
        
        label (str, optional): Restraint label. Defaults to "".
        """
        
        self.graph = graph
        self.weight = weight
        self.label = self.__class__.__name__+ "_" + label
        self.scores = {}
    
    def update(self, scores=None):
        """
        Updates the internal scores dict.
        
        Args:
        scores: Dict of score fields. Defaults to None.
        """
        
        if scores is None: scores = self.evaluate()
        self.scores.update(scores)
    
    def get_total_score(self):
        """
        Add all the scores.
        
        Returns:
        (float): sum of all the scores.
        """
        
        return sum(self.scores.values())
    
    def set_weight(self, w):
        """
        Set the weight factor for this restraint.
        
        Args:
        w (float): Restraint weight.
        """
        
        self.weight = w
    
    def evaluate(**kwargs):
        """
        Evaluate scores for this restraint. Defined in base classes.
        """
        
        pass
    
    def set_label(self, label=""):
        """
        Set label for this restraint.
        Args:
        label (str, optional): New label for this restraint. Defaults to "".
        """
        
        self.label = self.__class__.__name__ + "_" + label


class Compactness(Restraint):
    """
    Restraint on the structural compactness of rigid bodies. This also
    prevents the formation of too many rigid bodies.
    """
    
    def __init__(self, graph, weight=1.0, label=""):
        print("\nCreating Compactness restraint...")
        super().__init__(graph, weight=weight, label=label)
        self.edges = self.graph.edges
        #self.norm = sum([self.graph[u][v]["weight"] for (u, v) in self.edges])
        self.update()
        
    def evaluate(self, nodes=[]):
        """
        Evaluate scores for this restraint.
        
        Args:
        (list): List of nodes over which the scores are calculated.
        
        Returns:
        (dict): Dict of scores, where they keys are edges covered by the 
        list of nodes given.
        """
        
        if not nodes: nodes = list(self.graph.nodes)
        this_edges = [(u, v) for (u, v) in self.edges
                 if (u in nodes or v in nodes)]
        scores = {}
        for (u, v) in this_edges:
            score = (1-kronecker(u,v)) * self.graph[u][v]["weight"]
            scores[(u, v)] = self.weight * score
        return scores


class SubgraphConnectivity(Restraint):
    """
    Restraint on the structural contiguity of rigid bodies. This prevents
    splintered or fragmented rigid bodies.
    """
    
    def __init__(self, graph, weight=1.0, label=""):
        print("\nCreating SubgraphConnectivity restraint...")
        super().__init__(graph, weight=weight, label=label)
        self.non_edges = []
        for (u, v) in itertools.combinations(self.graph.nodes, 2):
            if (u, v) not in self.graph.edges: self.non_edges.append((u, v))
        #self.norm = len(self.non_edges)
        self.update()
    
    def evaluate(self, nodes=[]):
        """
        Evaluate scores for this restraint.
        
        Args:
        (list): List of nodes over which the scores are calculated.
        
        Returns:
        (dict): Dict of scores, where they keys are node pairs (from the list
        of given nodes) that don't have edges.
        """

        if not nodes: nodes = list(self.graph.nodes)
        this_non_edges = [(u, v) for (u, v) in self.non_edges
                 if (u in nodes or v in nodes)]
        scores = {}
        for (u, v) in this_non_edges:
            score = kronecker(u, v)
            scores[(u, v)] = self.weight * score
        return scores
    
    
class Crosslink(Restraint):
    """
    Crosslink satisfaction restraint. Satisfaction of structurally mappable
    crosslinks is scored favorably when within the same rigid body and 
    unfavorably when between rigid bodies.
    """
    
    def __init__(self, graph, crosslink_data_file, linker_cutoffs={},
                weight=1.0, label="", ):
        """
        Constructor.

        Args:
        graph (networkx Graph): protein_graph.Graph object. 
        
        crosslink_data_file (str): CSV file containing all crosslink data
        for this restraint. The CSV file should have at least five columns:
        'protein1', 'residue1', 'protein2', 'residue2', 'linker'.
        Different crosslinks can have different cutoffs.
        
        linker_cutoffs (dict): Cutoff distances (max. length of stretched
        spacer arms) of linkers referenced by linker names same as those in
        the crosslink data file.
        
        weight (float, optional): Weighing factor for a restraint
        relative to other restraints. Defaults to 1.0.
        
        label (str, optional): Restraint label. Defaults to "".
        """
        
        print("\nCreating Crosslink restraint...")
        super().__init__(graph, weight=weight, label=label)
        self.xl_fn = crosslink_data_file
        self.linker_cutoffs = linker_cutoffs
        self._mappable_xl_fn = "mappable_xl.csv"
        self._add_xl_data()
        self.update()
        
    def _add_xl_data(self):
        """Parse crosslink data from given crosslink data file."""
        
        print("Adding XL data from %s" % self.xl_fn)
        df = pd.read_csv(self.xl_fn)
        xls = []
        for i in range(len(df)):
            this_df = df.iloc[i]
            a1 = (this_df["protein1"], this_df["residue1"]) 
            a2 = (this_df["protein2"], this_df["residue2"])
            linker = this_df["linker"]
            if not ((a1 in self.graph.residue_node_map) and \
                    (a2 in self.graph.residue_node_map)):
                continue
            xl = (*a1, *a2, linker)
            xls.append(xl)
            
            if linker in self.linker_cutoffs:
                cutoff = self.linker_cutoffs[linker]
            else:
                cutoff = XL_LINKER_CUTOFF
            
            u, u_coord = self.graph.residue_node_map[a1]
            v, v_coord = self.graph.residue_node_map[a2]
            sat = float(np.linalg.norm(u_coord - v_coord) <= cutoff)
            key = (*a1, *a2, linker)
            val = (u, v, sat)
            self.graph.xl_data[key] = val
            
        columns = ["protein1", "residue1", "protein2", "residue2", "linker"]
        mappable_df = pd.DataFrame(xls, columns=columns)
        mappable_df.to_csv(self._mappable_xl_fn, index=False)
        frac_mappable = float(len(mappable_df)) / len(df)
        print("Mappable crosslinks: %d out of %d [%2.2f%%]" % \
             (len(mappable_df), len(df), 100*frac_mappable))
    
    def _get_xl_weight(self, u, v):
        """
        Get the weight for a crosslink. This is useful for modulating the
        score depending on whether one or more crosslinked residues have
        a lower weight (e.g. when they are loops with low structural quality.)
        
        Args:
        u, v: (protein_graph.Node): Protein graph nodes.
        
        Returns:
        (float): Arithmetic mean of weights of crosslinked weights.
        """
        
        return 0.5*(u.weight + v.weight) # arithmetic mean
        #return np.sqrt(u.weight * v.weight)  # geometric mean
    
    def evaluate(self, nodes=[]):
        """
        Evaluate scores for this restraint.
        
        Args:
        (list): List of crosslinked nodes over which the scores are calculated.
        
        Returns:
        (dict): Dict of scores, where they keys are node pairs corresponding
        to pairs or crosslinked residues, taken from the given node list.
        """
        
        if not nodes: nodes = list(self.graph.nodes)
        
        this_data = [(u,v,sat) for (u,v,sat) in self.graph.xl_data.values()
                     if (u in nodes or v in nodes)]
        
        scores = {(u,v): 0.0 for (u,v,sat) in this_data}
        for (u, v, sat) in this_data:
            score = sat*(1-kronecker(u,v)) + (1-sat)*kronecker(u,v)
            score *= self._get_xl_weight(u, v)
            scores[(u,v)] += self.weight * score
        
        # scores = {p: 0.0 for p in nodepairs}
        # for xl in xls:
        #     u, v, sat = self.xl_data[xl]
        #     score = sat*(1-kronecker(u,v)) + (1-sat)*kronecker(u,v)
        #     score*= self._get_xl_weight(u, v)
        #     scores[(u, v)] += score
                    
        return scores


class StructuralCoverage(Restraint):
    """ 
    Restraint on the amount of structure covered by the rigid bodies. This
    keeps a check on monte carlo samplers trying to reduce node weights
    (of loop nodes for the time being) arbitrarily.
    """
    
    def __init__(self, graph, weight=1.0, label=""):
        print("\nCreating StructuralCoverage restraint...")
        super().__init__(graph, weight=weight, label=label)
    
    def evaluate(self, nodes=[]):
        """
        Evaluate scores for this restraint.
        
        Args:
        (list): List of nodes over which the scores are calculated.
        
        Returns:
        (dict): Dict of scores, where they keys are given nodes.
        """
        
        if not nodes: nodes = list(self.graph.nodes)
        # use only loop nodes for now since other nodes don't
        # have weight movers
        loop_nodes = [u for u in nodes if u.struct == "loop"]
        scores = {}
        for u in loop_nodes:
            scores[u] = self.weight / (1 + u.weight)
        return scores


# ----------------------------------------
# DEPRECATED OR DON'T WORK TOO WELL (YET!)
# ----------------------------------------
# class RigidbodySizeLowerBound(Restraint):
#     def __init__(self, graph, weight=1.0, min_size=2, gamma=1.0, label=""):
#         print("\nCreating RigidbodySizeLowerBound restraint...")
#         super().__init__(graph, weight=weight, label=label)
#         self.min_size = min_size
#         self.gamma = gamma
#         self.update()
    
#     def evaluate(self, rbs=[]):
#         rbdict = self.graph.get_rb_dict()
#         if not rbs: rbs = list(rbdict.keys())
#         scores = {i: 0 for i in rbs}
#         for i in rbs:
#             rb_size = len(rbdict[i])
#             if rb_size <= self.min_size:
#                 scores[i] = self.gamma * (rb_size - self.min_size)**2
#         return scores
        

# class Modularity(Restraint):
#     def __init__(self, graph, weight=1.0, label="", gamma=1.0):
#         print("\nCreating Modularity restraint...")
#         super().__init__(graph, weight=weight, label=label)
#         self.gamma = gamma
#         self._coeffs = OrderedDict()
#         self._precompute_coeffs()
#         self.update()
        
#     def _precompute_coeffs(self):
#         m = len(self.graph.edges)
#         for (u, v) in itertools.combinations(self.graph.nodes, 2):
#             if self.graph.has_edge(u, v): a_uv = self.graph[u][v]["weight"]
#             else: a_uv = 0
#             k_u, k_v = self.graph.degree(u), self.graph.degree(v)
#             p_uv = k_u*k_v / (2*m)
#             self._coeffs[(u, v)] = (a_uv - self.gamma*p_uv) / (2*m)
    
#     def evaluate(self, nodes=[]):
#         if not nodes: nodes = list(self.graph.nodes)
#         scores = {}
#         for (u, v) in itertools.combinations(nodes, 2):
#             if (u,v) in self._coeffs: key = (u, v)
#             elif (v,u) in self._coeffs: key = (v, u)
#             else: continue
#             scores[key] = -self.weight * kronecker(u,v) * self._coeffs[key]
#         return scores
