import os
import random
import itertools
import unittest
import pandas as pd
import numpy as np

import IMP.test
import IMP.raindrops


def _get_graph(pdb_dir, topology_file):
    tp = IMP.raindrops.io_tools.TopologyParser(pdb_dir)
    tp.parse(topology_file)
    g = IMP.raindrops.protein_graph.Graph(tp, k=1, maxrb=10,
                                      symmetrize_edges=True)
    g.build()
    return g


class TestRaindrops(IMP.test.TestCase):
    def test_partition(self):
        """
        Test the partitioning of a single chain 
        into structured and loop segments.
        """
        regions = [IMP.raindrops.protein_graph.Region("5u8s", "2", 100, 500),
                   IMP.raindrops.protein_graph.Region("5u8s", "A", 1, 200)]
        
        min_ss = 4
        min_loop, max_loop = 5, 8
        p = IMP.raindrops.protein_graph.Partition("data/5u8s.pdb",
                                              regions=regions, 
                                              min_ss=min_ss,
                                              min_loop=min_loop, max_loop=max_loop)
        p.run()
        for k, v in p.segments.items():
            ss, loop = v
            for (begin, end) in ss:
                assert end-begin + 1 >= min_ss
            for (begin, end) in loop:
                looplen = end-begin+1
                assert looplen >= min_loop
                assert (looplen <= max_loop) or \
                       (looplen > max_loop and looplen - max_loop <= min_loop)
        
    
    def test_protein_graph(self):
        """
        Test protein graph construction for an input protein complex.
        """
        allowed_chains = {1: ["2", "3", "4", "5", "6", "7"],
                          2: ["A", "B", "C", "D"],
                          3: ["E"]}
        
        g = _get_graph("data", "data/topology_3.txt")    
        rbdict = g.get_rb_dict()
        
        assert sorted(rbdict) == [1,2,3]
        
        for rb, nodes in rbdict.items():
            chains = sorted(set([r.chain for u in nodes for r in u.regions]))
            assert chains == allowed_chains[rb]
            
    
    def test_compactness_restraint(self):
        """
        Test that the compactness score works correctly
        on a protein graph with unit weights for all edges.
        """
        g = _get_graph("data", "data/topology_3.txt")

        # set weights of 1 for all edges
        for (u,v) in g.edges:
            g[u][v]["weight"] = 1.0
        rbdict = g.get_rb_dict()
        
        # pick a node at random
        deg = 0
        nodes = list(g.nodes)
        while deg == 0:
            u = random.choice(nodes)
            deg = g.degree(u)
            
        cr = IMP.raindrops.restraints.Compactness(g, label="test_compactness",
                                              weight=1.0)
        scores = cr.scores
        assert 0 <= sum(scores.values()) <= deg
        for (u, v) in scores:
            assert g.has_edge(u, v)
            assert scores[(u,v)] == 1.0 - float(u.rb == v.rb)
        
    def _test_subgraph_connectivity_restraint(self):
        g = _get_graph("data", "data/topology_3.txt")

        # set weights of 1 for all nodes
        for (u,v) in g.edges:
            g[u][v]["weight"] = 1.0
        rbdict = g.get_rb_dict()
        
        # pick a node at random
        u = random.choice(list(g.nodes))
        
        # subgraph connectivity restraint
        scr = IMP.raindrops.restraints.SubgraphConnectivity(g,
                                    label="test_subgraph_connectivity",
                                    weight=1.0)
        scores = scr.scores
        assert 0 <= sum(scores.values()) <= len(g) - 1 - g.degree(u)
        for (u, v) in scores:
            assert not g.has_edge(u, v)
            assert scores[(u,v)] == float(u.rb == v.rb)
    
    def test_crosslink_restraint_mapping(self):
        """
        Test that the crosslink restraints correctly
        maps crosslinks to available structural components.
        """
        g = _get_graph("data", "data/topology_3.txt")
        
        # crosslink restraint
        xlr = IMP.raindrops.restraints.Crosslink(g, "data/xl.csv", weight=1.0,
                                             label="test_crosslink",
                                             linker_cutoffs={"DSS": 28.0})
        
        # the mappable crosslinks were saved from a previous run and 
        # stored back as xl.csv in the data folder
        tmp_fn = "mappable_xl.csv"
        df_tmp = pd.read_csv(tmp_fn)
        xl_tuples_tmp = [tuple(df_tmp.loc[i]) for i in range(len(df_tmp))]
        
        df = pd.read_csv("data/xl.csv")
        xl_tuples = [tuple(df.loc[i]) for i in range(len(df))]
        
        assert all([xl_tuples[i] == xl_tuples_tmp[i]
                    for i in range(len(df))])
        
        os.remove(tmp_fn)


    def test_crosslink_restraint_score(self):
        """
        Test the score calculation for the crosslink restraint
        for a graph with unit weights for all nodes.
        """
        g = _get_graph("data", "data/topology_3.txt")
        
        # set weights of 1 for all nodes
        g.set_node_weights([1.0] * len(g))
        
        # set rigid bodies of all nodes to the same
        g.set_rbs([1] * len(g))
        
        # crosslinks
        df = pd.read_csv("data/xl.csv")
        xls = [tuple(df.loc[i]) for i in range(len(df))]
        
        xlr = IMP.raindrops.restraints.Crosslink(g, "data/xl.csv", weight=1.0,
                                             label="test_crosslink",
                                             linker_cutoffs={"DSS": 28.0})
        os.remove("mappable_xl.csv")
        
        # calculate score manually
        m = g.models["5u8s"]
        mol2chain_map = {"mcm2": "2", "mcm3": "3", "mcm4": "4", "mcm5": "5",
                         "mcm6": "6", "mcm7": "7", "psf1": "A", "psf2": "B",
                         "psf3": "C", "sld5": "D", "cdc45": "E"}
        
        resid2atom_map = {}
        for c in ["2", "3", "4", "5", "6", "7", "A", "B", "C", "D", "E"]:
            resid2atom_map[c] = {r.id[1]: r["CA"] for r in m[c].get_residues()}
     
        my_score = 0.0
        for xl in xls:
            p1, r1, p2, r2, _ = xl
            u, _ = g.residue_node_map[(p1, r1)]
            v, _ = g.residue_node_map[(p2, r2)]
            c1, c2 = mol2chain_map[p1], mol2chain_map[p2]
            a1, a2 = resid2atom_map[c1][r1], resid2atom_map[c2][r2]
            my_score += float(a1-a2 <= 28.0)
        
        IMP.raindrops_score = sum(xlr.scores.values())
        assert IMP.raindrops_score + my_score == len(xls)
        
        
       
if __name__ == "__main__":
    IMP.test.main()
    
