"""Graph representation of macromolecular protein complexes."""

import os
import subprocess
import shutil
import random
import dill
import numpy as np
import networkx as nx
from collections import namedtuple, OrderedDict
from Bio.PDB import PDBParser, Select, PDBIO
from .io_tools import make_struct2seq_map

# secondary structure
STRIDE_EXEC = "stride"
SS = ["H", "G", "I", "E", "B", "b"]
LOOP = ["s", "T", "C"]
SS_Assignment = namedtuple("SSA", ["resid", "structure"])

# protein region definition
Region = namedtuple("Region", ["src", "chain", "begin", "end"])

# node structure types
NODE_STRUCT_TYPES = ["ss", "loop", "seed"]


class Trimmer(Select):
    """
    Overloaded Biopython Selector object, to trim a structure model
    by extracting given regions.
    """
    def __init__(self, regions, *args, **kwargs):
        """
        Constructor

        Args:
        regions (Region namedtuple): tuple containing (soure name, chain id,
        start residue, end residue) for a region.
        """

        super().__init__(*args, **kwargs)
        self.keep_residues = []
        for reg in regions:
            c, r0, r1 = reg.chain, reg.begin, reg.end
            this_region_keep_residues = [(c, r) for r in range(r0, r1+1)]
            self.keep_residues.extend(this_region_keep_residues)

    def accept_chain(self, chain):
        return chain.id in {x[0] for x in self.keep_residues}

    def accept_residue(self, residue):
        if residue.id[0] != " ":
            return False  # remove HETATMs
        return (residue.get_parent().id, residue.id[1]) in self.keep_residues


class Partition:
    """
    Secondary structure based contiguous segment partitioner. Structured
    segments (i.e. helices, sheets or combinations) can have small (within
    an allowed length) structurally missing regions within them.
    """

    def __init__(self, pdb_fn, regions=[], min_ss=2, min_loop=4, max_loop=10):
        """
        Constructor.

        Args:
        pdb_fn (str): PDB file name.

        regions (list, optional): List of protein regions to segment.
        Defaults to empty list, in which case, the entire PDB structure
        is selected.

        min_ss (int, optional): Min. length of a secondary structure segment.
        Defaults to 2.

        min_loop (int, optional): Min. length of a loop segment. Defaults to 4.

        max_loop (int, optional): [description]. Max. length of a loop
        segment. Defaults to 10.
        """

        self.pdb_fn = pdb_fn
        self.min_ss = min_ss
        self.min_loop = min_loop
        self.max_loop = max_loop
        self.regions = regions
        self.segments = OrderedDict()

        # parse pdb file into Biopython model
        self.model = PDBParser(QUIET=True).get_structure("x", self.pdb_fn)[0]
        self._trim()

    def run(self):
        """
        Partition the requested regions of the supplied protein complex.
        """
        ssa = self._get_ssa()
        for c, this_ssa in ssa.items():
            p_ss, p_loop = self._get_chain_partition(this_ssa)
            self.segments[c] = (p_ss, p_loop)

    def _trim(self):
        """
        Trim the supplied PDB file, by selecting only requested regions.
        """

        if not self.regions:
            return
        tmp_pdb_fn = self.pdb_fn.split("/")[-1] + "_tmp.pdb"
        io = PDBIO()
        io.set_structure(self.model)
        io.save(tmp_pdb_fn, select=Trimmer(self.regions), write_end=False,
                preserve_atom_numbering=True)
        self.model = PDBParser(QUIET=True).get_structure("x", tmp_pdb_fn)[0]
        os.remove(tmp_pdb_fn)

    def _get_ssa(self):
        """
        Use STRIDE to get secondary structure assignment of all residues.

        Raises:
        IOError: If STRIDE binary is not found on the system path and
        named 'stride'.

        Returns:
        (dict): Dictionary of secondary structure assignments
        with chains as keys. For each chain, the assignments are stored
        as a list of (residue id, Q3 secondary struct alphabet) tuples.
        """

        if shutil.which(STRIDE_EXEC) is None:
            raise IOError("stride executable not found in path. You can \
            download STRIDE from http://webclu.bio.wzw.tum.de/stride")

        p = subprocess.Popen([STRIDE_EXEC, self.pdb_fn],
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        result, err = p.communicate()
        result = result.decode("utf-8")
        ssa_lines = [i for i in result.split("\n") if i.startswith("ASG")]

        ssa = OrderedDict()
        resdict = {}
        for c in self.model.get_chains():
            resdict[c.id] = [r.id[1] for r in self.model[c.id].get_residues()]

        for line in ssa_lines:
            spl = line.split()
            chain, resid, struct = spl[2], int(spl[3]), spl[5]
            if chain not in resdict:
                continue
            if chain not in ssa:
                ssa[chain] = []
            if resid not in resdict[chain]:
                continue
            ssa[chain].append(SS_Assignment(resid, struct))

        return ssa

    def _get_chain_partition(self, ssa):
        """
        Main workhorse method that partitions a given secondary structure
        assignment into structured (helix, sheet) and loop segments.

        Args:
        ssa (str): Secondary structure assignment for a chain, reported
        as a list of (residue id, Q3 alphabet) tuples.

        Returns:
        (tuple): Two list of segments, one for all structured regions (helices
        or sheets) and the other for all loops. Each list consists of range
        tuples [(b1, e1), (b2, e2), ...], where (b_i, e_i) marks the beginning
        and end residue of the i^th segment.
        """

        # extract all contiguous SS residues greater than MINSS
        p_ss_ = []
        begin = None
        end = None
        for i in range(len(ssa)):
            if ssa[i].structure in SS:
                if begin is None:
                    begin, end = i, i
                else:
                    end = i
            else:
                if begin is not None:
                    if ssa[end].resid - ssa[begin].resid + 1 >= self.min_ss:
                        p_ss_.append((begin, end+1))
                    else:
                        # label critically short segments as random coil
                        for idx in range(begin, end+1):
                            ssa[idx] = ssa[idx]._replace(structure="C")
                    begin = None
                    end = None

        # merge short flexible regions with *both* sides flanked by ss regions
        # into these flanking ss regions
        p_ss = []
        bracket = p_ss_[0]
        for i in range(1, len(p_ss_)):
            if (ssa[p_ss_[i][0]].resid - ssa[bracket[1]-1].resid - 1
                    <= self.min_loop):
                bracket = bracket[0], p_ss_[i][1]
            else:
                p_ss.append(bracket)
                bracket = p_ss_[i]
        p_ss.append(bracket)

        # get a list of all loops (ssa indices) after critically short flexible
        # (i.e. both loops and missing regions) regions have been merged with
        # flanking secondary structured regions
        structured = []
        for begin, end in p_ss:
            structured.extend(range(begin, end))
        loops = [i for i in range(len(ssa)) if i not in structured
                 and ssa[i].structure in LOOP]

        # extract loops
        p_loop_ = []
        bracket = []
        if loops:
            bracket = [loops[0]]
        for i in range(1, len(loops)):
            if ssa[loops[i]].resid - ssa[bracket[-1]].resid == 1:
                bracket.append(loops[i])
            else:
                p_loop_.append((bracket[0], 1+bracket[-1]))
                bracket = [loops[i]]
        if bracket:
            p_loop_.append((bracket[0], 1+bracket[-1]))

        # merge short loops that still remain into adjacent ss regions.
        # these are short loops that separate ss and missing regions
        ss_beginnings = {p_ss[i][0]: i for i in range(len(p_ss))}
        ss_ends = {p_ss[i][1]: i for i in range(len(p_ss))}
        p_loop = []
        for p in p_loop_:
            if ssa[p[1]-1].resid - ssa[p[0]].resid + 1 >= self.min_loop:
                p_loop.append(p)
            # check if this loop is flanking N-end of some ss
            elif p[1] in ss_beginnings:
                idx = ss_beginnings[p[1]]
                p_ss[idx] = p[0], p_ss[idx][1]
            # check if this loop is trailing C-end of some ss
            elif p[0] in ss_ends:
                idx = ss_ends[p[0]]
                p_ss[idx] = p_ss[idx][0], p[1]

        # clip all loops to MAXLOOP.
        # Start clipping from the N-terminus of the loop.
        # The last C-terminal sub-loop might be longer than MAXLOOP, so as
        # not to produce a dangling loop less than MINLOOP length
        p_loop_clipped = []
        for p in p_loop:
            looplen = ssa[p[1]-1].resid - ssa[p[0]].resid + 1
            if looplen <= self.max_loop:
                p_loop_clipped.append(p)
            else:
                n = int(looplen / self.max_loop)
                for i in range(n):
                    begin = p[0] + i*self.max_loop
                    end = p[0] + (i+1)*self.max_loop
                    p_loop_clipped.append((begin, end))
                # check if the remaining loop segment is less than MINLOOP
                # If so merge with the last created segment.
                if (ssa[p[1]-1].resid - ssa[p[0]+n*self.max_loop-1].resid
                        < self.min_loop):
                    p_loop_clipped[-1] = (p_loop_clipped[-1][0], p[1])
                else:
                    p_loop_clipped.append((p[0] + n*self.max_loop, p[1]))

        # convert from ssa indices to actual residue ids
        p_ss_out = [(ssa[begin].resid, ssa[end-1].resid)
                    for (begin, end) in p_ss]

        p_loop_out = [(ssa[begin].resid, ssa[end-1].resid)
                      for (begin, end) in p_loop_clipped]

        return p_ss_out, p_loop_out


class Node:
    """
    Class definition for a node of a protein graph. A node represents
    one or more protein segments obtained by parttioning on the basis
    of secondary structure.
    """

    def __init__(self, regions, weight=1.0, rb=1, struct=None):
        """
        Constructor.

        Args:
        regions (list): List of Region namedtuples.

        weight (float, optional): Structural weight of a node. Defaults to 1.0.

        rb (int, optional): Rgid body assignment of this node. Defaults to 1.

        struct (str, optional): Structure type of the node.
        Can only be one of 'ss', 'loop' or 'seed'. Defaults to None.
        """

        if not isinstance(regions, list):
            regions = [regions]
        self.regions = regions
        self.weight = weight
        self.rb = rb

        self.struct = struct
        if self.struct not in NODE_STRUCT_TYPES:
            raise TypeError(
                "Node structure type can only be one of ss, loop or seed.")

        self.residues = {}
        self.coords = []
        self.com = None

    def __repr__(self):
        s = ["%s.%s:%d-%d" % reg for reg in self.regions]
        s = " , ".join(s)
        return s

    def __len__(self):
        return len(self.coords)

    def set_rb(self, rb):
        self.rb = rb

    def set_weight(self, w):
        self.weight = w

    def add_structure(self, models):
        """
        Add structure (3D coords) to a node from a PDB file.

        Args:
        models (dict): Dictionary of Biopython Model objects with pdb file
        prefixes as keys. This is usually called by node constructor methods
        of the protein graph object.
        """

        for reg in self.regions:
            for r in range(reg.begin, reg.end+1):
                key = (reg.src, reg.chain, r)
                try:
                    val = models[reg.src][reg.chain][r]["CA"].coord
                except KeyError:
                    print("Missing residue (%s, %s, %d)" % key)
                    continue
                self.residues[key] = val
                self.coords.append(val)

        self.coords = np.array(self.coords)
        self.com = np.mean(self.coords, axis=0)


class Graph(nx.Graph):
    """
    Graph representation of a protein complex built from nodes that
    represent contiguous structured regions, classified by a Q3 secondary
    structure assignment.
    """

    def __init__(self, topology_parser,
                 k=4, symmetrize_edges=True,
                 maxrb=500, **kwargs):
        """
        Constructor.

        Args:
        topology_parser (io_tools.TopologyParser): Protein complex topology
        parser object.

        k (int, optional): Min. k for constructing a k-NN edgelist, that
        just connects the graph. Defaults to 4. But in general, is system
        dependent.

        symmetrize_edges (bool, optional): True to consider only
        symmetric nearest neighbors for edge construction. Defaults to True.

        maxrb (int, optional): Maximum number of rigid bodies allowed.
        Defaults to 500.

        kwargs: (min_ss, max_loop, min_loop) parameters that control
        the partitioning of a chain into structured and loop-like segments
        and are passed to protein_graph.Partition
        """

        super().__init__()
        self.topology = topology_parser.topology
        self.k = k
        self.symmetrize_edges = symmetrize_edges
        self.maxrb = maxrb
        self._partition_builder_kwargs = kwargs

        self.pdb_dir = topology_parser.pdb_dir
        self.models = topology_parser.models
        self._partitions = OrderedDict()
        self._init_seed_rbs = []

        self.residue_node_map = {}
        self.xl_data = {}

    def _build_partition_input(self):
        """
        Extract regions that should be partitioned into secondary structure
        segments, and regions that should be maintained as intact rigid bodies.

        Raises:
        ValueError: If intact rigid bodies have been given rigid body id
        greater than the max. allowed # of rigid bodies (500 by default)

        Returns:
        (tuple): Dictionary containing all relevant inputs for the Partition
        object that segments protein complexes by secondary structure, and
        a dictionary of all seed regions that make up intact rigid body/(ies).
        """

        partition_input_dict = OrderedDict()
        seed_region_dict = OrderedDict()
        for t in self.topology:
            molname, src, chain, resrange, pdb_offset, rb = t
            reg = Region(src, chain, *resrange)
            if rb is None:
                # regions to be partitioned
                pdb_fn = os.path.join(self.pdb_dir, src + ".pdb")
                if src not in partition_input_dict:
                    partition_input_dict[src] = (pdb_fn, [])
                partition_input_dict[src][1].append(reg)
            else:
                # regions to be retained as intact rigid bodies
                if rb not in range(1, self.maxrb+1):
                    raise ValueError(
                        "Rigid body id out of bounds. Should be limited "
                        "to [1, %d]" % self.maxrb)
                if rb not in seed_region_dict:
                    seed_region_dict[rb] = []
                    self._init_seed_rbs.append(rb)
                seed_region_dict[rb].append(reg)
        return partition_input_dict, seed_region_dict

    def _build_residue_node_map(self):
        """
        Map a (chain, residue) tuple to a (graph node, residue coords) tuple.
        """

        struct2seq_map = make_struct2seq_map(self.topology)
        for u in self.nodes:
            for reskey, rescoords in u.residues.items():
                src, chain, r_struct = reskey
                molname, r_seq = struct2seq_map[(src, chain, r_struct)]
                self.residue_node_map[(molname, r_seq)] = (u, rescoords)

    def _build_kNN_edges(self):
        """
        Create edges between k-nearest neighbors (k-NN) of all nodes.
        Neighborhood may be symmetric, i.e. u and v both need to be within
        k-NN lists of each other.

        Normalized edge weights are calculated based on:
        Data clustering using a model granular magnet,
        Blatt, Wiseman and Domany, Neural Computation, 1997
        https://doi.org/10.1162/neco.1997.9.8.1805

        Raises:
        ValueError: If the neighborhood list size (k) is greater than the
        number of nodes.

        Returns:
        (list): List of weighted edges.
        """

        # compute distance matrix
        nodes = list(self.nodes)
        nnodes = len(nodes)
        distmat = np.zeros([nnodes, nnodes], np.float32)
        for i in range(nnodes-1):
            for j in range(i+1, nnodes):
                u, v = nodes[i], nodes[j]
                d = np.linalg.norm(u.com - v.com)
                distmat[i, j], distmat[j, i] = d, d

        # compute k-nearest-neighbors for each node
        if self.k >= nnodes:
            raise ValueError("Cannot construct %d-NN graph for %d nodes"
                             % (self.k, nnodes))
        knn = {}
        neigh_dist = []
        for i in range(nnodes):
            jlist = np.argsort(distmat[i])
            assert jlist[0] == i
            key = nodes[i]
            val = [(nodes[j], distmat[i, j]) for j in jlist[1:]][:self.k]
            knn[key] = val

        # form the edges
        edges = []
        neigh_dist = []
        for u, neighdata in knn.items():
            for v, d in neighdata:
                if (u, v) in edges:
                    continue
                if not self.symmetrize_edges:
                    edges.append((u, v, {"weight": d}))
                    neigh_dist.append(d)
                else:
                    if u in [x[0] for x in knn[v]]:
                        edges.append((u, v, {"weight": d}))
                        neigh_dist.append(d)

        # edge weight normalization:
        # based on the Blatt, Wiseman, Domany paper
        a = np.mean(neigh_dist)
        print("Averge neighborhood radius: %2.2f A" % a)
        for i, e in enumerate(edges):
            d = e[-1]["weight"]
            new_weight = np.exp(-0.5 * (d**2) / (a**2))
            # new_weight = 1.0 / (1 + d)
            edges[i][-1]["weight"] = new_weight
        return edges

    def build(self):
        """
        API method that builds the protein graph step-by-step: segment
        the protein by secondary structure, build nodes, find k-nearest
        neighbors, and finally connect the edges.
        """

        partition_input_dict, seed_region_dict = self._build_partition_input()

        # available labels
        available_rbs = [i for i in range(1, self.maxrb+1)
                         if i not in self._init_seed_rbs]

        # build partitions
        print("\nBuilding partitions for...")
        for src, data in partition_input_dict.items():
            print(src)
            P = Partition(pdb_fn=data[0], regions=data[1],
                          **self._partition_builder_kwargs)
            P.run()
            self._partitions[src] = P.segments

        # add nodes
        print("\nAdding nodes...")
        nodes = []
        for src, segments in self._partitions.items():
            for chain, (p_ss, p_loop) in segments.items():
                # ss nodes
                for p in p_ss:
                    region = Region(src, chain, *p)
                    u = Node(region, struct="ss",
                             rb=random.choice(available_rbs))
                    nodes.append(u)

                # loop nodes
                for p in p_loop:
                    region = Region(src, chain, *p)
                    u = Node(region, struct="loop",
                             rb=random.choice(available_rbs))
                    nodes.append(u)

        print("Added %d nodes" % len(nodes))

        # seed nodes
        if seed_region_dict:
            for rb, regions in seed_region_dict.items():
                u = Node(regions, rb=rb, struct="seed")
                nodes.append(u)

        self.add_nodes_from(nodes)

        # add structure to all nodes
        print("Adding structure to nodes...")
        [u.add_structure(self.models) for u in self.nodes]

        # add edges
        print("\nAdding edges...")
        edges = self._build_kNN_edges()
        self.add_edges_from(edges)
        print("Added %d edges" % len(edges))
        print("Graph connected: ", nx.is_connected(self))

        # map residues to nodes (by accounting for pdb offsets)
        self._build_residue_node_map()

    def set_rbs(self, rbs, nodes=[]):
        """
        Assign rigid body ids to one or more nodes.
        Args:

        rbs (list): List of rigid body ids to be assigned.

        nodes (list, optional): List of nodes to which rigid bodies are
        assigned. Must be same length as the list of rigid bodies.
        Defaults to empty list, in which case all graph nodes are selected.
        """

        if not nodes:
            nodes = list(self.nodes)
        assert len(nodes) == len(rbs)
        for i, u in enumerate(nodes):
            u.set_rb(int(rbs[i]))

    def set_node_weights(self, weights, nodes=[]):
        """
        Assign weights to one or more nodes.
        Args:

        rbs (list): List of node weights to be assigned.

        nodes (list, optional): List of nodes to which weights are
        assigned. Must be same length as the list of weights.
        Defaults to empty list, in which case all graph nodes are selected.
        """

        if not nodes:
            nodes = list(self.nodes)
        assert len(nodes) == len(weights)
        for i, u in enumerate(nodes):
            u.set_weight(weights[i])

    def get_rbs(self):
        """
        Get rigid body assignments of all nodes.

        Returns:
        (list): List of rigid body ids for all graph nodes.
        """

        return np.array([u.rb for u in self.nodes], np.uint8)

    def get_node_weights(self):
        """
        Get weights of all nodes.

        Returns:
        (list): Weights of all graph nodes.
        """

        return np.array([u.weight for u in self.nodes], np.float32)

    def get_rb_dict(self):
        """
        Get dictionary of rigid body assignments of all graph nodes, referenced
        by rigid body id.

        Returns:
        (dict): dict {rb id: list of nodes with this rb id}
        """

        rb_dict = OrderedDict()
        for u in self.nodes:
            if u.rb not in rb_dict:
                rb_dict[u.rb] = []
            rb_dict[u.rb].append(u)
        return rb_dict

    def get_intra_rb_xl_sat(self, node_weight_cutoff=0.0):
        """
        Get intra-rigid-body crosslink satisfaction.

        Args:
        node_weight_cutoff (float, optional): Nodes with weight below this
        cutoff will be ignored. Defaults to 0.0, i.e. all nodes.

        Returns:
        (tuple): Satisfaction statistics (a dict containing number of satisfied
        and mapped crosslinks on each rigid body) and the total number of
        crosslinks used (after ignoring nodes below the cutoff weight).
        """

        if not self.xl_data:
            return ValueError("No crosslinks found")

        xls = [(u, v, sat) for (u, v, sat) in self.xl_data.values()
               if u.weight >= node_weight_cutoff
               and v.weight >= node_weight_cutoff]

        rbs = sorted(self.get_rb_dict())
        stats = OrderedDict()
        for x in rbs:
            stats[x] = {"cov": 0, "sat": 0}

        ntot = len(xls)
        for (u, v, sat) in xls:
            if u.rb == v.rb:
                stats[u.rb]["cov"] += 1
                stats[u.rb]["sat"] += int(sat)
        return stats, ntot

    def save(self, outfn="protein_graph.pkl"):
        """
        Serialize the protein graph and save to a (dill) pickle.

        Args:
        outfn (str, optional): Filename of the output pickle.
        Defaults to 'protein_graph.pkl'
        """

        with open(outfn, "wb") as of:
            dill.dump(self, of)
