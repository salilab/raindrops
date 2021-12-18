"""Tools for input/output processing."""

import os
import string
import numpy as np
import pandas as pd

from Bio.PDB import PDBParser, PDBIO
from Bio.PDB import StructureBuilder

# default cutoff crosslink (typical value for DSS/BS3 linker)
XL_LINKER_CUTOFF = 30.0


def make_struct2seq_map(topology):
    """
    Utility function to map residue location in the PDB file to residue
    location in sequence.
    
    Args:
    topology (io_tools.TopologyParser.topology): Parsed protein complex
    topology from a topology.

    Returns:
    (dict): Dict mapping a (PDB file prefix, chain id, residue id_1) tuple to a
    (molecule name, residue id_2) tuple. The residue id in the key is its 
    id as given in the PDB file, while in the value, it is adjusted by any
    given offset.
    """
    
    struct2seq = {}
    for t in topology:
        molname, src, chain, resrange, pdb_offset, rb = t
        r0, r1 = resrange
        for r in range(r0, r1+1):
            struct2seq[(src, chain, r)] = (molname, r+pdb_offset)
    return struct2seq


def remap_rbs(rbs):
    """
    Utility to re-number rigid body ids so that they can be represented
    easily in UCSF ChimeraX.
    
    Args:
    rbs (list): Rigid body ids (of some or all nodes of a protein graph.)

    Returns:
    (list): Re-numbered rigid body ids.
    """
    unique_rbs = sorted(set(rbs))
    return {i: (ii+1) for (ii, i) in enumerate(unique_rbs)}


class TopologyParser:
    """
    Parse protein complex topology from a PMI (Python Modeling Interface)
    style topology file.
    
    A full description of PMI topology files can be found at:
    https://integrativemodeling.org/2.15.0/doc/ref/classIMP_1_1pmi_1_1topology_1_1TopologyReader.html
    """
    
    def __init__(self, pdb_dir="."):
        """
        Constructor.
        
        Args:
        pdb_dir (str, optional): Directory containing all PDB files.
        Defaults to ".".
        """
        
        self.pdb_dir = os.path.abspath(pdb_dir)
        self.topology = None
        self.models = {}
    
    def _get_all_models(self):
        """Parse Biopython Models from all PDB files."""
        
        srcs = set([t[1] for t in self.topology])
        for src in srcs:
            pdb_fn = os.path.join(self.pdb_dir, src + ".pdb")
            struct = PDBParser(QUIET=True).get_structure("x", pdb_fn)
            self.models[src] = struct[0]
    
    def _refine_residue_ranges(self):
        """Get correct residue ranges for each line in the topology file."""
        
        for i, t in enumerate(self.topology):
            molname, src, chain, resrange, pdb_offset, rb = t
            if not resrange: resrange = "1,END"
            begin, end = resrange.split(",")
            pdb_resids = [r.id[1] 
                        for r in self.models[src][chain].get_residues()]
        
            begin = int(begin)
            if begin == 1 or begin < min(pdb_resids):
                begin = min(pdb_resids)
            
            if isinstance(end, str) and end == "END":
                end = max(pdb_resids)
            else:
                end = int(end)
                if end > max(pdb_resids):
                    end = max(pdb_resids) 
            
            new_resrange = (begin, end)
            this_new_t = (molname, src, chain, new_resrange, pdb_offset, rb)
            self.topology[i] = this_new_t

    def _read_topology_file(self, fn):
        """
        Read a topology file line by line. Adapted from PMI's topology reader.
        https://github.com/salilab/pmi/blob/bc61a9e24192480b66f2239cab406e33f8e3ca68/pyext/src/topology/__init__.py#L1370
        
        Args:
        fn (str): Topology file name.
        """
        
        with open(fn, "r") as of:
            lines_ = of.readlines()
        # remove first line and any line with comments
        lines_ = lines_[1:]
        lines = [line for line in lines_ if not line.startswith("#")]
        # read line-by-line into a list
        top = []
        for line in lines:
            l = line.split("|")[1:]
            molname = l[0].strip() 
            src = l[1].strip()    
            if src == "BEADS": continue  # ignore flexible residues
            chain = l[2].strip()
            resrange = l[3].strip()
            pdb_offset = l[4].strip()
            pdb_offset = 0 if not pdb_offset else int(pdb_offset)
            rb = l[5].strip()
            rb = None if not rb else int(rb)
            top.append((molname, src, chain, resrange, pdb_offset, rb))
        self.topology = top
    
    def parse(self, topology_fn):
        """
        Parse protein complex topology.
        
        Args:
        topology_fn (str): Topology file.
        """
        print("Reading topology from %s" % topology_fn)
        self._read_topology_file(topology_fn)
        self._get_all_models()
        self._refine_residue_ranges()
        
        
class RigidBodyPdbWriter:
    """Write PDB files for each rigid body"""
    
    def __init__(self, graph, min_loop_quality=0.0, pdb_chain_map={}):
        """
        Constructor:
        
        Args:
        graph (networkx Graph): protein_graph.Graph object.
        
        min_loop_quality (float, optional): Min. allowed weight of 
        a loop element. Defaults to 0.0.
        
        pdb_chain_map (dict, optional): Dict mapping molecule names
        to target chain names in the PDB files that'll be written.
        Defaults to empty dict, in which case, molecules are assigned chains
        alphabetically depending on the order in which they appear in the 
        topology.
        """
        
        self.graph = graph
        self.min_loop_quality = min_loop_quality
        
        # pdb writer
        self.pdb_io = PDBIO()
        
        # get chain names for each molecule;
        # all molecule chain names should be supplied 
        # else the dict will be rebuilt
        molnames = set([t[0] for t in self.graph.topology])
        if len(pdb_chain_map) < len(molnames):
            pdb_chain_map = {mn: string.ascii_uppercase[i] 
                             for i, mn in enumerate(molnames)}
        self.pdb_chain_map = pdb_chain_map
            
        # remap rb ids for all graph nodes
        self.rbdict = self.graph.get_rb_dict()
        self.rb_reassignment_map = remap_rbs([i for i in self.rbdict.keys()])
        self.num_rb = len(self.rb_reassignment_map)

        # structure 2 sequence map
        self.struct2seq_map = make_struct2seq_map(self.graph.topology)
    
    def _write_rigid_body(self, rb, outdir):
        """
        Write PDB file for a single rigid body.
        
        Args:
        rb (int): ID of rigid body.
        
        outdir (str): Ouput directory.
        """
        
        target_rb = self.rb_reassignment_map[rb]
        out_pdb_fn = os.path.join(outdir, str(target_rb) + ".pdb")
        
        # valid nodes for this rb
        nodes = [u for u in self.rbdict[rb] 
        if (u.struct != "loop") or (u.weight >= self.min_loop_quality)]
        
        # filter empty rigid bodies
        if not nodes: return
        print(">Writing rigid body %d" % self.rb_reassignment_map[rb])
        
        # all residue keys from these valid nodes
        reskeys = [reskey for u in nodes for reskey in u.residues.keys()]
        
        # create empty hierarchy
        hier = {}
        for (src, chain, r_struct) in reskeys:
            molname, r_seq = self.struct2seq_map[(src, chain, r_struct)]
            if molname not in hier: hier[molname] = {}
            hier[molname][r_seq] = (src, chain, r_struct)
    
        # populate hierarchy with biopython objects
        sb = StructureBuilder.StructureBuilder()
        sb.init_structure("x")
        sb.init_model(0)
        for molname in hier:
            sb.init_seg(" ")
            sb.init_chain(self.pdb_chain_map[molname])
        
            for r in sorted(hier[molname]):
                src, chain, r_struct = hier[molname][r]
                src_residue = self.graph.models[src][chain][r_struct]
                src_atoms = list(src_residue.get_atoms())
            
                sb.init_residue(resname=src_residue.resname,
                                field=" ", resseq=r, icode=" ")
            
                for i, a in enumerate(list(src_atoms)):
                    sb.init_atom(name=a.name, coord=a.coord, b_factor=0.0,
                             occupancy=1.0, altloc=" ", serial_number=i+1,
                             fullname=" " + a.name + " ",
                             element=a.name[0])
                             
        self.pdb_io.set_structure(sb.get_structure())
        self.pdb_io.save(out_pdb_fn)
    
    def write(self, outdir="."):
        """
        Write PDB files for all rigid bodies in the topology.
        
        Args:
        outdir (str, optional): Output directory.. Defaults to ".".
        """
        
        print("\nWriting rigid bodies to %s" % outdir)
        outdir = os.path.abspath(outdir)
        os.makedirs(outdir, exist_ok=True)
        
        for i in sorted(self.rbdict.keys()):
            self._write_rigid_body(i, outdir)
        
        print ("Molecules have been mapped to chains as:")
        for k, v in self.pdb_chain_map.items():
            print(k, "-->", v)


class PMIStyleTopologyWriter:
    """
    Writes a topology file reporting the optimal rigid body decomposition
    of structured parts of an input protein complex. The file format is 
    a reduced version of the PMI (Python Modeling Interface) topology file.
    
    The main difference is that this class writes only the structured
    parts of the input protein complex into the topology file, and the only
    "BEADS" type regions are essentially (structurally) low quality regions
    (such as loops) present in the input file complex.
    
    See here for description of the PMI topology file format:
    https://integrativemodeling.org/2.15.0/doc/ref/classIMP_1_1pmi_1_1topology_1_1TopologyReader.html
    """
    
    def __init__(self, graph, min_loop_quality=0.0):
        """
        Constructor.
        
        Args:
        graph (networkx Graph): protein_graph.Graph object.
        
        min_loop_quality (float, optional): Min. allowed weight of 
        a loop element. Defaults to 0.0.
        """
        
        self.graph = graph
        self.min_loop_quality = min_loop_quality
        
        # structure 2 sequence map
        self.struct2seq_map = make_struct2seq_map(self.graph.topology)
        
        # remap rbs
        rb_reassignment_map = remap_rbs(self.graph.get_rbs())
        for u in self.graph.nodes:
            old_rb = u.rb
            new_rb = rb_reassignment_map[old_rb]
            u.set_rb(new_rb)
        
        # get rb sizes
        self.rb_sizes = {k: len(v) for k, v in self.graph.get_rb_dict().items()}
        
    def _get_molinfo_from_nodes(self, target_molname):
        """
        Get molecule info from protein graph nodes.
        
        Args:
        target_molname (str): Molecule name. Usually a molecule implies
        a complete chain.

        Returns:
        (dict): Dict containing structural source information of given
        molecule, referenced by residue id in the molecule sequence.
        """
        
        info = {}
        for u in self.graph.nodes:
            for (src, chain, r_struct) in u.residues:
                molname, r_seq = self.struct2seq_map[(src, chain, r_struct)]
                if molname != target_molname: continue
                pdb_offset = r_seq - r_struct
                if (u.struct != "loop") or (u.weight >= self.min_loop_quality):
                    this_info = {"src": src, "chain": chain,
                                 "pdb_offset": pdb_offset}
                    this_rigid_body = u.rb
                else:
                    this_info = {"src": "BEADS", "chain": "", "pdb_offset": ""}
                    this_rigid_body = u.rb if self.rb_sizes[u.rb] >  1 else ""
                
                this_info["residue"] = r_struct
                this_info["rigid_body"] = this_rigid_body
                
                info[r_seq] = this_info
        return info 
    
    def _get_components_from_molinfo(self, molname, molinfo):
        """
        Convert information from a molinfo dict to a PMI style component.
        
        Args:
        molname (str): Target molecule name. Usually a molecule implies a 
        complete chain.
        
        molinfo (dict): Dict containing structural source information of given
        molecule, referenced by residue id in the molecule sequence.

        Returns:
        (list): List of strings, one for each component, that'll form the 
        lines of the output PMI style topology file.
        """
        
        residues = sorted(molinfo)
        resranges = []
        this_resrange = [residues[0]]
        for next_r in residues[1:]:
            curr_r = this_resrange[-1]
            # tests to determine if these two residues are in the same component
            is_contiguous_seq = next_r - curr_r == 1
            is_same_src = molinfo[next_r]["src"] == molinfo[curr_r]["src"]
            is_same_chain = molinfo[next_r]["chain"] == molinfo[curr_r]["chain"]
            is_contiguous_struct = molinfo[next_r]["residue"] - \
                                   molinfo[curr_r]["residue"] == 1
            is_same_rigid_body = molinfo[next_r]["rigid_body"] == \
                                 molinfo[curr_r]["rigid_body"]

            same_comp = is_contiguous_seq and is_same_src and \
                        is_same_chain and is_contiguous_struct and \
                        is_same_rigid_body
            
            if same_comp:
                this_resrange.append(next_r)
            else:
                resranges.append(this_resrange)
                this_resrange = [next_r]
        
        # push in the last resrange
        resranges.append(this_resrange)
        
        # correct singleton resranges
        for i, resrange in enumerate(resranges):
            if len(resrange) == 1:
                resranges[i] *= 2 
        
        # create pmi style "components"
        components = []
        for resrange in resranges:
            r0, r1 = resrange[0], resrange[-1]
            this_component_list = [molname,  # molecule
                                   molinfo[r0]["src"], # source (pdb file)
                                   molinfo[r0]["chain"], # pdb chain
                                   "%d,%d" % (r0, r1), # residue-range
                                   str(molinfo[r0]["pdb_offset"]), # offset
                                   str(molinfo[r0]["rigid_body"])] # rigid-body
                                   
            this_component = "|".join(["%-30s" % c 
                                       for c in this_component_list])
            components.append("|" + this_component + "|")
        
        return components
    
    def write(self, topology_fn="PMI_topology.txt"):
        """
        Write the output PMI style topology file.
        
        Args:
        topology_fn (str, optional): Output topoology file name.
        Defaults to "PMI_topology.txt".
        """
        
        print("\nWriting (nearly) similar PMI style topology to %s" % \
            topology_fn)
        
        header_list = ["molecule", "src", "chain", "resrange",
                       "pdb_offset", "rb"]
        
        s = "|" + "|".join(["%-30s" % h for h in header_list]) + "|\n"
        
        molnames = set([t[0] for t in self.graph.topology])
        for this_molname in molnames:
            s += "#\n"
            this_molinfo = self._get_molinfo_from_nodes(this_molname)
            this_components = self._get_components_from_molinfo(this_molname,
                                                                this_molinfo)
            s += "\n".join(this_components) + "\n"
        with open(topology_fn, "w") as of: of.write(s)
        
        
def check_PMI_topology(topology_fn, rigid_body_pdb_dir, src_pdb_dir):
    """
    Check consistency of output topology file and PDB files for a 
    computed rigid body decomposition.

    Args:
    topology_fn (str): PMI style topology file 
    for computed rigid body assignment. 
    
    rigid_body_pdb_dir (str): Output directory containing separate PDB files
    for each computed rigid body.
    
    src_pdb_dir (str): Directory containing input PDB structures for the
    protein complex.

    Returns:
    (tuple): (bool, str) The first entry of the tuple is a boolean that is True
    only if the topology file and PDB files are consistent. The second entry
    is a status message that contains a report of any inconsistencies in chain
    and residue ids.
    """
    
    check = True
    status = ""

    tp = TopologyParser(src_pdb_dir)
    tp.parse(topology_fn)

    # get structured residues of each rigid body from topology file
    rb_resdict = {}
    for t in tp.topology:
        molname, src, chain, resrange, pdb_offset, rb = t
        if src == "BEADS": continue
        if rb not in rb_resdict: rb_resdict[rb] = {}
        if chain not in rb_resdict[rb]: rb_resdict[rb][chain] = []
        r0 = resrange[0] + pdb_offset
        r1 = resrange[1] + pdb_offset
        rb_resdict[rb][chain].extend(list(range(r0, r1+1)))
    
    # get residue ids from pdb files for each rigid body
    for rb, chaindict in rb_resdict.items():
        pdb_fn = os.path.join(rigid_body_pdb_dir, str(rb) + ".pdb")
        model = PDBParser(QUIET=True).get_structure("x", pdb_fn)[0]
        all_chains = [c.id for c in model.get_chains()]
        for c, reslist1 in chaindict.items():
            if c not in all_chains:
                check = False
                status += ">Chain %s found in topology but not in %d.pdb\n" % (c, rb)
                continue
            
            reslist1 = sorted(reslist1)
            reslist2 = sorted([r.id[1] for r in model[c].get_residues()])
            if reslist1 != reslist2:
                check = False
                status += ">Residue numbers of chain %s from topology not found in %d.pdb\n" % (c, rb)
    
    return check, status
