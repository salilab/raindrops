"""
Automatic script writer for rendering 3D protein graph and rigid body PDB
structures in UCSF ChimeraX.
"""

import os
import numpy as np
from collections import OrderedDict

# colors for rendering partitions
SS_COLOR = "purple"
LOOP_COLOR = "green"

# default colors for graph
DEFAULT_NODE_COLOR = (0.5, 0.5, 0.5)
DEFAULT_EDGE_COLOR = (0.0, 0.0, 0.0)

# SIZES OF GRAPH NODES AND EDGES (CHIMERAX MARKER OBJECTS)
NODE_SCALE = 8.0
EDGE_SCALE = 0.5


def make_colormap(n):
    """
    Get a random colormap for a rigid body decomposition.

    Args:
    n (int): Max. number of colors in the colormap

    Returns:
    (2D numpy array): <n> sets of random (R,G,B) values.
    """

    return np.random.random((n, 3))


def render_partitions(pdb_fn, segments, outfn="partition.cxc"):
    """
    Render partitions (segments) of protein complex. This can be done
    for the entire complex, but it is recommended that you only use
    this to visualize single chains, to reduce clutter.

    Args:
    pdb_fn (str): Input PDB file.

    segments (dict): PDB file segments as computed by the
    rbd.protein_graph.Partition object.

    outfn (str, optional): Output chimerax script file.
    Defaults to 'partition.cxc'.
    """

    s = ""
    s += "open %s\n" % os.path.abspath(pdb_fn)
    s += "hide all\n"
    s += "cartoon\n"
    s += "color %s\n\n" % DEFAULT_NODE_COLOR

    ss_spec = []
    loop_spec = []
    for chain, partition in segments.items():
        p_ss, p_loop = partition
        for (begin, end) in p_ss:
            ss_spec.append("/%s:%d-%d" % (chain, begin, end))
        for (begin, end) in p_loop:
            loop_spec.append("/%s:%d-%d" % (chain, begin, end))
    ss_spec = " ".join(ss_spec)
    loop_spec = " ".join(loop_spec)

    s += "color %s %s\n\n" % (ss_spec, SS_COLOR)
    s += "color %s %s\n\n" % (loop_spec, LOOP_COLOR)

    s += "hide all target p\n"
    with open(outfn, "w") as of:
        of.write(s)


class GraphRenderer:
    """
    Render protein graph using markers and links in ChimeraX.
    See here for more details:
    https://www.cgl.ucsf.edu/chimerax/docs/user/markers.html
    """

    def __init__(self, graph, outdir="."):
        """
        Constructor. (Also writes a .cmm file for ChimeraX, containing
        the marker and link definitions).

        Args:
        graph (networkx Graph): protein_graph.Graph object.

        outdir (str, optional): Output directory. Defaults to ".".
        """

        self.graph = graph
        self.colormap = make_colormap(self.graph.maxrb)

        # calculate node sizes
        nodesize = {}
        for u in self.graph.nodes:
            nodesize[u] = np.sqrt(np.sum((u.coords - u.com)**2) / len(u))
        max_nodesize = max(nodesize.values())

        # calculate edge sizes
        edgesize = {}
        for u, v in self.graph.edges:
            edgesize[(u, v)] = self.graph[u][v]["weight"]
        max_edgesize = max(edgesize.values())

        # write chimerax marker set
        s = ""
        # nodes (markers)
        nodes = list(self.graph.nodes)
        for i, u in enumerate(nodes):
            x, y, z = u.com
            r = NODE_SCALE * (nodesize[u] / max_nodesize)
            s_id = 'marker id="%d"' % (i+1)
            s_shape = 'x="%.3f" y="%.3f" z="%.3f" radius="%.3f"' % \
                      (x, y, z, r)
            s_color = 'r="%.3f" g="%.3f" b="%.3f"' % DEFAULT_NODE_COLOR

            this_s = "<" + s_id + " " + s_shape + " " + s_color + "/>"
            s += this_s + "\n"

        # edges (links)
        self.node2idx = {u: (i+1) for (i, u) in enumerate(nodes)}
        for (u, v) in self.graph.edges:
            r = EDGE_SCALE * (edgesize[(u, v)] / max_edgesize)
            id1, id2 = self.node2idx[u], self.node2idx[v]
            s_id = 'link id1="%d" id2="%d"' % (id1, id2)
            s_shape = 'radius="%.3f"' % r
            s_color = 'r="%.3f" g="%.3f" b="%.3f"' % DEFAULT_EDGE_COLOR
            this_s = "<" + s_id + " " + s_shape + " " + s_color + "/>"
            s += this_s + "\n"

        # XML header and footer
        s = '<marker_set name="protein_graph">' + "\n" + s + "</marker_set>"
        s += "\n"

        self.outdir = os.path.abspath(outdir)
        os.makedirs(self.outdir, exist_ok=True)
        self.topology_fn = os.path.join(outdir, "graph.cmm")
        with open(self.topology_fn, "w") as of:
            of.write(s)

    def _get_transparency(self, weights):
        """
        Make histogram of graph node transparencies based on their weights.

        Args:
        weights (list): List of node weights.

        Returns:
        (dict): Dict of (bin number, transparency) pairs. The transparency
        is discretized to a (0, 1] interval.
        """

        tvals = [(1-w) for w in weights]
        bins = np.linspace(0, 1, 11)
        bins[-1] = 0.95
        bin_indices = [int(t/0.1) for t in tvals]
        out = {b: [] for b in bins}
        for ii, i in enumerate(bin_indices):
            b = bins[i]
            out[b].append(ii+1)
        return out

    def render(self, frame=None):
        """
        Render the protein graph in ChimeraX with colors of a particular
        frame from a graph sampling run.

        Args:
        frame (int, optional): Frame id. Defaults to None.
        """

        if frame is not None:
            # per-frame render
            s = "#frame %d\n\n" % frame
            outfn = os.path.join(self.outdir, "%d.cxc" % frame)
        else:
            # one-shot render
            s = "open %s\n\n" % self.topology_fn
            outfn = os.path.join(self.outdir, "chroma.cxc")

        # color all nodes to default color first
        # this automtically colors all edges to the default color
        s += "color :1-%d rgb(%.3f, %.3f, %.3f)\n" % \
             (len(self.graph.nodes), *DEFAULT_EDGE_COLOR)
        s += "\n"

        # now color nodes according to their RB ID
        for rb, nodes in self.graph.get_rb_dict().items():
            color = self.colormap[rb-1]
            node_indices_str = ",".join([str(self.node2idx[u]) for u in nodes])
            s += "color :%s rgb(%.3f, %.3f, %.3f)\n" % \
                 (node_indices_str, *color)
        s += "\n"

        # add transparencies
        node_weights = [u.weight for u in self.graph.nodes]
        node_transparencies = self._get_transparency(node_weights)
        for t, indices in node_transparencies.items():
            if not indices:
                continue
            node_indices_str = ",".join([str(i) for i in indices])
            s += "transparency :%s %.3f\n" % (node_indices_str, t*100.0)
        s += "\n"

        with open(outfn, "w") as of:
            of.write(s)


class ProteinRenderer:
    """
    Render protein complex and color according to rigid body assignments
    computed from the associated protein graph.
    """

    def __init__(self, graph, pdb_dir=None, outdir="."):
        """
        Constructor. (Also write a .cxc ChimeraX file for the whole protein
        in a single color).

        Args:
        graph (networkx Graph): protein_graph.Graph object.

        pdb_dir (str): Directory containing PDB files for the input
        protein complex.

        outdir (str, optional): Output directory. Defaults to ".".
        """

        self.graph = graph
        self.colormap = make_colormap(self.graph.maxrb)

        if pdb_dir is None:
            pdb_dir = self.graph.pdb_dir
        self.pdb_dir = os.path.abspath(pdb_dir)

        # open all models as cartoon and set everything to default color
        self.src_model_map = OrderedDict()
        s, i = "", 1
        for src in self.graph.models:
            pdb_fn = os.path.join(pdb_dir, src+".pdb")
            s += "open %s\n" % pdb_fn
            self.src_model_map[src] = i
            i += 1
        s += "\n"

        s += "hide all target abcp\n"
        s += "cartoon\n"
        s += "color rbg(%.3f, %.3f, %.3f) target c\n" % DEFAULT_NODE_COLOR

        self.outdir = os.path.abspath(outdir)
        os.makedirs(self.outdir, exist_ok=True)
        self.topology_fn = os.path.join(outdir, "topology.cxc")
        with open(self.topology_fn, "w") as of:
            of.write(s)

    def _get_transparency(self, weights):
        """
        Make histogram of graph node transparencies based on their weights.

        Args:
        weights (list): List of node weights.

        Returns:
        (dict): Dict of (bin number, transparency) pairs. The transparency
        is discretized to a (0, 1] interval.
        """

        tvals = [(1-w) for w in weights]
        bins = np.linspace(0, 1, 11)
        bins[-1] = 0.95
        bin_indices = [int(t/0.1) for t in tvals]
        out = {b: [] for b in bins}
        for ii, i in enumerate(bin_indices):
            b = bins[i]
            out[b].append(ii)
        return out

    def render(self, frame=None):
        """
        Render the protein complex coloring according to the rigid body
        assignment given by the protein graph.

        Args:
        frame (int, optional): Frame id. Defaults to None.
        """

        if frame is not None:
            # per-frame render
            s = "#frame %d\n\n" % frame
            outfn = os.path.join(self.outdir, "%d.cxc" % frame)
        else:
            # one-shot render
            s = "open %s\n\n" % self.topology_fn
            outfn = os.path.join(self.outdir, "chroma.cxc")

        # color all nodes to default color first
        s += "color rgb(%.3f, %.3f, %.3f) target c\n\n" % DEFAULT_NODE_COLOR

        # color nodes according to their RB ID
        for rb, nodes in self.graph.get_rb_dict().items():
            color = self.colormap[rb-1]
            spec = []
            for u in nodes:
                for (src, c, r0, r1) in u.regions:
                    m = self.src_model_map[src]
                    this_spec = "#%d/%s:%d-%d" % (m, c, r0, r1)
                    spec.append(this_spec)
            spec_str = " ".join(spec)
            s += "color %s rgb(%.3f, %.3f, %.3f) target c\n" % \
                 (spec_str, *color)
        s += "\n"

        # add transparencies
        nodelist = list(self.graph.nodes)
        node_weights = [u.weight for u in self.graph.nodes]
        node_transparencies = self._get_transparency(node_weights)
        for t, indices in node_transparencies.items():
            if not indices:
                continue
            spec = []
            nodes = [nodelist[i] for i in indices]
            for u in nodes:
                for (src, c, r0, r1) in u.regions:
                    m = self.src_model_map[src]
                    this_spec = "#%d/%s:%d-%d" % (m, c, r0, r1)
                    spec.append(this_spec)
            spec_str = " ".join(spec)
            s += "transparency %s %3.2f target c\n" % (spec_str, t*100.0)
        s += "\n"

        with open(outfn, "w") as of:
            of.write(s)
