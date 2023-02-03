"""Monte-Carlo movers for stochastic graph partitioning"""

import os
import random
import copy
import itertools
import numpy as np
import networkx as nx
from scipy.stats import norm
from .graphics import GraphRenderer

EPS = 1E-15
TEMP_SCHEDULES = ["linear", "geometric"]


def shuffle_rbs(graph):
    """
    Assign random rigid body ids to protein graph nodes.

    Args:
    graph (networkx.Graph): protein_graph.Graph object
    """

    rbs = random.choices(range(1, graph.maxrb+1), k=len(graph.nodes))
    graph.set_rbs(rbs, nodes=list(graph.nodes))


def shuffle_node_weights(graph, nodes=[]):
    """
    Assign random weights (0 < weight < 1) to selected graph nodes.

    Args:
    graph (networkx.Graph): protein_graph.Graph object

    nodes (list, optional): List of nodes to assign random weights.
    Defaults to empty list, in which case all graph nodes are selected.
    """

    if not nodes:
        nodes = list(graph.nodes)
    for u in nodes:
        u.set_weight(random.random())


class Annealer:
    """
    Simulated annealing wrapper for stochastic partitioning of protein graphs
    into rigid bodies.
    """

    def __init__(self, graph, movers,
                 render_per_frame=False, outdir=".",
                 steps_per_frame=1, nframes=100,
                 max_temp=1.0, min_temp=0.001,
                 temp_schedule="linear"):
        """
        Constructor.

        Args:
        graph (networkx.Graph): protein_graph.Graph object

        movers (list): List of Markov-Chain-Monte-Carlo (MCMC) proposal
        generators. These are mcmc.Mover objects.

        render_per_frame (bool, optional): When True, a ChimeraX script
        will be written for rendering the graph state at each frame.
        Defaults to False.

        outdir (str, optional): Output directory. Defaults to ".".

        steps_per_frame (int, optional): Number of monte-carlo steps
        per iteration. The system temperature is kept constant across
        these steps. Defaults to 1.

        nframes (int, optional): Total number of iterations. The system
        temperature is lowered at each iteration. Defaults to 100.

        max_temp (float, optional): Max. annealing temperature.
        Defaults to 1.0.

        min_temp (float, optional): Min. annealing temperature.
        Defaults to 0.001.

        temp_schedule (str, optional): Temperature reduction schedule.
        Can be one of 'linear' or 'geometric. Defaults to "linear".
        """

        self.graph = graph
        self.movers = movers
        self.render_per_frame = render_per_frame
        self.outdir = os.path.abspath(outdir)

        # iteration steps
        self.niter = nframes * steps_per_frame
        self.steps_per_frame = steps_per_frame

        # temp schedule
        self.min_temp, self.max_temp = min_temp, max_temp
        self.temp = max_temp
        if temp_schedule not in TEMP_SCHEDULES:
            raise TypeError("Temperature schedule can only be one of%s"
                            % ", ".join(TEMP_SCHEDULES))
        self.temp_schedule = temp_schedule
        self._set_alpha()

        if not isinstance(self.movers, list):
            self.movers = [self.movers]
        self.restraints = None
        self.renderer = None
        self.setup()

    def _set_alpha(self):
        """
        Set alpha parameter for temperature schedules.
        """

        n = int(self.niter / self.steps_per_frame)
        if self.temp_schedule == "linear":
            alpha = (self.max_temp - self.min_temp) / n
        elif self.temp_schedule == "geometric":
            alpha = (self.min_temp / (self.max_temp + EPS)) ** (1/n)
        else:
            alpha = 1.0
        self.alpha = alpha

    def render(self, frame):
        """
        Render the graph state at each frame in ChimeraX.
        Args:
        frame (int): Frame number.
        """

        if not self.render_per_frame:
            return
        if self.render is None:
            raise AttributeError("Renderer has not been setup")
        self.renderer.render(frame)

    def setup(self):
        """
        Sets up the optimizer by creating file paths and linking
        a graphics.GraphRenderer object to render graph states in ChimeraX.
        """

        print("\nSetting up optimizer; opening score and rb files.")

        # add restraints
        self.restraints = list({r for m in self.movers for r in m.restraints})

        # file paths
        self.chimerax_dir = os.path.join(self.outdir, "chimerax")
        os.makedirs(self.chimerax_dir, exist_ok=True)

        self.rb_fn = os.path.join(self.outdir, "rbs.txt")
        self.of_rb = open(self.rb_fn, "w")

        self.weight_fn = os.path.join(self.outdir, "weights.txt")
        self.of_weight = open(self.weight_fn, "w")

        self.score_fn = os.path.join(self.outdir, "scores.txt")
        self.of_score = open(self.score_fn, "w")

        self.stat_fn = os.path.join(self.outdir, "stats.txt")
        self.of_stat = open(self.stat_fn, "w")

        # open score file for writing
        score_header_str = " ".join(["%8s" % r.label for r in self.restraints])
        score_header_str = "frame" + " " + score_header_str + "\n"
        self.of_score.write(score_header_str)

        # setup the renderer
        self.renderer = GraphRenderer(graph=self.graph,
                                      outdir=self.chimerax_dir)

        # add labels to movers if not provided
        for i, m in enumerate(self.movers):
            if not m.label.split("_"):
                m.set_label(str(i))

    def write(self, frame):
        """
        Write scores for a frame (i.e. iteration) to file.

        Args:
        frame (int): Frame number.
        """

        rbs = self.graph.get_rbs()
        rb_str = " ".join([str(x) for x in rbs]) + "\n"
        self.of_rb.write(rb_str)

        weights = self.graph.get_node_weights()
        weights_str = " ".join(["%.3f" % x for x in weights]) + "\n"
        self.of_weight.write(weights_str)

        scores = [r.get_total_score() for r in self.restraints]
        score_str = " ".join(["%3.3f" % x for x in scores])
        score_str = "%6d" % frame + "\t" + score_str + "\n"
        self.of_score.write(score_str)

        stdout = ["%s: %.3f" % (self.restraints[i].label, scores[i])
                  for i in range(len(scores))]
        stdout.append("total: %3.3f" % sum(scores))

        n_rbs = len(set(self.graph.get_rbs()))
        stdout.append("num_rigid_bodies: %5d" % n_rbs)

        stdout_str = "frame: %8d" % frame + "\t" + \
                     "temp: %.3f" % self.temp + "\t" + \
                     ", ".join(stdout)
        print(stdout_str)

    def close(self):
        """Close all open score and stat. files."""

        for of in [self.of_rb, self.of_weight, self.of_score, self.of_stat]:
            of.close()

    def reduce_temp(self):
        """
        Reduce system temperature by one alpha parameter step, depending
        on the reduction schedule.
        """

        if self.temp_schedule == "linear":
            self.temp -= self.alpha

        elif self.temp_schedule == "geometric":
            self.temp *= self.alpha

    def run(self):
        """Run simulated annealing."""

        # statistic collectors
        nacc = {m.label: 0.0 for m in self.movers}
        natt = {m.label: 0.0 for m in self.movers}

        # main loop
        for i in range(self.niter):
            if not (i % self.steps_per_frame):
                # report to disk and screen
                frame = i / self.steps_per_frame
                self.render(frame)
                self.write(frame)

                # change temperature
                self.reduce_temp()

            # monte-carlo move
            for m in self.movers:
                this_nacc, this_natt = m.move(self.temp)
                nacc[m.label] += this_nacc
                natt[m.label] += this_natt

        # calculate mcmc stats
        s = ""
        for m in self.movers:
            label = m.label
            this_acc_ratio = 100.0 * float(nacc[label]) / float(natt[label])
            s += "mover: %20s, acceptance ratio: %2.2f%%\n" % \
                 (label, this_acc_ratio)
        print("\n" + s)
        self.of_stat.write(s)

        print("\nFinished optimization, closing score and rb files.")
        self.close()


class Mover:
    """
    Base class for Monte-Carlo movers that change the state
    (i.e. rigid body assignment) of one or more graph nodes
    depending on restraint scores.
    """

    def __init__(self, graph, restraints, label=""):
        """
        Constructor.

        Args:
        graph (networkx.Graph): protein_graph.Graph object.

        restraints (list): List of graph restraints which are
        restraints.Restraint objects.

        label (str, optional): Name for this mover for logging to file.
        Defaults to "".
        """

        self.graph = graph
        self.restraints = restraints
        if not isinstance(self.restraints, list):
            self.restraints = [self.restraints]
        self.label = self.__class__.__name__ + "_" + label

    def move(self, temp=1.0):
        """
        Move the graph state, i.e. make a proposal to change the
        rigid body assignments of one or more nodes, then accept / reject
        the proposal and update the state of all nodes affected.

        Defined in subclasses.

        Args:
        temp (float, optional): System temperature. Defaults to 1.0.
        """

        pass

    def set_label(self, label=""):
        """
        Set a label for this mover.

        Args:
        label (str, optional): New label. Defaults to "".
        """

        self.label = self.__class__.__name__ + label


class GibbsRigidBodyMover(Mover):
    """
    Independent Gibbs mover. Randomly flips the state of one graph node
    at a time.
    """

    def move(self, temp=1.0):
        """
        Flip the rigid body assignment of a single node at a time. Then use
        the Metropolis-Hastings criteria to accept or reject the change,
        and update the node state accordingly.

        Args:
        temp (float, optional): System temperature for calculating
        Metropolis-Hastings ratio. Defaults to 1.0.

        Returns:
        (tuple): number of accepted and attempted moves. The latter is always
        1 for this mover.
        """

        nacc, natt = 0, 1
        u = random.choice(list(self.graph.nodes))
        nacc += int(self._move(u, T=temp))
        return nacc, natt

    def _get_log_post_ratio(self, u, T):
        """
        Calculate log of Metropolis-Hastings ratio.

        Args:

        u (rotein_graph.Node): Graph Node object.

        T (float): Temperature.

        Returns:
        (tuple): log of the MH ratio, and a dict of updates to restraint scores
        resulting from the move.
        """

        new_scores = {}
        this_score = None
        for r in self.restraints:
            if "Compactness" in r.label:
                this_score = r.evaluate(nodes=[u])
            elif "Crosslink" in r.label:
                this_score = r.evaluate(nodes=[u])
            elif "SubgraphConnectivity" in r.label:
                this_score = r.evaluate(nodes=[u])
            else:
                continue
            new_scores[r.label] = this_score

        delta_score = 0.0
        for r in self.restraints:
            for k, v in new_scores[r.label].items():
                delta_score += (v - r.scores[k])
        log_post_ratio = -delta_score / (T+EPS)
        return log_post_ratio, new_scores

    def _move(self, u, T):
        """
        Actual implementation of the Gibbs single node flip move.

        Args:
        u (protein_graph.Node): Graph Node object.

        T (float): Temperature.

        Returns:
        (bool): True if the move was accepted, otherwise False.
        """

        accepted = False
        curr_rb = copy.copy(u.rb)
        new_rb = random.choice([x for x in range(1, self.graph.maxrb+1)
                                if x != curr_rb])
        u.set_rb(new_rb)

        log_post_ratio, new_scores = self._get_log_post_ratio(u, T)
        MH_ratio = np.exp(log_post_ratio)
        pacc = min(1, MH_ratio)
        if pacc >= random.random():
            accepted = True
            for r in self.restraints:
                r.update(new_scores[r.label])
        else:
            u.set_rb(curr_rb)
        return accepted


class SWCRigidBodyMover(Mover):
    """
    Efficient cluster mover that implements the
    Swendsen-Wang-Cuts (SWC-1) algorithm described in:
    'Generalizing Swendsen-Wang to Sampling Arbitrary Posterior Probabilities'
    Barbu & Zhu, IEEE T. Pattern Anal., 27(8), 2005
    """

    # probabilities for flipping a node state to a neighbor node,
    p_flip = (10.0, 1.0, 0.1)

    def set_flip_probabilities(self, p=None):
        """
        Set the node flipping probabilities. Default values are (10, 1, 0.1)

        Args:
        p_flip (tuple, optional): Tuple of three probabilities for flipping
        the state of a cluster (i.e. collection of nodes belonging to
        the same rigid body) to:
        (i) that of a node adjacent to this cluster.
        (ii) that of a node non-adjacent to this cluster.
        (iii) A completely new rigid body id.
        Defaults to None.
        """

        if p is not None:
            self.p_flip = p

    def move(self, temp=1.0):
        """
        Identify clusters of correlated states (i.e. rigid body ids) using
        the Swendsen Wang cluster growth algorithm, then flip all of their
        states at once. Accept or reject the change based on the Metropolis-
        Hastings criteria.

        Args:
        temp (float, optional): System temperature for calculating
        Metropolis-Hastings ratio. Defaults to 1.0.

        Returns:
        (tuple): number of accepted and attempted moves.
        """

        nacc, natt = 0, 1

        # grow cluster
        R = self._get_SW_cluster()
        assert all([u.rb == R[0].rb for u in R])
        curr_rb = copy.copy(R[0].rb)

        # ----------
        # SWC-1 algo
        # ----------
        # 1) propose new rigid body id for the cluster nodes
        w_forward = self._get_cluster_reassignment_weights(R)
        new_rb = random.choices(range(1, self.graph.maxrb+1),
                                weights=w_forward, k=1)[0]
        self._flip_SW_cluster(R, new_rb)
        w_backward = self._get_cluster_reassignment_weights(R)

        # 2) calculate log proposal ratio
        term1 = np.log(w_backward[curr_rb-1]+EPS) - \
            np.log(w_forward[new_rb-1] + EPS)
        term2 = self._get_log_connectivity(R, new_rb) - \
            self._get_log_connectivity(R, curr_rb)
        log_prop_ratio = term1 + term2

        # 3) calculate log posterior ratio
        log_post_ratio, new_scores = self._get_log_post_ratio(R, temp)

        # 4) Metropolis-Hastings acceptance / rejection
        MH_ratio = np.exp(log_prop_ratio + log_post_ratio)
        pacc = min(1, MH_ratio)
        if pacc >= random.random():
            nacc += 1
            for r in self.restraints:
                r.update(new_scores[r.label])
        else:
            self._flip_SW_cluster(R, curr_rb)

        return nacc, natt

    def _get_SW_cluster(self):
        """
        Grow a cluster of nodes based on the Swendsen-Wang algorithm.
        (https://en.wikipedia.org/wiki/Swendsen%E2%80%93Wang_algorithm)

        Returns:
        (list): Protein graph nodes in a cluster.
        """

        G = nx.Graph()
        G.add_nodes_from(self.graph.nodes)
        for (u, v) in self.graph.edges:
            if u.rb == v.rb and \
               self.graph[u][v]["weight"] >= random.random():
                G.add_edge(u, v)
        components = list(nx.connected_components(G))
        R = random.choice(components)
        return list(R)

    def _flip_SW_cluster(self, R, target_rb):
        """
        Flip the state (i.e. rigid body id) of all nodes in a cluster.

        Args:
        R (list): Cluster of protein graph nodes.

        target_rb (int): Re-assign all nodes in the cluster to
        this rigid body id.
        """

        for u in R:
            u.set_rb(target_rb)

    def _get_complementary_bipartite_edges(self, U, V):
        """
        Get a set of bipartite edges between complementary nodes
        from two given sets.

        Args:
        U (list): Protein graph node set.

        V (list): Protein graph node set.

        Returns:
        (list): Tuples of edges (u, v) such that u \\in U, and v \\in V - U
        """

        U_complement = [v for v in V if v not in U]
        return [(s, t) for (s, t) in itertools.product(U, U_complement)
                if (s, t) in self.graph.edges]

    def _get_cluster_reassignment_weights(self, R):
        """
        Get probabilities for reassigning the state of a cluster to other
        available states (rigid body ids).

        Args:
        R (list): Cluster of protein graph nodes.

        Returns:
        (list): Re-assignment weights for each available rigid body id
        in the range (1, max. # of rbs allowed for this graph).
        """

        active_rbs = {rb for rb in self.graph.get_rbs()}

        cbe = self._get_complementary_bipartite_edges(R, self.graph.nodes)
        adjacent_rbs = {v.rb for (u, v) in cbe}

        weights = []
        for rb in range(1, self.graph.maxrb+1):
            if rb not in active_rbs:
                weights.append(self.p_flip[2])
            elif rb not in adjacent_rbs:
                weights.append(self.p_flip[1])
            else:
                weights.append(self.p_flip[0])

        curr_rb = R[0].rb
        weights[curr_rb-1] = 0
        return weights

    def _get_log_connectivity(self, R, target_rb):
        """
        Calculate the log-connectivity. This is a factor that is modifies
        the overall MH ratio calculation to account for detailed balance.

        For more details, please see:
        'Generalizing Swendsen-Wang to Sampling Arbitrary Posterior
        Probabilities' Barbu & Zhu, IEEE T. Pattern Anal., 27(8), 2005

        Args:
        R (list): Cluster of protein graph nodes.

        target_rb (int): Proposed rigid body id for all nodes in this cluster.

        Returns:
        (float): log of connectivity for the proposed re-assignment.
        """

        rbdict = self.graph.get_rb_dict()
        try:
            other = rbdict[target_rb]
        except KeyError:
            other = []

        out = 0.0
        cbe = self._get_complementary_bipartite_edges(R, other)
        for (u, v) in cbe:
            out += np.log(1.0 - self.graph[u][v]["weight"] + EPS)
        return out

    def _get_log_post_ratio(self, R, T):
        """
        Calculate log of Metropolis-Hastings ratio.

        Args:

        u (protein_graph.Node): Graph Node object.

        T (float): Temperature.

        Returns:
        (tuple): log of the MH ratio, and a dict of updates to restraint scores
        resulting from the move.
        """

        new_scores = {}
        this_score = None
        for r in self.restraints:
            if "Compactness" in r.label:
                this_score = r.evaluate(nodes=R)
            elif "Crosslink" in r.label:
                this_score = r.evaluate(nodes=R)
            elif "SubgraphConnectivity" in r.label:
                this_score = r.evaluate(nodes=R)
            else:
                continue
            new_scores[r.label] = this_score

        delta_score = 0.0
        for r in self.restraints:
            for k, v in new_scores[r.label].items():
                delta_score += (v - r.scores[k])
        log_post_ratio = -delta_score / (T+EPS)
        return log_post_ratio, new_scores


class CrosslinkedLoopMover(Mover):
    """
    Mover for the node weight of loop nodes that have crosslinked residues
    on them. These nodes are deemed to be of lower structural quality
    than secondary structured nodes with crosslinks, or loop nodes without
    crosslinks.
    """

    # gaussian proposal generator params
    sigma = 0.1
    prob = 1

    def __init__(self, graph, restraints, label=""):
        super().__init__(graph, restraints,
                         label="xl-loop-mover_%s" % label)

        self.nodes = []
        for (u, v, sat) in self.graph.xl_data.values():
            if sat:
                continue
            if u.struct == "loop":
                self.nodes.append(u)
            if v.struct == "loop":
                self.nodes.append(v)
        self.nodes = set(self.nodes)

    def set_sigma(self, sigma=None):
        """
        Set variance for normal proposals.

        Args:
        sigma (float, optional): Proposal variance (must be a fraction
        between 0 and 1). Defaults to None.
        """

        if sigma is None:
            return

        if not (0 < sigma < 1):
            raise ValueError("Proposal variance must be in (0, 1)")
        self.sigma = sigma

    def set_move_prob(self, p=None):
        """
        Set probability of this move.

        Args:
        p (float, optional): Move probability (should be in [0,1]).
        Defaults to None.
        """

        if p is not None:
            return

        if not (0 <= p <= 1):
            raise ValueError("Move probability should be in [0,1]")
            self.prob = p

    def move(self, temp=1.0):
        """
        Make indepedent random walk Metropolis moves on the weight of all
        crosslinked loop nodes.

        Args:
        temp (float, optional): System temperature for calculating
        Metropolis-Hastings ratio. Defaults to 1.0.

        Returns:
        (tuple): number of accepted and attempted moves.
        """

        nacc, natt = 0, 0
        if self.prob >= random.random():
            for u in self.nodes:
                natt += 1
                nacc += int(self._move(u, T=temp))
        return nacc, natt

    def _get_log_post_ratio(self, u, T):
        """
        Calculate log of Metropolis-Hastings ratio.

        Args:

        u (protein_graph.Node): Graph Node object.

        T (float): Temperature.

        Returns:
        (tuple): log of the MH ratio, and a dict of updates to restraint scores
        resulting from the move.
        """

        new_scores = {}
        this_score = None
        for r in self.restraints:
            if "Crosslink" in r.label:
                this_score = r.evaluate(nodes=[u])
            elif "StructuralCoverage" in r.label:
                this_score = r.evaluate(nodes=[u])
            else:
                continue
            new_scores[r.label] = this_score

        delta_score = 0.0
        for r in self.restraints:
            for k, v in new_scores[r.label].items():
                delta_score += (v - r.scores[k])
        log_post_ratio = -delta_score / (T+EPS)
        return log_post_ratio, new_scores

    def _move(self, u, T):
        """
        Actual implementation of the RWMC move for a single node.

        Args:
        u (protein_graph.Node): Graph Node object.

        T (float): Temperature.

        Returns:
        (bool): True if the move was accepted, otherwise False.
        """

        accepted = False
        curr_weight = copy.copy(u.weight)

        new_weight = 999.0
        while not (0.0 <= new_weight <= 1.0):
            new_weight = curr_weight + norm.rvs(scale=self.sigma)
        u.set_weight(new_weight)

        log_post_ratio, new_scores = self._get_log_post_ratio(u, T)
        MH_ratio = np.exp(log_post_ratio)
        pacc = min(1, MH_ratio)
        if pacc >= random.random():
            accepted = True
            for r in self.restraints:
                r.update(new_scores[r.label])
        else:
            u.set_weight(curr_weight)
        return accepted
