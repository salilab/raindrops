"""
Calculate statistics and analyze protein graph sampling.
"""

import os
import glob
import dill
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RUN_COLORS = ["dodgerblue", "salmon", "darkgreen", "dimgrey", "black"]
MIN_LOOP_QUALITY_RANGE = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7,
                          0.8, 0.9, 1.0]
IDEAL_XL_SAT = 0.90

SAVE_FLOAT_FMT = "%2.2f"
EPS = 1e-9


class Analysis:
    """
    Check convergence of rigid body assignments by stochastic graph
    partitioning through simulated annealing.

    Calculate average performance metrics like crosslink satisfaction,
    compactness of the rigid body decomposition, and structural information
    lost while removing low quality nodes.
    """

    def __init__(self, anneal_dir=".", outdir="analysis", runprefix="run_",
                 min_q=MIN_LOOP_QUALITY_RANGE, nbins=20,
                 last_nframes=10, nskip=1, plt_fmt="png"):
        """
        Constructor.

        Args:
        anneal_dir (str, optional): Directory containing all the
        independent simulated annealing runs. Defaults to ".".

        outdir (str, optional): Output directory. Defaults to "analysis".

        runprefix (str, optional): Prefix of sub-directories,
        each containing one independent simulated annealing run.
        Defaults to "run_".

        min_q (list, optional): List of critical loop-node weights.
        The analysis will be repeated taking each value in the
        list as the min. allowed loop quality. Defaults to
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        nbins (int, optional): Number of bins for histogramming.
        Defaults to 20.

        last_nframes (int, optional): Average statistics over the last
        <last_nframes> frames. Defaults to 10.

        nskip (int, optional): Simulated annealing trajectory reading
        frequency. Defaults to 1.

        plt_fmt (str, optional): Format for output 2D plots. Defaults to "png".
        """

        s = os.path.join(os.path.abspath(anneal_dir), runprefix + "*")
        self.anneal_dirs = sorted(glob.glob(s))
        self.n_runs = len(self.anneal_dirs)

        self.outdir = os.path.abspath(outdir)
        os.makedirs(self.outdir, exist_ok=True)

        self.min_q = min_q
        self.nbins = nbins
        self.last_nframes = last_nframes
        self.nskip = nskip
        self.plt_fmt = plt_fmt

        self.graph = None
        self.restraints = []

        self.extract_graph()
        self.get_restraints()

    def _savefig(self, fig, figname, tight_layout=True):
        """
        Saves a matplotlib figure to disk.

        Args:
        fig: Matplotlib figure object.

        figname (str): Prefix of output file.

        tight_layout (bool, optional): Apply matplotlib's tight layout setting.
        Defaults to True.
        """

        fn = os.path.join(self.outdir, figname + "." + self.plt_fmt)
        if tight_layout:
            fig.tight_layout()
        fig.savefig(fn, bbox_inches="tight", dpi=100)

    def extract_graph(self):
        """Load saved protein graph from (dill) pickle."""

        graph_pkl = os.path.join(self.anneal_dirs[0], "graph.pkl")
        with open(graph_pkl, "rb") as of:
            g = dill.load(of)
        self.graph = g

    def get_restraints(self):
        """Get names of restraints used in graph sampling."""

        fn = os.path.join(self.anneal_dirs[0], "scores.txt")
        with open(fn, "r") as of:
            line = of.readlines()[0]
        self.restraints = [r.split("_")[0] for r in line.strip().split()[1:]]

    def check_scores_convergence(self):
        """Check convergence of scores."""

        nr = len(self.restraints)
        fig1, axs = plt.subplots(1, nr+1, figsize=(5*nr, 5))
        for i, r in enumerate(self.restraints):
            axs[i].set_xlabel("iteration", fontsize=15)
            axs[i].set_title(r, fontsize=15)
        axs[0].set_ylabel("score", fontsize=15)
        axs[-1].set_xlabel("iteration", fontsize=15)
        axs[-1].set_title("Total", fontsize=15)

        for n, d in enumerate(self.anneal_dirs):
            fn = os.path.join(d, "scores.txt")
            this_scores = np.loadtxt(fn, skiprows=1)[0::self.nskip, 1:]
            this_total_score = np.sum(this_scores, axis=1)
            x = [k*self.nskip for k in range(len(this_scores))]
            color = RUN_COLORS[n]
            lbl = "run " + str(n+1)

            for i, r in enumerate(self.restraints):
                axs[i].plot(x, this_scores[:, i], lw=2, color=color, label=lbl)

            axs[i+1].plot(x, this_total_score, lw=2, color=color, label=lbl)

        if self.n_runs > 1:
            for ax in axs:
                ax.legend(loc="best", prop={"size": 12})
        self._savefig(fig1, "scores_convergence")

    # ----------
    # CROSSLINKS
    # ----------
    def check_XL_satisfaction_convergence(self):
        """Check the convergence of crosslink satisfaction metrics."""

        self.graph.set_node_weights([1.] * len(self.graph.nodes))
        xls = list(self.graph.xl_data.values())
        nxl = len(xls)
        max_frac = sum([xl[-1] for xl in xls]) / float(nxl)

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        ax1, ax2, ax3 = axs

        ax1.axhline(IDEAL_XL_SAT, ls="--", lw=1.5, color="black")
        ax1.set_title("Avg. intra-rigid-body XL sat.", fontsize=15)
        ax1.set_xlabel("iteration", fontsize=15)
        ax1.set_ylabel("fraction", fontsize=15)
        ax1.set_ylim([0.75, 1])

        ax2.axhline(IDEAL_XL_SAT, ls="--", lw=1.5, color="black")
        ax2.set_title("Avg. intra-rigid-subcomplex XL sat.", fontsize=15)
        ax2.set_xlabel("iteration", fontsize=15)
        ax2.set_ylim([0.75, 1])

        ax3.axhline(max_frac, ls="--", lw=1.5, color="black")
        ax3.set_title("Avg. overall XL sat.", fontsize=15)
        ax3.set_xlabel("iteration", fontsize=15)
        ax3.set_ylim([0.0, max_frac*1.2])

        for n, d in enumerate(self.anneal_dirs):
            fn = os.path.join(d, "rbs.txt")
            rb_frames = np.loadtxt(fn, dtype=np.uint8)
            frac1, frac2, frac3 = [], [], []

            for i in range(0, len(rb_frames), self.nskip):
                self.graph.set_rbs(rb_frames[i])
                stats, _ = self.graph.get_intra_rb_xl_sat()
                this_frac1 = np.mean([st["sat"] / (float(st["cov"] + EPS))
                                      for st in stats.values()])
                frac1.append(this_frac1)

                this_frac2_num = sum([st["sat"] for st in stats.values()])
                this_frac2_denom = sum([st["cov"] for st in stats.values()])
                this_frac2 = this_frac2_num / (float(this_frac2_denom) + EPS)
                frac2.append(this_frac2)

                this_frac3 = sum([st["sat"] for st in stats.values()])
                this_frac3 /= float(nxl)
                frac3.append(this_frac3)

            # write to file
            outfn = os.path.join(self.outdir,
                                 "XL_sat_convergence_run%d.csv" % n)
            columns = ["intra-rigid-body", "intra-rigid-subcomplex", "overall"]
            df = pd.DataFrame(zip(frac1, frac2, frac3), columns=columns)
            df.to_csv(outfn, index=False, float_format=SAVE_FLOAT_FMT)

            # plot
            x = [k*self.nskip for k in range(len(frac1))]
            color = RUN_COLORS[n]
            lbl = "run " + str(n+1)
            ax1.plot(x, frac1, ls="-", lw=2, color=color, label=lbl)
            ax2.plot(x, frac2, ls="-", lw=2, color=color)
            ax3.plot(x, frac3, ls="-", lw=2, color=color)

        if self.n_runs > 1:
            ax1.legend(loc="best", prop={"size": 12})
        self._savefig(fig, "XL_sat_convergence")

    def calculate_optimal_XL_satisfaction(self):
        """
        Calculate the best crosslink satisfaction possible for different
        allowed critical loop qualities.
        """

        self.graph.set_node_weights([1.] * len(self.graph.nodes))
        xls = list(self.graph.xl_data.values())
        nxl = len(xls)
        max_frac = sum([xl[-1] for xl in xls]) / float(nxl)

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(1, 1, 1)

        ax.axhline(IDEAL_XL_SAT, ls="--", lw=1.5, color="black")
        ax.axhline(max_frac, ls="--", lw=1.5, color="black")
        ax.set_xlabel("min. loop quality", fontsize=15)
        ax.set_ylabel("fraction", fontsize=15)
        ax.set_xlim([0, 1.1])
        ax.set_ylim([0, 1.1])

        frac1, frac2, frac3 = [], [], []

        for n, d in enumerate(self.anneal_dirs):
            rb_fn = os.path.join(d, "rbs.txt")
            rb_frames = np.loadtxt(rb_fn, dtype=np.uint8)
            rb_frames = rb_frames[-self.last_nframes:, :]
            weight_fn = os.path.join(d, "weights.txt")
            weight_frames = np.loadtxt(weight_fn)[-self.last_nframes:, :]

            frac1_ = np.zeros(len(self.min_q))
            frac2_ = np.zeros(len(self.min_q))
            frac3_ = np.zeros(len(self.min_q))

            for i in range(self.last_nframes):
                self.graph.set_rbs(rb_frames[i])
                self.graph.set_node_weights(weight_frames[i])

                for j, q in enumerate(self.min_q):
                    stats, ntot = self.graph.get_intra_rb_xl_sat(q)
                    this_frac1_ = np.mean([st["sat"] / (float(st["cov"] + EPS))
                                           for st in stats.values()])
                    frac1_[j] += this_frac1_

                    this_frac2_num = sum([st["sat"] for st in stats.values()])
                    this_frac2_denom = sum([st["cov"]
                                            for st in stats.values()])
                    this_frac2_ = this_frac2_num / (
                        float(this_frac2_denom) + EPS)
                    frac2_[j] += this_frac2_

                    this_frac3_ = sum([st["sat"] for st in stats.values()])
                    this_frac3_ /= (float(ntot) + EPS)
                    frac3_[j] += this_frac3_

            frac1_ /= self.last_nframes
            frac1.append(frac1_)

            frac2_ /= self.last_nframes
            frac2.append(frac2_)

            frac3_ /= self.last_nframes
            frac3.append(frac3_)

        frac1 = np.array(frac1)
        frac1_mean = np.mean(frac1, axis=0)
        frac1_err = np.std(frac1, axis=0, ddof=1)
        frac1_err = np.nan_to_num(frac1_err, nan=0.0)

        frac2 = np.array(frac2)
        frac2_mean = np.mean(frac2, axis=0)
        frac2_err = np.std(frac2, axis=0, ddof=1)
        frac2_err = np.nan_to_num(frac2_err, nan=0.0)

        frac3 = np.array(frac3)
        frac3_mean = np.mean(frac3, axis=0)
        frac3_err = np.std(frac3, axis=0, ddof=1)
        frac3_err = np.nan_to_num(frac3_err, nan=0.0)

        # write to file
        out_fn = os.path.join(self.outdir, "XL_sat_best.csv")
        columns = ["min_loop_quality",
                   "mean_intra-rigid-body", "std_intra-rigid-body",
                   "mean_intra-rigid-subcomplex", "std_intra-rigid-subcomplex",
                   "mean_overall", "std_overall"]
        df = pd.DataFrame(zip(self.min_q, frac1_mean, frac1_err,
                              frac2_mean, frac2_err,
                              frac3_mean, frac3_err),
                          columns=columns)
        df.to_csv(out_fn, index=False, float_format=SAVE_FLOAT_FMT)

        # plot
        ax.errorbar(self.min_q, frac1_mean, yerr=frac1_err,
                    ls="-", lw=2,
                    marker="o", markersize=6,
                    color="dodgerblue",
                    label="intra-rigid-body XL sat.")

        ax.errorbar(self.min_q, frac2_mean, yerr=frac2_err,
                    ls="-", lw=2,
                    marker="o", markersize=6,
                    color="salmon",
                    label="intra-rigid-subcomplex XL sat.")

        ax.errorbar(
            self.min_q, frac3_mean, yerr=frac3_err,
            ls="-", lw=2,
            marker="o", markersize=6,
            color="darkgreen",
            label="overall XL sat.")

        ax.legend(loc="best", prop={"size": 12})
        self._savefig(fig, "XL_sat_best", tight_layout=False)

    # -----------
    # COMPACTNESS
    # -----------
    def _get_excess_connected_components(self):
        """
        Get excess connected components formed by all nodes belonging
        to a rigid body. The ideal number should be 1, so anything greater
        than 1 is in excess.

        Returns:
        (float): Avg. number of excess connected subgraphs.
        """

        rbdict = self.graph.get_rb_dict()
        excess_num_cps = []
        for rb, nodes in rbdict.items():
            g = nx.Graph()
            g.add_nodes_from(nodes)
            for (u, v) in self.graph.edges:
                if (u in nodes) and (v in nodes):
                    g.add_edge(u, v)
            cps = list(nx.connected_components(g))
            excess_num_cps.append(len(cps)-1)
        return np.mean(excess_num_cps)

    def check_compactness_convergence(self):
        """
        Check convergence of compactness measures of the rigid body
        decomposition.
        """

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        ax1, ax2 = axs

        ax1.set_xlabel("iteration", fontsize=15)
        ax1.set_ylabel("number", fontsize=15)
        ax1.set_title("Number of rigid bodies", fontsize=15)

        ax2.set_ylabel("iteration", fontsize=15)
        ax2.set_ylabel("number", fontsize=15)
        ax2.set_title("Avg. excess connnected subgraphs", fontsize=15)

        n_rb_0 = []
        for n, d in enumerate(self.anneal_dirs):
            fn = os.path.join(d, "rbs.txt")
            rb_frames = np.loadtxt(fn, dtype=np.uint8)
            n_rb, c_rb = [], []

            for i in range(0, len(rb_frames), self.nskip):
                self.graph.set_rbs(rb_frames[i])
                rbdict = self.graph.get_rb_dict()
                n_rb.append(len(rbdict))
                c_rb.append(self._get_excess_connected_components())
                if i == 0:
                    n_rb_0.append(len(rbdict))

            # write to file
            out_fn = os.path.join(self.outdir,
                                  "rigid_body_compactness_run_%d.csv" % n)
            columns = ["num_rigid_bodies", "num_excess_connected_subgraphs"]
            df = pd.DataFrame(zip(n_rb, c_rb), columns=columns)
            df.to_csv(out_fn, index=False, float_format=SAVE_FLOAT_FMT)

            x = [k*self.nskip for k in range(len(n_rb))]
            color = RUN_COLORS[n]
            lbl = "run " + str(n+1)
            ax1.plot(x, n_rb, lw=2, ls="-", color=color, label=lbl)
            ax2.plot(x, c_rb, lw=2, ls="-", color=color)

        ax1.set_ylim([0, max(n_rb_0) + 2])
        if self.n_runs > 1:
            ax1.legend(loc="best", prop={"size": 12})
        self._savefig(fig, "compactness_convergence")

    # -------------------------------------
    # STRUCTURAL INFORMATION LOST / REMOVED
    # -------------------------------------
    def calculate_loop_quality_distribution(self):
        """Calculate the distribution of weights of crosslinked loop nodes."""

        loop_nodes = []
        for (u, v, sat) in self.graph.xl_data.values():
            if u.struct == "loop":
                loop_nodes.append(u)
            if v.struct == "loop":
                loop_nodes.append(v)
        loop_nodes = set(loop_nodes)
        bin_centers, bin_vals = None, []

        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlabel("loop quality", fontsize=15)
        ax.set_ylabel("distribution", fontsize=15)
        ax.set_xlim([0, 1])
        ax.set_xticks(MIN_LOOP_QUALITY_RANGE)

        for n, d in enumerate(self.anneal_dirs):
            fn = os.path.join(d, "weights.txt")
            weight_frames = np.loadtxt(fn)[-self.last_nframes:, :]
            loop_weights = []

            for i in range(self.last_nframes):
                self.graph.set_node_weights(weight_frames[i])
                loop_weights.append([u.weight for u in loop_nodes])

            loop_weights = np.array(loop_weights).flatten()
            this_loop_hist = np.histogram(loop_weights, bins=self.nbins,
                                          range=(0, 1))

            y = this_loop_hist[0]
            x = [0.5*(this_loop_hist[1][i] + this_loop_hist[1][i+1])
                 for i in range(self.nbins)]

            if n == 0:
                bin_centers = x
            bin_vals.append(y)

        bin_vals = np.array(bin_vals)
        bin_vals_mean = np.mean(bin_vals, axis=0)
        bin_vals_err = np.std(bin_vals, axis=0, ddof=1)
        bin_vals_err = np.nan_to_num(bin_vals_err, nan=0.0)

        # write to file
        out_fn = os.path.join(self.outdir, "loop_quality_histogram.csv")
        columns = ["bin_center", "mean_bin_count", "std_bin_count"]
        df = pd.DataFrame(zip(bin_centers, bin_vals_mean, bin_vals_err),
                          columns=columns)
        df.to_csv(out_fn, index=False, float_format=SAVE_FLOAT_FMT)

        # plot
        ax.bar(x=bin_centers, height=bin_vals_mean, yerr=bin_vals_err,
               width=0.1, color="dodgerblue", edgecolor="black", capsize=2)
        self._savefig(fig, "loop_quality_histogram")

    def check_lost_struct_info(self):
        """
        Check how much structural information is lost if loops below
        a critical weight are removed from consideration. Do this for
        different values of the critical loop node weight.
        """

        self.graph.set_node_weights([1.] * len(self.graph.nodes))
        xls = list(self.graph.xl_data.values())

        loop_nodes = []
        loop_xls = []
        for (u, v, sat) in xls:
            if u.struct == "loop":
                loop_nodes.append(u)
            if v.struct == "loop":
                loop_nodes.append(v)
            if (u.struct == "loop" or v.struct == "loop"):
                loop_xls.append((u, v))
        loop_nodes = set(loop_nodes)
        nres_loop = sum([len(u) for u in loop_nodes])

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        ax1, ax2 = axs
        ax1.axhline(len(loop_xls), ls="--", lw=1.5, color="black")
        ax1.set_xlabel("min. loop quality", fontsize=15)
        ax1.set_ylabel("number", fontsize=15)
        ax1.set_title("Number of XLs removed", fontsize=15)

        ax2.axhline(nres_loop, ls="--", lw=2, color="black")
        ax2.set_xlabel("min. loop quality", fontsize=15)
        ax2.set_ylabel("number", fontsize=15)
        ax2.set_title("Number of residues removed", fontsize=15)

        nxl_del, nres_del = [], []

        for n, d in enumerate(self.anneal_dirs):
            weight_fn = os.path.join(d, "weights.txt")
            weight_frames = np.loadtxt(weight_fn)[-self.last_nframes:, :]

            nxl_del_ = np.zeros(len(self.min_q))
            nres_del_ = np.zeros(len(self.min_q))

            for i in range(self.last_nframes):
                self.graph.set_node_weights(weight_frames[i])

                for j, q in enumerate(self.min_q):
                    this_nxl_del_ = sum([1 for (u, v) in loop_xls
                                        if u.weight < q or v.weight < q])
                    nxl_del_[j] += this_nxl_del_

                    this_nres_del_ = sum([len(u) for u in loop_nodes
                                          if u.weight < q])
                    nres_del_[j] += this_nres_del_

            nxl_del_ /= self.last_nframes
            nxl_del.append(nxl_del_)

            nres_del_ /= self.last_nframes
            nres_del.append(nres_del_)

        nxl_del = np.array(nxl_del)
        nxl_del_mean = np.mean(nxl_del, axis=0)
        nxl_del_err = np.std(nxl_del, axis=0, ddof=1)
        nxl_del_err = np.nan_to_num(nxl_del_err, nan=0.0)

        nres_del = np.array(nres_del)
        nres_del_mean = np.mean(nres_del, axis=0)
        nres_del_err = np.std(nres_del, axis=0, ddof=1)
        nres_del_err = np.nan_to_num(nres_del_err, nan=0.0)

        # write to file
        out_fn = os.path.join(self.outdir, "lost_struct_info.csv")
        columns = ["min_loop_quality", "num_XLs_removed",
                   "num_residues_removed"]
        df = pd.DataFrame(zip(self.min_q, nxl_del_mean, nres_del_mean),
                          columns=columns)
        df.to_csv(out_fn, index=False, float_format=SAVE_FLOAT_FMT)

        # plot
        ax1.errorbar(self.min_q, nxl_del_mean, yerr=nxl_del_err,
                     ls="-", lw=2,
                     marker="o", markersize=6,
                     color="dodgerblue")

        ax2.errorbar(self.min_q, nres_del_mean, yerr=nres_del_err,
                     ls="-", lw=2,
                     marker="o", markersize=6,
                     color="dodgerblue")

        self._savefig(fig, "lost_struct_info")
