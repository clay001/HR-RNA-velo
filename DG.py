import sys

sys.path.append('/Users/wangxin/PycharmProjects/Landscape/Landscape/VeloLand/')
import numpy as np
import matplotlib.pyplot as plt
import VeloLand.velocity_estimation as v
import pickle
import os
import scanpy as sc
import anndata
from scvi.dataset import LoomDataset


def plot_with_label(dg):
    plt.figure(figsize=(10, 10))
    v.scatter_viz(dg.embedding[:, 0], dg.embedding[:, 1], c=dg.colorandum, s=2)

    # 打中间的类名标注
    for i in range(max(dg.ca["Clusters"]) + 1):
        ts_m = np.median(dg.embedding[dg.ca["Clusters"] == i, :], 0)
        plt.text(ts_m[0], ts_m[1], str(dg.cluster_labels[dg.ca["Clusters"] == i][0]),
                 fontsize=13, bbox={"facecolor": "w", "alpha": 0.6})
    plt.show()


def preprocess(save_path, file_name, pickle_name, latent):
    print("Loading Data ...")
    dg = v.VelocytoLoom(save_path + file_name)
    colors_dict = {'RadialGlia': np.array([0.95, 0.6, 0.1]), 'RadialGlia2': np.array([0.85, 0.3, 0.1]),
                   'ImmAstro': np.array([0.8, 0.02, 0.1]),
                   'GlialProg': np.array([0.81, 0.43, 0.72352941]), 'OPC': np.array([0.61, 0.13, 0.72352941]),
                   'nIPC': np.array([0.9, 0.8, 0.3]),
                   'Nbl1': np.array([0.7, 0.82, 0.6]), 'Nbl2': np.array([0.448, 0.85490196, 0.95098039]),
                   'ImmGranule1': np.array([0.35, 0.4, 0.82]),
                   'ImmGranule2': np.array([0.23, 0.3, 0.7]), 'Granule': np.array([0.05, 0.11, 0.51]),
                   'CA': np.array([0.2, 0.53, 0.71]),
                   'CA1-Sub': np.array([0.1, 0.45, 0.3]), 'CA2-3-4': np.array([0.3, 0.35, 0.5])}

    # add cluster_labels, colorandum, cluster_ix, cluster_uid
    dg.set_clusters(dg.ca["ClusterName"], cluster_colors_dict=colors_dict)
    
    print("Filtering cells ...")
    # remove cells
    bool_sift = dg.initial_Ucell_size > np.percentile(dg.initial_Ucell_size, 0.4)
    dg.filter_cells(bool_array=dg.initial_Ucell_size > np.percentile(dg.initial_Ucell_size, 0.4))
    dg.ts = np.column_stack([dg.ca["TSNE1"], dg.ca["TSNE2"]])
    # 18140 cells now
    
    print("Filtering genes ...")
    # create detection_level_selected(bool)
    dg.score_detection_levels(min_expr_counts=40, min_cells_express=30)
    # filter
    dg.filter_genes(by_detection_levels=True)
    # 13843 genes now

    # SVR CV vs mean fit, create cv_mean_score, cv_mean_selected(bool)
    dg.score_cv_vs_mean(3000, plot=True, max_expr_avg=35)
    # figure 1
    plt.show()
    if not os.path.exists("./figures"):
        os.makedirs("./figures")
    plt.savefig("./figures/DG_SVR.pdf")
    print("DG_SVR.pdf has already been generated in figures folder.")
    
    # filter
    dg.filter_genes(by_cv_vs_mean=True)
    # 3001 genes now

    # create detection_level_selected(bool)
    # dg.score_detection_levels(min_expr_counts=40, min_cells_express=30, min_expr_counts_U=25, min_cells_express_U=20)
    # # create clu_avg_selected(bool)
    # dg.score_cluster_expression(min_avg_U=0.01, min_avg_S=0.08)
    # dg.filter_genes(by_detection_levels=True, by_cluster_expression=True)

    # similar to "both", create U_norm, S_norm
    # dg._normalize_S(relative_size=dg.S.sum(0),
    #                 target_size=np.mean(dg.S.sum(0)))
    # dg._normalize_U(relative_size=dg.U.sum(0),
    #                 target_size=np.mean(dg.U.sum(0)))
    dg.normalize("both", size=True, log=True)

    # figure 2
    dg.plot_fractions()
    if not os.path.exists("./figures"):
        os.makedirs("./figures")
    plt.savefig("./figures/DG_fractions.pdf")
    print("Show fractions ... \nDG_fractions.pdf has already been generated in figures folder.")
    
    # figure 3
    plot_with_label(dg)
    if not os.path.exists("./figures"):
        os.makedirs("./figures")
    plt.savefig("./figures/DG_label.pdf")
    print("DG_label.pdf has already been generated in figures folder.")

    if os.path.isfile(save_path + pickle_name):
        # pickle load ls
        with open(save_path + pickle_name, 'rb') as file:
            latent = pickle.load(file)
            dg.ls = np.array(latent)
    else:
        dg.ls = latent
    
    dg.ls = dg.ls[bool_sift]
    return dg, bool_sift


def knn_imputation(dg, k=500, assumption="constant_velocity"):
    # knn_imputation
    print("Start KNN pooling with k =", k)
    dg.knn_imputation_VI(n_pca_dims=n_comps，k=k, balanced=True, b_sight=k * 8, b_maxl=k * 4, n_jobs=16)
    
    print("Predict with ", assumption, "assumption ... ")
    # Fit gamma using spliced and unspliced data
    dg.fit_gammas(limit_gamma=False, fit_offset=False)
    # gamma model fit new feature： Upred = gamma * S
    dg.predict_U()
    # U measured 减去 predict
    dg.calculate_velocity()
    # 新特征 delta_S (np.ndarray) – The variation in gene expression
    dg.calculate_shift(assumption=assumption)
    # Sx_sz_t (np.ndarray) – the extrapolated expression profile
    # used_delta_t (float) – stores delta_t for future usage
    dg.extrapolate_cell_at_t(delta_t=1.)


def transition_compute(dg):
    # transition matrix
    print("Start estimate transition prob on")
    dg.estimate_transition_prob(hidim="Sx_sz", embed="embedding", transform="sqrt", psc=1,
                                n_neighbors=2000, knn_random=True, sampled_fraction=0.5)
    # 投影方向
    # transition_prob: the transition probability calculated using the exponential kernel on the correlation coefficient
    # delta_embedding:  The resulting vector
    # Use the transition probability to project the velocity direction on the embedding
    dg.calculate_embedding_shift(sigma_corr=0.05, expression_scaling=False)

    # Calculate the velocity using a points on a regular grid and a gaussian kernel
    dg.calculate_grid_arrows(smooth=0.8, steps=(40, 40), n_neighbors=300)


def small_arrow_plot(dg):
    # 画图
    plt.figure(None, (14, 14))
    quiver_scale = 60

    plt.scatter(dg.embedding[:, 0], dg.embedding[:, 1],
                c="0.8", alpha=0.2, s=10, edgecolor="")

    # ix_choice = np.random.choice(vlm.embedding.shape[0], size=int(vlm.embedding.shape[0]/1.), replace=False)
    # plt.scatter(vlm.embedding[ix_choice, 0], vlm.embedding[ix_choice, 1],
    # c="0.8", alpha=0.4, s=10, edgecolor=(0,0,0,1), lw=0.3, rasterized=True)

    quiver_kwargs = dict(headaxislength=7, headlength=11, headwidth=8, linewidths=0.25, width=0.00045, edgecolors="k",
                         color=dg.colorandum, alpha=1)
    plt.quiver(dg.embedding[:, 0], dg.embedding[:, 1],
               dg.delta_embedding[:, 0], dg.delta_embedding[:, 1],
               scale=quiver_scale, **quiver_kwargs)
    plt.axis("on")
    plt.show()


def big_arrow_plot(dg):
    plt.figure(None, (20, 10))
    dg.plot_grid_arrows(quiver_scale=0.48,
                        scatter_kwargs_dict={"alpha": 0.35, "lw": 0.35, "edgecolor": "0.4", "s": 38,
                                             "rasterized": True}, min_mass=24, angles='xy', scale_units='xy',
                        headaxislength=2.75, headlength=5, headwidth=4.8, minlength=1.5,
                        plot_random=True, scale_type="absolute")
    plt.axis("on")
    plt.show()


def umap_embed(save_path, dg, bool_sift, plot=True):
    gene_dataset = LoomDataset("DentateGyrus.loom", save_path=save_path)

    post_adata = anndata.AnnData(X=gene_dataset.X[bool_sift])
    post_adata.obsm["X_scVI"] = dg.ls
    post_adata.obs['cell_type'] = np.array(dg.cluster_labels)
    sc.pp.neighbors(post_adata, use_rep="X_scVI", n_neighbors=15)
    sc.tl.umap(post_adata, min_dist=0.1)

    fig, ax = plt.subplots(figsize=(7, 6))
    sc.pl.umap(post_adata, color=["cell_type"], ax=ax, show=plot)
    plt.show()
    return post_adata.obsm["X_umap"]


def differential_expression(dg, gene_dataset):
    return
