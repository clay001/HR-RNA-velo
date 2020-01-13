import sys
import VeloLand.velocity_estimation as v
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import scanpy as sc
import anndata
import umap
import time
from scvi.dataset import LoomDataset
from scipy.stats import norm
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE

sys.path.append('/Users/wangxin/PycharmProjects/Landscape/Landscape/VeloLand/')


class Caltime:
    def __init__(self,func):
        self.func = func

    def __call__(self, *args, **kwargs):
        start = time.perf_counter()
        print("Handling the dataset now ....")
        self.func(*args)
        end = time.perf_counter()
        print(f"Done, running time: {str(end - start)}")


def plot_with_label(dg):
    plt.figure(figsize=(10, 10))
    v.scatter_viz(dg.embedding[:, 0], dg.embedding[:, 1], c=dg.colorandum, s=2)

    # 打中间的类名标注
    for i in range(max(dg.ca["Clusters"]) + 1):
        ts_m = np.median(dg.embedding[dg.ca["Clusters"] == i, :], 0)
        plt.text(ts_m[0], ts_m[1], str(dg.cluster_labels[dg.ca["Clusters"] == i][0]),
                 fontsize=13, bbox={"facecolor": "w", "alpha": 0.6})
    plt.show()
    if not os.path.exists("./figures"):
        os.makedirs("./figures")
    plt.savefig("./figures/label_pic.pdf")
    print("label_pic.pdf has already been generated in figures folder.")


def preprocess(dg, save_path, pickle_name, latent):
    """
    :param dg: target object
    :param save_path: target dir location
    :param pickle_name: latent space name
    :param latent: if there is already existed a latent
    :return: bool_sift: the filtered cell list  *(processed object)
    """
    start = time.perf_counter()
    print("Handling the dataset now ....")
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

    # remove cells
    print("Filtering cells ... ")
    bool_sift = dg.initial_Ucell_size > np.percentile(dg.initial_Ucell_size, 0.4)
    dg.filter_cells(bool_array=dg.initial_Ucell_size > np.percentile(dg.initial_Ucell_size, 0.4))
    dg.ts = np.column_stack([dg.ca["TSNE1"], dg.ca["TSNE2"]])
    # 18140 cells now

    print("Filtering genes ... ")
    # create detection_level_selected(bool)
    dg.score_detection_levels(min_expr_counts=40, min_cells_express=30)
    # filter
    dg.filter_genes(by_detection_levels=True)
    # 13843 genes now

    print("Start SVR ... ")
    # SVR CV vs mean fit, create cv_mean_score, cv_mean_selected(bool)
    dg.score_cv_vs_mean(3000, plot=True, max_expr_avg=35)
    # figure 1
    if not os.path.exists("./figures"):
        os.makedirs("./figures")
    plt.savefig("./figures/CV.pdf")
    print("CV.pdf has already been generated in figures folder.")

    # filter
    dg.filter_genes(by_cv_vs_mean=True)
    # 3001 genes now

    # # create detection_level_selected(bool)
    # dg.score_detection_levels(min_expr_counts=40, min_cells_express=30, min_expr_counts_U=25, min_cells_express_U=20)
    # # create clu_avg_selected(bool)
    # dg.score_cluster_expression(min_avg_U=0.01, min_avg_S=0.08)
    # dg.filter_genes(by_detection_levels=True, by_cluster_expression=True)
    # # 2159 genes now

    # similar to "both", create U_norm, S_norm
    dg._normalize_S(relative_size=dg.S.sum(0),
                    target_size=np.mean(dg.S.sum(0)))
    dg._normalize_U(relative_size=dg.U.sum(0),
                    target_size=np.mean(dg.U.sum(0)))

    if os.path.isfile(save_path + pickle_name):
        # pickle load ls
        with open(save_path + pickle_name, 'rb') as file:
            latent = pickle.load(file)
            dg.ls = np.array(latent)
    else:
        dg.ls = latent

    dg.ls = dg.ls[bool_sift]
    end = time.perf_counter()
    print(f"Done, running time: {str(end - start)}")

    return bool_sift


def knn_imputation_ls(dg, k=500, assumption="constant_velocity"):
    start = time.perf_counter()
    print("KNN pooling on latent space ... ")
    # knn_imputation
    dg.knn_imputation_VI(k=k, balanced=True, b_sight=k * 8, b_maxl=k * 4, n_jobs=16)
    # Fit gamma using spliced and unspliced data
    print("Fitting parameters")
    dg.fit_gammas(limit_gamma=False, fit_offset=False)
    # gamma model fit new feature： Upred = gamma * S

    print("Velocity estimation ... ")
    dg.predict_U()
    # U measured 减去 predict
    dg.calculate_velocity()
    # 新特征 delta_S (np.ndarray) – The variation in gene expression
    dg.calculate_shift(assumption=assumption)

    # Sx_sz_t (np.ndarray) – the extrapolated expression profile
    # used_delta_t (float) – stores delta_t for future usage
    dg.extrapolate_cell_at_t(delta_t=1.)
    end = time.perf_counter()
    print(f"Done, running time: {str(end - start)}")



def transition_compute(dg, space="umap"):
    start = time.perf_counter()
    print("Compute transition probability ... ")
    # transition matrix
    if (space == "tsne"):
        dg.embedding = dg.ts
    elif (space == "umap"):
        dg.embedding = dg.umap
    dg.estimate_transition_prob(hidim="Sx_sz", embed="embedding", transform="sqrt", psc=1,
                                n_neighbors=2000, knn_random=True, sampled_fraction=0.5)
    # 投影方向
    # transition_prob: the transition probability calculated using the exponential kernel on the correlation coefficient
    # delta_embedding:  The resulting vector
    # Use the transition probability to project the velocity direction on the embedding
    dg.calculate_embedding_shift(sigma_corr=0.05, expression_scaling=False)

    # Calculate the velocity using a points on a regular grid and a gaussian kernel
    dg.calculate_grid_arrows(smooth=0.8, steps=(40, 40), n_neighbors=300)
    end = time.perf_counter()
    print(f"Done, running time: {str(end - start)}")

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
    if not os.path.exists("./figures"):
        os.makedirs("./figures")
    plt.savefig("./figures/embed_arrow.pdf")
    print("embed_arrow.pdf has already been generated in figures folder.")

def big_arrow_plot(dg):
    plt.figure(None, (20, 10))
    dg.plot_grid_arrows(quiver_scale=0.48,
                        scatter_kwargs_dict={"alpha": 0.35, "lw": 0.35, "edgecolor": "0.4", "s": 38,
                                             "rasterized": True}, min_mass=24, angles='xy', scale_units='xy',
                        headaxislength=2.75, headlength=5, headwidth=4.8, minlength=1.5,
                        plot_random=True, scale_type="absolute")
    plt.axis("on")
    if not os.path.exists("./figures"):
        os.makedirs("./figures")
    plt.savefig("./figures/grid_arrow.pdf")
    print("grid_arrow.pdf has already been generated in figures folder.")

def re_ts_embed(dg, space =dg.pcs[:, :25]):
    dg.ts = TSNE().fit_transform(space)

def re_umap_embed(dg, space):
    dg.umap = umap.UMAP().fit_transform(space)


def _umap_embed(save_path, dg, bool_sift, plot=True):
    '''
    :param save_path: target dir location
    :param dg: target object
    :param bool_sift: cell filter
    :param plot: if need plot
    :return: None (create dg.umap)
    '''
    start = time.perf_counter()
    print("Create UMAP embedding ... ")
    gene_dataset = LoomDataset("DentateGyrus.loom", save_path=save_path)

    post_adata = anndata.AnnData(X=gene_dataset.X[bool_sift])
    post_adata.obsm["X_scVI"] = dg.ls
    post_adata.obs['cell_type'] = np.array(dg.cluster_labels)
    sc.pp.neighbors(post_adata, use_rep="X_scVI", n_neighbors=15)
    sc.tl.umap(post_adata, min_dist=0.1)

    fig, ax = plt.subplots(figsize=(7, 6))
    sc.pl.umap(post_adata, color=["cell_type"], ax=ax, show=plot)
    plt.savefig("./figures/umap_embedding.pdf")
    print("umap_embedding.pdf has already been generated in figures folder.")

    dg.umap = post_adata.obsm["X_umap"]
    end = time.perf_counter()
    print(f"Done, running time: {str(end - start)}")

def K_neighbors_zoom(dg, xlim:list, ylim:list):
    plt.figure(None, (6, 6))

    def gaussian_kernel(X, mu=0, sigma=1):
        return np.exp(-(X - mu) ** 2 / (2 * sigma ** 2)) / np.sqrt(2 * np.pi * sigma ** 2)
    # 步长
    steps = 45, 45
    grs = []
    for dim_i in range(dg.embedding.shape[1]):
        m, M = np.min(dg.embedding[:, dim_i]), np.max(dg.embedding[:, dim_i])
        m = m - 0.025 * np.abs(M - m)
        M = M + 0.025 * np.abs(M - m)
        gr = np.linspace(m, M, steps[dim_i])
        grs.append(gr)

    meshes_tuple = np.meshgrid(*grs)
    gridpoints_coordinates = np.vstack([i.flat for i in meshes_tuple]).T
    gridpoints_coordinates = gridpoints_coordinates + norm.rvs(loc=0, scale=0.15, size=gridpoints_coordinates.shape)

    nn = NearestNeighbors()
    nn.fit(dg.embedding)
    dist, ixs = nn.kneighbors(gridpoints_coordinates, 20)
    ix_choice = ixs[:, 0].flat[:]
    ix_choice = np.unique(ix_choice)

    nn = NearestNeighbors()
    nn.fit(dg.embedding)
    dist, ixs = nn.kneighbors(dg.embedding[ix_choice], 20)
    density_extimate = gaussian_kernel(dist, mu=0, sigma=0.5).sum(1)
    bool_density = density_extimate > np.percentile(density_extimate, 25)
    ix_choice = ix_choice[bool_density]

    plt.scatter(dg.embedding[:, 0], dg.embedding[:, 1],
                c=dg.colorandum, alpha=0.2, s=120, edgecolor="")
    plt.scatter(dg.embedding[ix_choice, 0], dg.embedding[ix_choice, 1],
                c=dg.colorandum[ix_choice], alpha=1, s=120, edgecolor="k")

    quiver_kwargs = dict(scale=6.8, headaxislength=9, headlength=15, headwidth=14, linewidths=0.4, edgecolors="k",
                         color="k", alpha=1)
    plt.quiver(dg.embedding[ix_choice, 0], dg.embedding[ix_choice, 1],
               dg.delta_embedding[ix_choice, 0], dg.delta_embedding[ix_choice, 1],
               **quiver_kwargs)

    plt.xlim(xlim[0], xlim[1])
    plt.ylim(ylim[0], ylim[1])
    plt.savefig("./figures/zoom_in.pdf")
    print("zoom_in.pdf has already been generated in figures folder.")

def differential_expression(dg, gene_dataset):
    return
