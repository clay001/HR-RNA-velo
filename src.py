import torch
import anndata
import numpy as np
print("cuda available:", torch.cuda.is_available())

from VeloLand.dataset import loom
from VeloLand.models.vae import VAE
from VeloLand.inference.inference import UnsupervisedTrainer
from VeloLand.models.vae import LDVAE
import matplotlib.pyplot as plt
import pandas as pd
import os



def data_loading(save_path, file_name):
    gene_dataset = loom.LoomDataset(file_name, save_path=save_path)
    return gene_dataset


def train(gene_dataset, save_path, pkl_name, latent_name, n_epochs_all=None, show_plot=True,
          n_epochs=100, lr=1e-3,
          use_batches=False, use_cuda=True):
    # torch.cuda.set_device(0)
    vae = VAE(gene_dataset.nb_genes, n_batch=gene_dataset.n_batches * use_batches)
    trainer = UnsupervisedTrainer(
        vae,
        gene_dataset,
        train_size=0.75,
        use_cuda=use_cuda,
        frequency=5,
    )
    if os.path.isfile('%s/%s.pkl' % (save_path, pkl_name)):
        trainer.model.load_state_dict(torch.load('%s/%s.pkl' % (save_path, pkl_name)))
        trainer.model.eval()
        print("model has already been trained before.")
    else:
        trainer.train(n_epochs=n_epochs, lr=lr)
        torch.save(trainer.model.state_dict(), '%s/%s.pkl' % (save_path, pkl_name))
        elbo_train_set = trainer.history["elbo_train_set"]
        elbo_test_set = trainer.history["elbo_test_set"]
        x = np.linspace(0, 500, (len(elbo_train_set)))
        plt.plot(x, elbo_train_set)
        plt.plot(x, elbo_test_set)
        plt.show()
        if not os.path.exists("./figures"):
            os.makedirs("./figures")
        plt.savefig("./figures/Loss_curve.pdf")
        print("Loss_curve.pdf has already been generated in figures folder.")

    full = trainer.create_posterior(trainer.model, gene_dataset, indices=np.arange(len(gene_dataset)))
    latent, batch_indices, labels = full.sequential().get_latent()

    import pickle
    # pickle a variable to a file
    file = open(latent_name + '.pickle', 'wb')
    pickle.dump(latent, file)
    file.close()

    return full, latent, trainer

def LDVAE_train(adata ,cells_dataset,save_path, pkl_name,n_epochs = 50):
    vae = LDVAE(cells_dataset.nb_genes,
                n_batch=cells_dataset.n_batches,
                n_latent=10,
                n_layers=1,
                n_hidden=128,
                reconstruction_loss='nb'
                )

    trainer = UnsupervisedTrainer(vae,
                                  cells_dataset,
                                  frequency=1,
                                  use_cuda=True
                                  )

    if os.path.isfile('%s/%s.pkl' % (save_path, pkl_name)):
        trainer.model.load_state_dict(torch.load('%s/%s.pkl' % (save_path, pkl_name)))
        trainer.model.eval()
        print("model has already been trained before.")
    else:
        trainer.train(n_epochs)
        torch.save(trainer.model.state_dict(), '%s/%s.pkl' % (save_path, pkl_name))
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(trainer.history["elbo_train_set"])
        ax.plot(trainer.history["elbo_train_set"])
        plt.show()
        if not os.path.exists("./figures"):
            os.makedirs("./figures")
        plt.savefig("./figures/LDVAE_Loss_curve.pdf")
        print("LDVAE_Loss_curve.pdf has already been generated in figures folder.")

    full = trainer.create_posterior(trainer.model, cells_dataset, indices=np.arange(len(cells_dataset)))
    Z_hat = full.sequential().get_latent()[0]
    for i, z in enumerate(Z_hat.T):
        adata.obs[f'latent_{i}'] = z

    loadings = list(vae.decoder.factor_regressor.parameters())[0].cpu().detach().numpy()
    loadings = pd.DataFrame.from_records(loadings, index=cells_dataset.gene_names,
                                         columns=[f'latent_{i}' for i in range(10)])
    return adata,trainer,loadings

def show_latent_space(adata):
    plt.figure(figsize=(12, 8))
    for f in range(0, 9, 2):
        plt.subplot(2, 3, int(f / 2) + 1)

        plt.scatter(adata.obs[f'latent_{f}'], adata.obs[f'latent_{f + 1}'], marker='.', s=4, label='Cells')

        plt.xlabel(f'latent_{f}')
        plt.ylabel(f'latent_{f + 1}')

    plt.subplot(2, 3, 6)
    plt.scatter(adata.obs[f'latent_{f}'], adata.obs[f'latent_{f + 1}'], marker='.', label='Cells', s=4)
    plt.scatter(adata.obs[f'latent_{f}'], adata.obs[f'latent_{f + 1}'], c='w', label=None)
    plt.gca().set_frame_on(False)
    plt.gca().axis('off')

    lgd = plt.legend(scatterpoints=3, loc='upper left')
    for handle in lgd.legendHandles:
        handle.set_sizes([200])

    plt.tight_layout()
    if not os.path.exists("./figures"):
        os.makedirs("./figures")
    plt.savefig("./figures/latent_space.pdf")
    print("latent_space.pdf has already been generated in figures folder.")

def show_topfive_genes(loadings):
    print('Top related genes\n#######################')
    for clmn_ in loadings:
        loading_ = loadings[clmn_].sort_values()
        fstr = clmn_ + ':\t' + '\nTop negative related genes\n'
        fstr += '\t\t'.join([f'{i}: {loading_[i]:.2}' for i in loading_.head(5).index])
        #     for i in loading_.tail(5).index:
        #         print(i)
        #         print(loading_[i])
        fstr += '\nTop positive related genes\n'
        fstr += '\t\t'.join([f'{i}: {loading_[i]:.2}' for i in loading_.tail(5).index])
        print(fstr + '\n#######################\n')


def plot_detail_loss(trainer,ymin=7000,ymax=8000):
    elbo_train_set = trainer.history["elbo_train_set"]
    elbo_test_set = trainer.history["elbo_test_set"]
    x = np.linspace(0, 500, (len(elbo_train_set)))
    plt.plot(x, elbo_train_set)
    plt.plot(x, elbo_test_set)
    plt.ylim(ymin, ymax)
    plt.show()
    if not os.path.exists("./figures"):
        os.makedirs("./figures")
    plt.savefig("./figures/Loss_curve_detail.pdf")
    print("Loss_curve_detail.pdf has already been generated in figures folder.")

def differential_expression(dg,full,gene_dataset,couple_celltypes:tuple,genes_of_interest:list,
                            heatmap = True):
    gene_dataset.cell_types = dg.cluster_uid
    gene_dataset.labels = dg.cluster_ix
    # the couple types on which to study DE  tuple(4,5)

    print("\nDifferential Expression A/B for cell types\nA: %s\nB: %s\n" %
          tuple((gene_dataset.cell_types[couple_celltypes[i]] for i in [0, 1])))

    cell_idx1 = gene_dataset.labels.ravel() == couple_celltypes[0]
    cell_idx2 = gene_dataset.labels.ravel() == couple_celltypes[1]

    n_samples = 100
    M_permutation = 100000

    de_res = full.differential_expression_score(
        cell_idx1,
        cell_idx2,
        n_samples=n_samples,
        M_permutation=M_permutation,
    )

    # choose gene: gene_dataset.gene_names
    de_res.filter(items=genes_of_interest, axis=0)

    if(heatmap):
        per_cluster_de, cluster_id = full.one_vs_all_degenes(cell_labels=gene_dataset.labels.ravel(), min_cells=1)

        markers = []
        for x in per_cluster_de:
            markers.append(x[:10])
        markers = pd.concat(markers)

        genes = np.asarray(markers.index)
        expression = [x.filter(items=genes, axis=0)['raw_normalized_mean1'] for x in per_cluster_de]
        expression = pd.concat(expression, axis=1)
        expression = np.log10(1 + expression)
        expression.columns = gene_dataset.cell_types

        # show
        plt.figure(figsize=(20, 20))
        im = plt.imshow(expression, cmap='RdYlGn', interpolation='none', aspect='equal')
        ax = plt.gca()
        ax.set_xticks(np.arange(0, 7, 1))
        ax.set_xticklabels(gene_dataset.cell_types, rotation='vertical')
        ax.set_yticklabels(genes)
        ax.set_yticks(np.arange(0, 70, 1))
        ax.tick_params(labelsize=14)
        plt.colorbar(shrink=0.2)
    return

def data_loading_L(save_path, file_name):
    dd = anndata.read_loom(filename= save_path + file_name)
    X_umap = np.loadtxt("./data/X_umap.txt")
    bool_sift = np.load("./data/bool.npy")
    adata, oldkey = buildAnn(dd, bool_sift, X_umap)
    adata.var_names_make_unique()
    adata.obs["clusters"].index = oldkey
    adata.obs["age(days)"].index = oldkey
    adata.var["gene_ids"].index = dd.var["Accession"].index
    return adata

def buildAnn(tmp, bool_sift, X_umap):
    n = tmp.obs["ClusterName"].values[bool_sift].shape[0]
    oldkey = list(tmp.obs["ClusterName"].index[bool_sift])
    oldvalue = tmp.obs["ClusterName"].values[bool_sift]

    new = anndata.AnnData(tmp.X[bool_sift])
    for i in range(n):
        oldkey[i] = oldkey[i][8:-1]
    new.obs["clusters"] = oldvalue
    # print(oldkey)

    cur = tmp.obs["Age"].values[bool_sift]
    for i in range(n):
        cur[i] = cur[i][1:]

    new.obs["age(days)"] = cur
    new.var["gene_ids"] = tmp.var["Accession"].values

    # new.obs["clusters_enlarged"] = new.obs["clusters"]
    # new.obs["clusters_enlarged"].index = oldkey

    new.uns["clusters_colors"] = np.array(['#3ba458', '#404040', '#7a7a7a', '#fda762', '#6950a3', '#2575b7',
                                        '#08306b', '#e1bfb0', '#e5d8bd', '#79b5d9', '#f14432', '#fc8a6a',
                                        '#98d594', '#d0e1f2'], dtype=object)

    new.obsm["X_tsne"] = np.column_stack([tmp.obs["TSNE1"].values[bool_sift], tmp.obs["TSNE2"].values[bool_sift]])
    new.obsm["X_umap"] = X_umap

    new.layers["ambiguous"] = tmp.layers["ambiguous"][bool_sift]
    new.layers["spliced"] = tmp.layers["spliced"][bool_sift]
    new.layers["unspliced"] = tmp.layers["unspliced"][bool_sift]

    new.obs["clusters"].index = oldkey
    new.obs["age(days)"].index = oldkey
    new.var["gene_ids"].index = tmp.var["Accession"].index

    return new, oldkey




# def get_cells_dataset():
#     cells_dataset = AnnDatasetFromAnnData(adata)





