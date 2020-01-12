# -*- coding:utf-8 -*-
import src as VI
from DG import *
import matplotlib.pyplot as plt

if __name__ == '__main__':
    ### DentateGyrus.loom ###
    save_path = "../data/scVI/"
    file_name = "DentateGyrus.loom"
    pickle_name = "latent.pickle"

    gene_dataset = VI.data_loading(save_path, file_name)
    full,latent,trainer = VI.train(gene_dataset,save_path =save_path,pkl_name="SM",latent_name = "SM")

    # experiment 1
    # use ls to plot the same figure

    dg,bool_sift = preprocess(save_path,file_name,pickle_name,latent)
    knn_imputation(dg)
    transition_compute(dg)
    small_arrow_plot(dg)
    ## dg = v.load_velocyto_hdf5("./my_velocyto_analysis")
    big_arrow_plot(dg)


    # experiment 2
    # get umap corr and calculate on umap
    dg.embedding = umap_embed(save_path, dg, bool_sift, plot=True)
    # calculate new direction
    transition_compute(dg)
    # plot
    small_arrow_plot(dg)
    big_arrow_plot(dg)
    plot_with_label(dg)

    # experiment 3
    # Differential Expression
    couple_celltypes = (4,5)
    genes_of_interest = ["GAD2", "CRHBP"]
    differential_expression(dg,full,gene_dataset,couple_celltypes, genes_of_interest,heatmap = True)
    #bayes_factor: The bayes factor for cell type 1 having a higher expression than cell type 2
    #raw_meani: average UMI counts in cell type i
    #non_zeros_proportioni: proportion of non-zero expression in cell type i
    #raw_normalized_meani: average UMI counts in cell type i normalized by cell size
    #scalei: average scVI imputed gene expression scale in cell type i



