# -*- coding:utf-8 -*-
import src as VI
from DG import *
import matplotlib.pyplot as plt

if __name__ == '__main__':
    ### DentateGyrus.loom ###
    save_path = "../data/scVI/"
    file_name = "DentateGyrus.loom"
    pickle_name = "latent.pickle"

    #gene_dataset = VI.data_loading(save_path, file_name)
    #full,latent,trainer = VI.train(gene_dataset,save_path =save_path,pkl_name="SM",latent_name = "SM")

    # experiment 1
    # use ls to plot the same figure
    import pickle

    with open("../data/scVI/latent.pickle", 'rb') as file:
        latent = pickle.load(file)
    dg = v.VelocytoLoom(save_path + file_name)
    bool_sift = preprocess(dg, save_path, pickle_name, latent)
    # # 03 - 05
    knn_imputation_ls(dg, k=500, assumption="constant_velocity")
    #
    # # 06-17
    transition_compute(dg, space="umap")
    # #
    # small_arrow_plot(dg)
    # dg = v.load_velocyto_hdf5("./my_velocyto_analysis")
    # big_arrow_plot(dg)


    # experiment 2
    # get umap corr and calculate on umap
    # umap_embed(save_path, dg, bool_sift, plot=True)
    # calculate new direction
    # transition_compute(dg,space = "umap")
    # plot
    small_arrow_plot(dg)
    big_arrow_plot(dg)
    plot_with_label(dg)

    # experiment 3
    gene_dataset.labels.ravel()