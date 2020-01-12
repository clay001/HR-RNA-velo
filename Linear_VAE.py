import argparse
import src as VI
import warnings
import torch
import numpy as np
import anndata
from VeloLand.dataset.anndataset import AnnDatasetFromAnnData
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='VeloLand')
parser.add_argument("-f", type = str, default="simulation.loom",help = "datafile to use (default:simulation.loom)")
parser.add_argument("-e", type = int, default= 50, help= "running epoch (default:50)")
parser.add_argument("-p", type = str, default= "y", help= "if plot figures (default:y)")
parser.add_argument("-o", type = str, default= "simulation_LD", help= "output filename (default:simulation_LD)")
args = parser.parse_args()

torch.manual_seed(0)
np.random.seed(0)
filename = args.f  #"DentateGyrus.loom"
use_batches = False
use_cuda = True
if args.p == "y":
    p = True
elif args.p == "n":
    p = False
output = args.o
n_epochs = args.e

save_path = "./data/"
n_epochs_all = None

print("### Data Loading ...")
dd = anndata.read_loom(filename= save_path + filename)
X_umap = np.loadtxt("./data/X_umap.txt")
bool_sift = np.load("./data/bool.npy")
adata,oldkey = VI.buildAnn(dd,bool_sift,X_umap)
adata.var_names_make_unique()
adata.obs["clusters"].index = oldkey
adata.obs["age(days)"].index = oldkey
adata.var["gene_ids"].index = dd.var["Accession"].index
cells_dataset = AnnDatasetFromAnnData(adata)
dd.var_names_make_unique()
cells_dataset.gene_names = np.array(dd.var["Accession"].index,dtype = "<U64")

print("### Start training")
adata,trainer,loadings= VI.LDVAE_train(adata ,cells_dataset,save_path = save_path,
                                       pkl_name = output,
                                       n_epochs = n_epochs,
                                       )
VI.plot_detail_loss(trainer,ymin=5500,ymax=7000)

VI.show_latent_space(adata)
print("### Show top five genes")
VI.show_topfive_genes(loadings)


