import argparse
import my_code as VI
import warnings
import datetime
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='VeloLand')
parser.add_argument("-f", type = str, default="simulation.loom",help = "datafile to use (default:simulation.loom)")
parser.add_argument("-e", type = int, default= 100, help= "running epoch (default:100)")
parser.add_argument("-p", type = str, default= "y", help= "if plot figures (default:y)")
parser.add_argument("-lr", type = float, default= 1e-3, help= "learning rate (default:1e-3)")
parser.add_argument("-o", type = str, default= "simulation", help= "output filename (default:simulation)")
args = parser.parse_args()

filename = args.f  #"DentateGyrus.loom"
use_batches = False
use_cuda = True
if args.p == "y":
    p = True
elif args.p == "n":
    p = False
lr = args.lr
output = args.o
n_epochs = args.e

save_path = "./data"
n_epochs_all = None

print("### Data Loading ...")
gene_dataset = VI.data_loading(save_path, file_name = filename)

print("### Start training")
full,latent,trainer = VI.train(gene_dataset,save_path =save_path,pkl_name=output,
                               latent_name =output,show_plot=p,use_batches=use_batches,n_epochs=n_epochs,lr=lr,
                               use_cuda = use_cuda)

VI.plot_detail_loss(trainer,ymin=7000,ymax=8000)


