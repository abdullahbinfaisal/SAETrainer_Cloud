import os
import torch
import warnings
import torch.nn as nn
import matplotlib.pyplot as plt
from lib.train_sae import train_pacs_SAEs
warnings.filterwarnings("ignore", category=UserWarning)
from domainbed.algorithms import DANN, CORAL, Mixup, MMD, IRM, ERM, SagNet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()


base_dir = "./oracle_selection_models"
algo_classes = {"DANN": DANN, "CORAL": CORAL, "Mixup": Mixup, "MMD": MMD, "IRM": IRM, "ERM": ERM, "SagNet": SagNet}

def run():
    for name in os.listdir(base_dir):

        if f"SAE_{name}.pt" in os.listdir(r"./oracle_saes"):
            continue
        else:
            print(name, "does not exist")

        algo_name, backbone_name, testenv = name.split("_")   
        
        model_path = os.path.join(os.path.join(base_dir, name), "model.pkl")
        checkpoint = torch.load(model_path)

        ModelClass = algo_classes[algo_name]
        
        backbone = ModelClass(
            input_shape=checkpoint["model_input_shape"],
            hparams=checkpoint["model_hparams"],
            num_domains=checkpoint["model_num_domains"],
            num_classes=checkpoint["model_num_classes"]
        )
            
        train_pacs_SAEs(backbone, r"./oracle_saes", name)    
        break

if __name__ == "__main__":
    run()