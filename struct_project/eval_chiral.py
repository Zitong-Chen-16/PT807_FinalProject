import argparse
import os

import torch
import pickle
import numpy as np
from pathlib import Path

import utils
import build_geom_dataset
from configs.datasets_config import get_dataset_info
from os.path import join
from qm9.sampling import sample
from qm9.utils import compute_mean_mad
from qm9 import dataset
from qm9.models import get_latent_diffusion
from equivariant_diffusion.utils import assert_mean_zero_with_mask, remove_mean_with_mask,\
    assert_correctly_masked

def check_mask_correct(variables, node_mask):
    for variable in variables:
        assert_correctly_masked(variable, node_mask)

def assert_correctly_masked(variable, node_mask):
    assert (variable * (1 - node_mask)).abs().sum().item() < 1e-8

def test(args, flow_dp, nodes_dist, device, dtype, loader):
    flow_dp.eval()
    nll_epoch = 0
    n_samples = 0
    embeddings = []
    with torch.no_grad():
        for i, data in enumerate(loader):
            # Get data
            x = data['positions'].to(device, dtype)
            node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
            edge_mask = data['edge_mask'].to(device, dtype)
            one_hot = data['one_hot'].to(device, dtype)
            charges = (data['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype)

            batch_size = x.size(0)

            x = remove_mean_with_mask(x, node_mask)
            check_mask_correct([x, one_hot], node_mask)
            assert_mean_zero_with_mask(x, node_mask)

            h = {'categorical': one_hot, 'integer': charges}

            context = None
            bs, n_nodes, n_dims = x.size()
            edge_mask = edge_mask.view(bs, n_nodes * n_nodes)
            assert_correctly_masked(x, node_mask)

            z_x_mu, sigma_x0, z_h_mu, sigma_h0 = flow_dp.vae.encode(
                x, h, node_mask, edge_mask, context
            )
            z_mu = torch.cat([z_x_mu, z_h_mu], dim=-1)  # [B, N, latent_dim]
            mol_vec = (z_mu * node_mask).sum(1) / node_mask.sum(1)
            embeddings.append(mol_vec.cpu())
    embeddings = torch.cat(embeddings)

    return embeddings

def load_data(conformation_file):
    path = Path(conformation_file)
    base_path = path.parent.absolute()

    # Load conformations
    all_data = np.load(conformation_file)  # 2d array: num_atoms x 5
    mol_id = all_data[:, 0].astype(int)
    conformers = all_data[:, 1:]

    # Get ids corresponding to new molecules
    split_indices = np.nonzero(mol_id[:-1] - mol_id[1:])[0] + 1
    data_list = np.split(conformers, split_indices)
    data_list = np.array(data_list, dtype="object")
    return data_list

def retrieve_dataloaders(cfg, data_file):

    dataset_info = get_dataset_info(cfg.dataset, cfg.remove_h)

    data = load_data(data_file)
    transform = build_geom_dataset.GeomDrugsTransform(dataset_info,
                                                      cfg.include_charges,
                                                      cfg.device,
                                                      cfg.sequential)
    dataset = build_geom_dataset.GeomDrugsDataset(data, transform=transform)
    dataloader = build_geom_dataset.GeomDrugsDataLoader(
        sequential=cfg.sequential, 
        dataset=dataset,
        batch_size=cfg.batch_size,
        shuffle=False
    )
    
    charge_scale = None
    return dataloader, charge_scale

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="outputs/drugs_latent2",
                        help='Specify model path')
    parser.add_argument('--dataset_path', type=str, default='data/geom/geom_drugs_chiral_30.npy',
                        help='Path of dataset')
    parser.add_argument('--output_path', type=str, default='outputs/drugs_latent2/embeddings.npy')

    eval_args, unparsed_args = parser.parse_known_args()
    assert eval_args.model_path is not None

    with open(join(eval_args.model_path, 'args.pickle'), 'rb') as f:
        args = pickle.load(f)

    # CAREFUL with this -->
    if not hasattr(args, 'normalization_factor'):
        args.normalization_factor = 1
    if not hasattr(args, 'aggregation_method'):
        args.aggregation_method = 'sum'

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device
    dtype = torch.float32
    utils.create_folders(args)
    # Retrieve dataloaders
    dataloader, charge_scale = retrieve_dataloaders(args, eval_args.dataset_path)
    dataset_info = get_dataset_info(args.dataset, args.remove_h)

    # Load model
    generative_model, nodes_dist, prop_dist = get_latent_diffusion(args, device, dataset_info, dataloader)
    generative_model.to(device)

    fn = 'generative_model_ema.npy' if args.ema_decay > 0 else 'generative_model.npy'
    flow_state_dict = torch.load(join(eval_args.model_path, fn), map_location=device)
    generative_model.load_state_dict(flow_state_dict)

    # Extract latent embeddings using the model
    embeddings = test(
                    args,
                    generative_model,
                    nodes_dist,
                    device,
                    dtype,
                    dataloader
                )
    np.save(embeddings, eval_args.output_path, )

if __name__ == "__main__":
    main()
