"""Utilities extracted from implicit_mesh_reconstruction.ipynb.

This module follows the notebook-first layout used in the reference coursework repo: the notebook keeps the narrative, and this file collects reusable definitions.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import spatial
from tqdm.auto import tqdm


def get_pc(path):
    ## Load the oriented point set. You can use the function np.loadtxt
    return point_cloud, normals


def compute_sdf(point_cloud, normals, points_query):
    ## Compute SDF on points_query from the shape defined by point_cloud and normals
    return sdf


def compute_sdf_grid(point_cloud, normals, grid_size=40):
    ## Compute SDF on a XYZ grid. First generate the grid (it has to enclose the point cloud)
    ## Then compute the sdf
    #compute the enclosing grid
   

    
    return sdf


class SDFNet(nn.Module):
    def __init__(self, ninputchannels, dropout=0.2, gamma=0, sal_init=False, eik=False):
        super(SDFNet, self).__init__()
        ## Prepare the layers
        ## Don't forget to initialize your weights correctly.

        ## gamma, sal_init, eik are for later
        self.gamma=gamma
        self.eik = eik
        


        #custom weights init

    def forward(self,x):
        ## Logic of the neural network
        ## You can add dropout if you want
        return x


def training_sdf(sdf_gt, gtp):
    geomnet = SDFNet(3)
    geomnet.to(device)
    gtpoints = torch.from_numpy(gtp).float().to(device)
    gtsdf = torch.from_numpy(sdf_gt).float().to(device)

    lpc = []

    optim = torch.optim.Adam(params = geomnet.parameters(), lr=1e-5)

    nepochs=10000
    pbar = tqdm(total=nepochs,
                desc="Training")

    for epoch in range(nepochs):
        loss = evaluate_loss(geomnet, gtpoints, gtsdf, device, lpc, delta = 0.1, batch_size=2500)
        optim.zero_grad()
        loss.backward()
        optim.step()
        if epoch % 100 == 0:
        #     print(f"Epoch {epoch}/{nepochs} - loss : {loss.item()}")
            pbar.set_postfix({'loss': loss.item()})
        pbar.update(1)
    return lpc, geomnet


def training_sal(point_cloud, loss_function, sigma=0.02):
    geomnet = SDFNet(3, gamma=0.5, sal_init=True)
    geomnet.to(device)

    pc_norm = get_normalized_pointcloud(point_cloud)
    points_torch = torch.from_numpy(pc_norm).float().to(device)

    lpc = []

    optim = torch.optim.Adam(params = geomnet.parameters(), lr=1e-4)

    nepochs=5000
    pbar = tqdm(total=nepochs,
                desc="Training")

    for epoch in range(nepochs):
        loss = loss_function(geomnet, points_torch, sigma, device, lpc, batch_size=5000)
        optim.zero_grad()
        loss.backward()
        optim.step()
        if epoch % 100 == 0:
        #     print(f"Epoch {epoch}/{nepochs} - loss : {loss.item()}")
            pbar.set_postfix({'loss': loss.item()})
        pbar.update(1)


    return lpc, geomnet


__all__ = ['get_pc', 'compute_sdf', 'compute_sdf_grid', 'SDFNet', 'training_sdf', 'training_sal']
