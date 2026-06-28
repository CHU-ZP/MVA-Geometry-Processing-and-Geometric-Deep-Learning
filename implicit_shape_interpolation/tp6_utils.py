"""Utilities extracted from TP_6_ZepengCHU.ipynb.

This module follows the notebook-first layout used in the reference coursework repo: the notebook keeps the narrative, and this file collects reusable definitions.
"""

import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from tqdm.auto import tqdm


class SirenLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        ## Create the layer, and initialize it. You can do it in init_weights
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                # First layer: U(-1/d_in, 1/d_in)
                bound = 1.0 / self.in_features
                self.linear.weight.uniform_(-bound, bound)
                if self.linear.bias is not None:
                    self.linear.bias.uniform_(-bound, bound)
            else:
                # Other layers: U(-sqrt(6/d_in)/ω0, sqrt(6/d_in)/ω0)
                bound = math.sqrt(6.0 / self.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)
                if self.linear.bias is not None:
                    self.linear.bias.uniform_(-bound, bound)


    def forward(self, input):
        ## Logic
        return torch.sin(self.omega_0 * self.linear(input))


class SirenNet(torch.nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, num_layers, skip = [], omega_0 = 30.):
        super().__init__()
        self.num_layers = num_layers
        self.dim_hidden = dim_hidden
        self.skip = [i in skip for i in range(num_layers)]
        self.omega_0 = omega_0

        ## Create layer.
        ## Last layer is a simple linear layer. Don't forget to intialize your weights as before!
        # SIREN hidden layers
        layers = []
        for i in range(num_layers):
            is_first = (i == 0)
            in_features = dim_in if is_first else dim_hidden
            layers.append(
                SirenLayer(
                    in_features=in_features,
                    out_features=dim_hidden,
                    is_first=is_first,
                    omega_0=omega_0
                )
            )
        self.layers = torch.nn.ModuleList(layers)

        # Last linear layer
        self.final_linear = torch.nn.Linear(dim_hidden, dim_out)

        # Initialize last layer weights (non-first SIREN layers)
        with torch.no_grad():
            bound = math.sqrt(6.0 / dim_hidden) / omega_0
            self.final_linear.weight.uniform_(-bound, bound)
            if self.final_linear.bias is not None:
                self.final_linear.bias.uniform_(-bound, bound)


    def forward(self, x):
        ## Network logic
        ## You can ignore skip connections at the beginning
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
        out = self.final_linear(h)
        return out


def loss_data(predicted_sdf):
    return (predicted_sdf**2).mean()


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


def loss_normalign(u,v):
    dot_product = (u * v).sum(dim=-1, keepdim=True)
    return ((1.0 - dot_product) ** 2).mean()


def loss_shape_data(net, pc, normals, batch_size=2000, dim_space=2):
    # Sample points
    num_points = pc.shape[0]
    indices = torch.randint(0, num_points, (batch_size,), device=pc.device)
    sample_pc = pc[indices]
    sample_pc.requires_grad = True
    sample_nc = normals[indices]

    if sample_pc.shape[-1] == dim_space:
        t_zero = torch.zeros(batch_size, 1, device=pc.device)
        input_for_net = torch.cat([sample_pc, t_zero], dim=-1)
    else:
        input_for_net = sample_pc  # Use the provided 3D points (which include correct t)

    predicted_sdf = net(input_for_net)

    # Spatial gradients (take only first 2 dims corresponding to x, y)
    grad_pc = gradient(predicted_sdf, sample_pc)[:, 0:dim_space]

    loss_data_term = loss_data(predicted_sdf)
    loss_normal_term = loss_normalign(grad_pc, sample_nc)

    return 100 * loss_data_term + loss_normal_term


def loss_amb(net, pc_hint, gt_sdf, batch_size=2000):
    num_points = pc_hint.shape[0]
    indices = torch.randint(0, num_points, (batch_size,), device=pc_hint.device)
    sample_pc = pc_hint[indices]
    sample_gt_sdf = gt_sdf[indices]

    sample_gt_sdf = sample_gt_sdf.view(-1, 1)

    # 2D vs 3D inputs
    if sample_pc.shape[-1] == 2:
        t = torch.zeros(batch_size, 1, device=pc_hint.device)
        inp = torch.cat([sample_pc, t], dim=-1)
    else:
        inp = sample_pc

    pred_sdf = net(inp)

    loss = ((pred_sdf - sample_gt_sdf)**2).mean()
    return 10 * loss


def loss_eikonal_pts(grad_pts):
    ## Get the eikonal loss given spatial gradients
    norm = torch.norm(grad_pts, dim=-1, keepdim=True)
    return ((norm - 1.0) ** 2).mean()


def loss_eikonal(net, batch_size, dim_space=2):
    ## Sample random points in space ([-1, 1]^2) and in time ([0, 1])

    # Random spatial coords in [-1, 1]
    pts_space = torch.rand(batch_size, dim_space, device=device) * 2 - 1
    # Random time in [0, 1]
    pts_time = torch.rand(batch_size, 1, device=device)

    pts_random = torch.cat([pts_space, pts_time], dim=-1)
    pts_random.requires_grad = True

    sdf_random = net(pts_random)

    grad_tot_random = gradient(sdf_random, pts_random)
    grad_spatial = grad_tot_random[:,0:2]

    return loss_eikonal_pts(grad_spatial)


def loss_lse_eq(net, vf, batch_size, dim_space=2):
    ## Sample random points in space ([-1, 1]^2) and in time ([0, 1])
    pts_space = torch.rand(batch_size, dim_space, device=device) * 2 - 1
    pts_time = torch.rand(batch_size, 1, device=device)

    pts_random = torch.cat([pts_space, pts_time], dim=-1)
    pts_random.requires_grad = True

    sdf_random = net(pts_random)

    grad_tot_random = gradient(sdf_random, pts_random)
    # spatial gradient
    grad_random = grad_tot_random[:,0:2]

    # temporal gradient (derivative w.r.t t)
    tempgrad_random = grad_tot_random[:,2:]

    ## compute the loss: |dF/dt + <grad_x F, V>|^2
    # Ensure vf is broadcastable. If vf is (2,), we might need to view it
    advection = (grad_random * vf).sum(dim=-1, keepdim=True)

    loss_lse = ((tempgrad_random + advection) ** 2).mean()
    return loss_lse


def evaluate_loss_cst_vf(net, pc, normals, hints_pc, gtsdf, vf,
                           lpc, leik, lh, llse,
                           lambda_pc = 1, lambda_eik = 2, lambda_hint = 1, lambda_lse = 1, batch_size = 2000):



    # compute and store standard losses
    loss_pc = loss_shape_data(net, pc, normals, batch_size)

    loss_hint = loss_amb(net, hints_pc, gtsdf, batch_size)

    loss_eik = loss_eikonal(net, batch_size)

    loss_lse = loss_lse_eq(net, vf, batch_size)

    # append all the losses
    lpc.append(float(loss_pc))
    leik.append(float(loss_eik))
    lh.append(float(loss_hint))
    llse.append(float(loss_lse))

    # sum the losses of reach of this set of points
    loss = lambda_pc*loss_pc + lambda_eik*loss_eik + lambda_hint*loss_hint + lambda_lse*loss_lse

    return loss


def optimize_nise_vf(net, pc0, nc0, hints0, gtsdf0,
                           vf, lpc, leik, lh, llse,
                           lambda_pc = 1, lambda_eik = 2, lambda_hint = 1, lambda_lse = 2, batch_size = 2000, nepochs=100, plot_loss = True):

    optim = torch.optim.Adam(params=net.parameters(), lr=2e-5)

    tinit = time.time()
    pbar = tqdm(total=nepochs,
                desc="Training")
    for batch in range(nepochs):
        optim.zero_grad()

        loss = evaluate_loss_cst_vf(net, pc0, nc0, hints0, gtsdf0, vf, lpc, leik, lh, llse, lambda_pc, lambda_eik, lambda_hint, lambda_lse, batch_size)

        loss.backward()
        optim.step()
        if batch % 100 == 99 or batch == 0:
        #     print(f"Epoch {epoch}/{nepochs} - loss : {loss.item()}")
            pbar.set_postfix({"loss": loss.item()})
        pbar.update(1)

    tend = time.time()

    print("Optimizing NN took", '{:.2f}'.format(tend-tinit),"s.")


def get_vector_field_meshgrid(X_value, Y_value):
    # Create a grid of points
    x = np.linspace(-1, 1, 20)
    y = np.linspace(-1, 1, 20)
    X, Y = np.meshgrid(x, y)

    # Define the constant vector field (e.g., constant vector [1, 1] everywhere)
    U = np.ones_like(X)*X_value/(10*max(X_value, Y_value))  # constant x-component
    V = np.ones_like(Y)*Y_value/(10*max(X_value, Y_value))  # constant y-component
    return X,Y,U,V


def loss_lse_morph(net, pc_hint, gt_sdf, batch_size):
    idx = torch.randint(0, pc_hint.shape[0], (batch_size,))

    coords = pc_hint[idx, :2] # x, y
    f1_val = gt_sdf[idx]      # f1(x)

    f1_val = f1_val.view(-1, 1)

    t_rand = torch.rand(batch_size, 1, device=pc_hint.device)
    random_hints = torch.cat([coords, t_rand], dim=-1)
    random_hints.requires_grad = True

    sdf_random_hints = net(random_hints)
    grad_tot_ = gradient(sdf_random_hints, random_hints)

    spatial_grad = grad_tot_[:, 0:2]
    temp_deriv = grad_tot_[:, 2:]
    norm_spatial = torch.norm(spatial_grad, dim=-1, keepdim=True)

    # (B, 1) - (B, 1)
    target_update = (f1_val - sdf_random_hints) * norm_spatial

    l_lse = ((temp_deriv - target_update) ** 2).mean()
    return l_lse


def evaluate_loss_morphing(net, pc0, normals0, hints_pc0, gtsdf0,
                           pc1, normals1, hints_pc1, gtsdf1,
                           lpc, leik, lh, llse,
                           lambda_pc = 100, lambda_eik = 2e2, lambda_hint = 1e2, lambda_lse = 1e2, batch_size = 2000):


    # compute and store standard losses
    loss_pc = loss_shape_data(net, pc0, normals0, batch_size) + loss_shape_data(net, pc1, normals1, batch_size)
    loss_hint = loss_amb(net, hints_pc0, gtsdf0, batch_size) + loss_amb(net, hints_pc1, gtsdf1, batch_size)

    loss_eik = loss_eikonal(net, batch_size)

    loss_lse = loss_lse_morph(net, hints_pc1, gtsdf1, batch_size)

    # append all the losses
    lpc.append(float(loss_pc))
    leik.append(float(loss_eik))
    lh.append(float(loss_hint))
    llse.append(float(loss_lse))

    # sum the losses of reach of this set of points
    loss = lambda_pc*loss_pc + lambda_eik*loss_eik + lambda_hint*loss_hint + lambda_lse*loss_lse

    return loss


def optimize_nise_morphing(net, pc0, nc0, pc1, nc1, hints0, gtsdf0,
                           hints1, gtsdf1, lpc, leik, lh, llse,
                           lambda_pc = 1, lambda_eik = 2, lambda_hint = 1, lambda_lse = 2, batch_size = 2000, nepochs=100, plot_loss = True):

    optim = torch.optim.Adam(params=net.parameters(), lr=2e-5)
    #optim = torch.optim.LBFGS(params=net.parameters())
    tinit = time.time()
    pbar = tqdm(total=nepochs,
                desc="Training")
    for batch in range(nepochs):
        optim.zero_grad()

        loss = evaluate_loss_morphing(net, pc0, nc0, pc1, nc1, hints0, gtsdf0, hints1, gtsdf1, lpc, leik, lh, llse, lambda_pc, lambda_eik, lambda_hint, lambda_lse, batch_size)

        loss.backward()
        optim.step()

        if batch % 100 == 99 or batch == 0:
            pbar.set_postfix({"loss": loss.item()})
        pbar.update(1)

    tend = time.time()

    print("Optimizing NN took", '{:.2f}'.format(tend-tinit),"s.")


__all__ = ['SirenLayer', 'SirenNet', 'loss_data', 'gradient', 'loss_normalign', 'loss_shape_data', 'loss_amb', 'loss_eikonal_pts', 'loss_eikonal', 'loss_lse_eq', 'evaluate_loss_cst_vf', 'optimize_nise_vf', 'get_vector_field_meshgrid', 'loss_lse_morph', 'evaluate_loss_morphing', 'optimize_nise_morphing']
