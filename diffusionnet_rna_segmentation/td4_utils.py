"""Utilities extracted from TD4_mva_geom_ZepengCHU.ipynb.

This module follows the notebook-first layout used in the reference coursework repo: the notebook keeps the narrative, and this file collects reusable definitions.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from tqdm.notebook import tqdm


def project_to_basis(x, evecs, vertex_areas):
    """
    Project an input signal x to the spectral basis.

    Parameters
    -------------------
    x            : (B, n, p) Tensor of input
    evecs        : (B, n, K) Tensor of eigenvectors
    vertex_areas : (B, n,)   Tensor of vertex areas

    Output
    -------------------
    projected_values : (B, K, p) Tensor of coefficients in the basis
    """
    # A f
    x_weighted = vertex_areas.unsqueeze(-1) * x        # (B, n, p)

    # Φ^T A f
    projected_values = evecs.transpose(1, 2) @ x_weighted   # (B, K, p)

    return projected_values


def unproject_from_basis(coeffs, evecs):
    """
    Transform input coefficients in basis into a signal on the complete shape.

    Parameters
    -------------------
    coeffs : (B, K, p) Tensor of coefficients in the spectral basis
    evecs  : (B, n, K) Tensor of eigenvectors

    Output
    -------------------
    decoded_values : (B, n, p) values on each vertex
    """
    # f = Φ α
    decoded_values = evecs @ coeffs   # (B, n, p)

    return decoded_values


class SpectralDiffusion(nn.Module):

    def __init__(self, n_channels):
        """
        Initializes the module with time parameters to 0.

        Parameters
        ------------------
        n_channels : int - number of input feature functions
        """
        super().__init__()

        self.n_channels = n_channels

        #todo
        self.diffusion_times = nn.Parameter(torch.empty(n_channels))
        init.constant_(self.diffusion_times, 0.0)


    def forward(self, x, evals, evecs, vertex_areas):
        """
        Given input features x and information on the current meshes
        return diffused versions of the features.

        Parameters
        ------------------------
        x     : (B, n, p) batch of input features. p = self.n_channels
        evals : (B, K,) batch of eigenvalues
        evecs : (B, n, K) batch of eigenvectors
        vertex_areas : (B, n,) batch of vertex areas

        Output
        ------------------------
        x_diffused : (B, n, p) diffused version of each input feature
        """
        # Remove negative
        with torch.no_grad():
            self.diffusion_times.data = torch.clamp(self.diffusion_times, min=1e-8)

        # β = Φ^T A x  → (B, K, p)
        beta = project_to_basis(x, evecs, vertex_areas)   # (B, K, p)

        # in specral space α_t = exp(-t_j λ_k) * β
        # evals: (B, K) → (B, K, 1)
        lambdas = evals.unsqueeze(-1)                     # (B, K, 1)
        # diffusion_times: (p,) → (1, 1, p)
        t = self.diffusion_times.view(1, 1, -1)          # (1, 1, p)

        decay = torch.exp(-t * lambdas)                  # (B, K, p)
        alpha_t = decay * beta                           # (B, K, p)

        # back to sptial x_t = Φ α_t
        x_diffused = unproject_from_basis(alpha_t, evecs)   # (B, n, p)

        return x_diffused


class SpatialGradient(nn.Module):
    """
    Module which computes g_v from vertex embeddings.
    """
    def __init__(self, n_channels):
        """
        Initializes the module.

        Parameters
        ------------------
        n_channels : int - number of input feature functions
        """

        super().__init__()

        self.n_channels = n_channels

        # Real and Imaginary part of B
        self.B_re = nn.Linear(self.n_channels, self.n_channels, bias=False)
        self.B_im = nn.Linear(self.n_channels, self.n_channels, bias=False)

    def forward(self, vects):
        """
        Parameters
        ----------------------
        Vects : (N, P, 2) per-vertex vector field (w_v)

        Output
        ---------------------
        features : (N, P) per-vertex scalar field
        """
        vects_re = vects[...,0]  # (N,P) real part of w_v
        vects_im = vects[...,1]  # (N,P) imaginary part of w_v

        ## TODO Perform forward pass
        # Re(Bw) = B_re * x - B_im * y
        # Im(Bw) = B_re * y + B_im * x
        Bw_re = self.B_re(vects_re) - self.B_im(vects_im)  # (N, P)
        Bw_im = self.B_re(vects_im) + self.B_im(vects_re)  # (N, P)

        # Complex inner product <w, Bw>_C = x * Re(Bw) + y * Im(Bw)
        inner = vects_re * Bw_re + vects_im * Bw_im         # (N, P)

        # Apply tanh element-wise
        features = torch.tanh(inner)                        # (N, P)

        return features


class MiniMLP(nn.Sequential):
    '''
    A simple MLP with activation and potential dropout
    '''
    def __init__(self, layer_sizes, dropout=False, activation=nn.ReLU):
        """
        Activation and dropout is applied after all layer BUT the last one

        Parameters
        ---------------------------
        layer_size : list of ints - list of sizes of the MLP
        dropout    : book - whether to add droupout or not
        activation : nn.module : activation function
        """
        super().__init__()

        layer_list = []

        ## TODO FILL THE LAYER LIST
        L = len(layer_sizes) - 1
        for i in range(L):
            in_dim = layer_sizes[i]
            out_dim = layer_sizes[i+1]

            layer_list.append(nn.Linear(in_dim, out_dim))

            # for not last layer, add activation + dropout
            if i < L - 1:
                layer_list.append(activation())
                if dropout:
                    layer_list.append(nn.Dropout())

        self.layer = nn.Sequential(*layer_list)

    def forward(self, x):
        """
        Parameters
        --------------------
        x : (n, p) - input features, batch size is the number of vertices !

        Output
        -------------------
        y : (n,p') - output features
        """
        # NOTHING TO DO HERE
        return self.layer(x)


class DiffusionNetBlock(nn.Module):
    """
    Complete Diffusion block
    """

    def __init__(self, n_channels, mlp_hidden_dims, dropout=True):
        """
        Initializes the module.

        Parameters
        ------------------
        n_channels      : int - number of feature functions (serves as both input and output)
        mlp_hidden_dims : list of int - sizes of HIDDEN layers of the miniMLP.
                          You should add the input and output dimension to it.
        """
        super(DiffusionNetBlock, self).__init__()

        # Specified dimensions
        self.n_channels = n_channels

        self.dropout = dropout

        # Diffusion block
        # TODO DEFINE THE 3 SUBPARTS
        # 1 Learned spectral diffusion
        self.diffusion = SpectralDiffusion(n_channels)

        # 2 Gradient feature module
        self.grad_module = SpatialGradient(n_channels)

        # 3 MLP on concatenated features: [x_in, x_diffuse, grad_features]
        mlp_layer_sizes = [3 * n_channels] + mlp_hidden_dims + [n_channels]
        self.mlp = MiniMLP(mlp_layer_sizes, dropout=dropout, activation=nn.ReLU)



    def forward(self, x_in, vertex_areas, evals, evecs, gradX, gradY):
        """
        Parameters
        -------------------
        x_in         : (B,n,p) - Tensor of input signal.
        vertex_areas : (B,n) - Tensor of vertex areas
        evals        : (B, K,) batch of eigenvalues
        evecs        : (B, n, K) batch of eigenvectors
        gradX        : Half of gradient matrix, sparse real tensor with dimension [B,N,N]
        gradY        : Half of gradient matrix, sparse real tensor with dimension [B,N,N]

        Output
        -------------------
        x_out : (B,n,p) - Tensor of output signal.
        """

        # Manage dimensions
        # B = x_in.shape[0] # batch dimension
        B, N, P = x_in.shape

        # Diffusion block
        x_diffuse = self.diffusion(x_in, evals, evecs, vertex_areas)# DIFFUSED X_in  # (B, N, p)


        # Compute the batch of gradients
        x_grads = [] # Manually loop over the batch
        for b in range(B):
            # gradient after diffusion
            x_gradX = torch.mm(gradX[b,...], x_diffuse[b,...])
            x_gradY = torch.mm(gradY[b,...], x_diffuse[b,...])

            x_grads.append(torch.stack((x_gradX, x_gradY), dim=-1))

        x_grad = torch.stack(x_grads, dim=0)  # (B, N, P, 2)

        # TODO EVALUATE GRADIENT FEATURES
        grad_in = x_grad.view(B * N, P, 2)
        grad_feats = self.grad_module(grad_in)      # (B*N, P)
        grad_feats = grad_feats.view(B, N, P)       # (B, N, P)


        # TODO APPLY THE MLP TO THE CONCATENATED FEATURES
        mlp_input = torch.cat([x_in, x_diffuse, grad_feats], dim=-1)  # (B, N, 3P)
        mlp_input_flat = mlp_input.view(B * N, 3 * P)                 # (B*N, 3P)

        mlp_output_flat = self.mlp(mlp_input_flat)                    # (B*N, P)
        mlp_output = mlp_output_flat.view(B, N, P)                    # (B, N, P)


        # TODO APPLY THE RESIDUAL CONNECTION
        x_out = x_in + mlp_output

        return x_out


class DiffusionNet(nn.Module):

    def __init__(self, p_in, p_out, n_channels=128, N_block=4, last_activation=None, mlp_hidden_dims=None, dropout=True):
        """
        Construct a DiffusionNet.
        Parameters
        --------------------
        p_in            : int - input dimension of the network
        p_out           : int - output dimension  of the network
        n_channels      : int - dimension of internal DiffusionNet blocks (default: 128)
        N_block         : int - number of DiffusionNet blocks (default: 4)
        last_activation : int - a function to apply to the final outputs of the network, such as torch.nn.functional.log_softmax
        mlp_hidden_dims : list of int - a list of hidden layer sizes for MLPs (default: [C_width, C_width])
        dropout         : bool - if True, internal MLPs use dropout (default: True)
        """

        super(DiffusionNet, self).__init__()

        ## Store parameters

        # Basic parameters
        self.p_in = p_in
        self.p_out = p_out
        self.n_channels = n_channels
        self.N_block = N_block

        # Outputs
        self.last_activation = last_activation

        # MLP options
        if mlp_hidden_dims == None:
            mlp_hidden_dims = [n_channels, n_channels]
        self.mlp_hidden_dims = mlp_hidden_dims
        self.dropout = dropout


        ## TODO SETUP THE NETWORK (LINEAR LAYERS + BLOCKS)
        in_mlp_sizes = [p_in] + self.mlp_hidden_dims + [n_channels]
        self.input_mlp = MiniMLP(in_mlp_sizes,
                                 dropout=self.dropout,
                                 activation=nn.ReLU)



        self.blocks = [
            DiffusionNetBlock(n_channels=self.n_channels,
                              mlp_hidden_dims=self.mlp_hidden_dims,
                              dropout=self.dropout)
            for _ in range(self.N_block)
        ]
        self.net = nn.ModuleList(self.blocks) # you can then access each block by self.net[i]

        self.output_linear = nn.Linear(self.n_channels, self.p_out)


    def forward(self, x_in, vertex_areas, evals=None, evecs=None, gradX=None, gradY=None):
        """
        Progapate a signal through the network.
        Can handle input without batch dimension (will add a dummy dimension to set batch size to 1)

        Parameters
        --------------------
        x_in         : (n,p) or (B,n,p) - Tensor of input signal.
        vertex_areas : (n,) or (B,n) - Tensor of vertex areas
        evals        : (B, K,) or (K,) batch of eigenvalues
        evecs        : (B, n, K) or (n, K) batch of eigenvectors
        gradX        : Half of gradient matrix, sparse real tensor with dimension [N,N] or [B,N,N]
        gradY        : Half of gradient matrix, sparse real tensor with dimension [N,N] or [B,N,N]

        Output
        -----------------------
        x_out (tensor):    Output with dimension [N,C_out] or [B,N,C_out]
        """


        ## Check dimensions, and append batch dimension if not given
        if x_in.shape[-1] != self.p_in:
            raise ValueError(f"DiffusionNet was constructed with p_in={self.p_in}, "
                             f"but x_in has last dim={x_in.shape[-1]}")
        N = x_in.shape[-2]

        if len(x_in.shape) == 2:
            appended_batch_dim = True

            # add a batch dim to all inputs
            x_in = x_in.unsqueeze(0) # (B, N, P)
            vertex_areas = vertex_areas.unsqueeze(0) # (B, N)
            if evals != None: evals = evals.unsqueeze(0) # (B,K)
            if evecs != None: evecs = evecs.unsqueeze(0) # (B,N,K)
            if gradX != None: gradX = gradX.unsqueeze(0) # (B,N,N)
            if gradY != None: gradY = gradY.unsqueeze(0) # (B,N,N)

        elif len(x_in.shape) == 3:
            appended_batch_dim = False

        else: raise ValueError("x_in should be tensor with shape (n,p) or (B,n,p)")

        ##  TODO PROCESS THE INPUTS
        B = x_in.shape[0]

        # Input MLP
        x = x_in.view(B * N, self.p_in)             # (B*N, p_in)
        x = self.input_mlp(x)                       # (B*N, n_channels)
        x = x.view(B, N, self.n_channels)           # (B, N, n_channels)

        # Diffusion blocks
        for block in self.net:
            x = block(x, vertex_areas, evals, evecs, gradX, gradY)  # (B, N, n_channels)

        # Output linea
        x_out = self.output_linear(x)               # (B, N, p_out)

        # Optional last activation
        if self.last_activation is not None:
            x_out = self.last_activation(x_out)

        # Remove batch dim if we added it
        if appended_batch_dim:
            x_out = x_out.squeeze(0) # (N, p_out)

        return x_out


class Trainer(object):

    def __init__(self, diffusionnet_cls, model_cfg, train_loader, valid_loader, device='cuda',
                 lr=1e-3, weight_decay=1e-4, num_epochs=200,
                 lr_decay_every = 50, lr_decay_rate = 0.5,
                 log_interval=10, save_dir=None):

        """
        diffusionnet_cls: (nn.Module) class of the DiffusionNet model
        model_cfg: (dict) keyword arguments for model
        train_loader: (torch.utils.DataLoader) DataLoader for training set
        valid_loader: (torch.utils.DataLoader) DataLoader for validation set
        device: (str) 'cuda' or 'cpu'
        lr: (float) learning rate
        weight_decay: (float) weight decay for optimiser
        num_epochs: (int) number of epochs
        lr_decay_every: (int) decay learning rate every this many epochs
        lr_decay_rate: (float) decay learning rate by this factor
        log_interval: (int) print training stats every this many iterations
        save_dir: (str) directory to save model checkpoints
        """

        # TOD build the network from the model_cfg
        self.model = diffusionnet_cls(
            p_in      = model_cfg['p_in'],
            p_out     = model_cfg['p_out'],
            n_channels= model_cfg.get('n_channels', 128),
            N_block   = model_cfg.get('N_block', 4),
            last_activation=None,
        )

        self.loss = nn.CrossEntropyLoss()## USE A MEANINGFUL LOSS

        ## THIS PART JUST STORES SOME OTHER PARAMETERS
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)


        self.lr_decay_every = lr_decay_every
        self.lr_decay_rate = lr_decay_rate
        self.log_interval = log_interval
        self.save_dir = save_dir

        self.train_losses = []
        self.test_losses = []
        self.train_accs = []
        self.test_accs = []

        self.inp_feat = model_cfg.get('inp_feat', 'xyz')
        self.num_eig = model_cfg.get('num_eig', 128)
        if not self.inp_feat in ['xyz', 'hks', 'wks']:
            raise ValueError('inp_feat must be one of xyz, hks, wks')

        self.model.to(self.device)


    def forward_step(self, verts, faces, frames, vertex_area, L, evals, evecs, gradX, gradY):
        """
        Perform a forward step of the model.

        Args:
            verts (torch.Tensor): (N, 3) tensor of vertex positions
            faces (torch.Tensor): (F, 3) tensor of face indices
            frames (torch.Tensor): (N, 3, 3) tensor of tangent frames.
            vertex_area (torch.Tensor): (N, N) sparse Tensor of vertex areas.
            L (torch.Tensor): (N, N) sparse Tensor of cotangent Laplacian.
            evals (torch.Tensor): (num_eig,) tensor of eigenvalues.
            evecs (torch.Tensor): (N, num_eig) tensor of eigenvectors.
            gradX (torch.Tensor): (N, N) tensor of gradient in X direction.
            gradY (torch.Tensor): (N, N) tensor of gradient in Y direction.

        Returns:
            pred (torch.Tensor): (N, p_out) tensor of predicted labels.
        """

        if self.inp_feat == 'xyz':
            features = verts
        elif self.inp_feat == 'hks':
            features = self.compute_HKS(verts, faces, self.num_eig, n_feat=32)
        elif self.inp_feat == 'wks':
            features = self.compute_WKS(verts, faces, self.num_eig, num_E=32)

        preds = self.model(features, vertex_area, evals=evals, evecs=evecs, gradX=gradX, gradY=gradY)

        # MAYBE ADD ACTIVATION
        return preds



    def train_epoch(self):
        """
        Train the network for one epoch
        """
        train_loss = 0
        train_acc = 0
        for i, batch in enumerate(tqdm(self.train_loader, "Train epoch")):

            verts = batch["vertices"].to(self.device)
            faces = batch["faces"].to(self.device)
            frames = batch["frames"].to(self.device)
            vertex_area = batch["vertex_area"].to(self.device)
            L = batch["L"].to(self.device)
            evals = batch["evals"].to(self.device)
            evecs = batch["evecs"].to(self.device)
            gradX = batch["gradX"].to(self.device)
            gradY = batch["gradY"].to(self.device)
            labels = batch["labels"].to(self.device)

            self.optimizer.zero_grad()

            preds = self.forward_step(verts, faces, frames, vertex_area, L, evals, evecs, gradX, gradY)
            # MAYBE DO SOMETHING TO THE PREDS

            # COMPUTE THE LOSS
            loss = self.loss(preds, labels.long())
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()

            # COMPUTE TRAINING ACCURACY
            pred_labels = preds.argmax(dim=-1)             # (N,)
            n_correct = pred_labels.eq(labels).sum().item() # number of correct predictions
            train_acc += n_correct/labels.shape[0]

        return train_loss/len(self.train_loader), train_acc/len(self.train_loader)

    def valid_epoch(self):
        """
        Run a validation epoch
        """
        val_loss = 0
        val_acc = 0
        print("Start val epoch")
        for i, batch in enumerate(self.valid_loader):

            # READ BATCH
            verts = batch["vertices"].to(self.device)
            faces = batch["faces"].to(self.device)
            frames = batch["frames"].to(self.device)
            vertex_area = batch["vertex_area"].to(self.device)
            L = batch["L"].to(self.device)
            evals = batch["evals"].to(self.device)
            evecs = batch["evecs"].to(self.device)
            gradX = batch["gradX"].to(self.device)
            gradY = batch["gradY"].to(self.device)
            labels = batch["labels"].to(self.device)

            # TODO PERFORM FORWARD STEP
            preds = self.forward_step(verts, faces, frames, vertex_area, L, evals, evecs, gradX, gradY)
            # MAYBE DO SOMETHING TO THE PREDS

            # Compute Loss - THIS DEPENDS ON YOUR CHOICE OF LOSS
            loss = self.loss(preds, labels.long())
            val_loss += loss.item()

            # Compute ACCURACCY
            pred_labels = preds.argmax(dim=-1)

            n_correct = pred_labels.eq(labels).sum().item() # number of correct predictions
            val_acc += n_correct/labels.shape[0]
        print("End val epoch")
        return val_loss/len(self.valid_loader), val_acc/len(self.valid_loader)

    def run(self):
        os.makedirs('./models', exist_ok=True)
        for epoch in range(self.num_epochs):
            self.model.train()

            if epoch % self.lr_decay_every == 0:
                self.adjust_lr()

            train_ep_loss, train_ep_acc = self.train_epoch()
            self.train_losses.append(train_ep_loss)
            self.train_accs.append(train_ep_acc)

            if epoch % self.log_interval == 0:
                val_loss, val_acc = self.valid_epoch()
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'model_latest.pth'))
                print(f'Epoch: {epoch:03d}/{self.num_epochs}, '
                      f'Train Loss: {train_ep_loss:.4f}, '
                      f'Train Acc: {1e2*train_ep_acc:.2f}%, '
                      f'Val Loss: {val_loss:.4f}, '
                      f'Val Acc: {1e2*val_acc:.2f}%')
        torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'model_final.pth'))


    # def visualize(self):
    #     """
    #     We only test the first two shapes of validation set.
    #     """
    #     self.model.eval()
    #     test_seg_meshes = []

    #     for i, batch in enumerate(self.valid_loader):
    #         verts = batch["vertices"].to(self.device)
    #         faces = batch["faces"].to(self.device)
    #         frames = batch["frames"].to(self.device)
    #         vertex_area = batch["vertex_area"].to(self.device)
    #         L = batch["L"].to(self.device)
    #         evals = batch["evals"].to(self.device)
    #         evecs = batch["evecs"].to(self.device)
    #         gradX = batch["gradX"].to(self.device)
    #         gradY = batch["gradY"].to(self.device)
    #         labels = batch["labels"].to(self.device)


    #         preds = self.forward_step(verts, faces, frames, vertex_area, L, evals, evecs, gradX, gradY)
    #         pred_labels = torch.max(preds, dim=1).indices

    #         test_seg_meshes.append([TriMesh(verts.cpu().numpy(), faces.cpu().numpy()),
    #                               pred_labels.cpu().numpy()])
    #         if i==1:
    #             break


    #     cmap1 = plt.get_cmap("jet")(test_seg_meshes[0][-1] / (146))[:,:3]
    #     cmap2 = plt.get_cmap("jet")(test_seg_meshes[1][-1] / (146))[:,:3]

    #     plu.double_plot(test_seg_meshes[0][0], test_seg_meshes[1][0], cmap1, cmap2)
    #     #return plot_multi_meshes(test_seg_meshes, cmap='vert_colors')

    def visualize(self, num_shapes: int = 2):
      """
      Visualize segmentation on the first `num_shapes` shapes
      from the validation set.
      """
      self.model.eval()
      test_seg_meshes = []

      with torch.no_grad():
          for i, batch in enumerate(self.valid_loader):
              verts = batch["vertices"].to(self.device)
              faces = batch["faces"].to(self.device)
              frames = batch["frames"].to(self.device)
              vertex_area = batch["vertex_area"].to(self.device)
              L = batch["L"].to(self.device)
              evals = batch["evals"].to(self.device)
              evecs = batch["evecs"].to(self.device)
              gradX = batch["gradX"].to(self.device)
              gradY = batch["gradY"].to(self.device)
              labels = batch["labels"].to(self.device)

              preds = self.forward_step(
                  verts, faces, frames,
                  vertex_area, L, evals, evecs, gradX, gradY
              )  # (N, C) or (B, N, C)

              if preds.dim() == 3:
                  preds_b = preds[0]      # (N, C)
                  verts_b = verts[0]      # (N, 3)
                  faces_b = faces[0]      # (F, 3)
              else:
                  preds_b = preds         # (N, C)
                  verts_b = verts         # (N, 3)
                  faces_b = faces         # (F, 3)

              pred_labels = preds_b.argmax(dim=-1)  # (N,)

              test_seg_meshes.append({
                  "V": verts_b.cpu().numpy(),
                  "F": faces_b.cpu().numpy(),
                  "labels": pred_labels.cpu().numpy(),
              })

              if len(test_seg_meshes) >= num_shapes:
                  break

      if hasattr(self.model, "p_out"):
          n_classes = int(self.model.p_out)
      else:
          n_classes = int(
              max(m["labels"].max() for m in test_seg_meshes) + 1
          )

      from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
      import matplotlib.pyplot as plt
      import numpy as np

      fig = plt.figure(figsize=(10, 5))

      for idx in range(min(2, len(test_seg_meshes))):
          mesh = test_seg_meshes[idx]
          V = mesh["V"]          # (N, 3)
          F = mesh["F"]          # (F, 3)
          lab = mesh["labels"]   # (N,)


          cmap = plt.get_cmap("jet")(lab / max(n_classes - 1, 1))[:, :3]  # (N, 3)

          face_colors = cmap[F].mean(axis=1)  # (F, 3)

          ax = fig.add_subplot(1, 2, idx + 1, projection="3d")
          tri = ax.plot_trisurf(
              V[:, 0], V[:, 1], V[:, 2],
              triangles=F,
              linewidth=0.1,
              antialiased=True,
              shade=False,
          )
          tri.set_facecolors(face_colors)

          ax.set_title(f"Shape {idx}")
          ax.set_axis_off()
          ax.view_init(elev=20, azim=-60)

      plt.tight_layout()
      plt.show()


    def adjust_lr(self):
        lr = self.lr * self.lr_decay_rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def compute_HKS(self, evecs, evals, num_eig, n_feat):
        """
        Compute the HKS features for each vertex in the mesh.
        Args:
            evecs (torch.Tensor): (N, K) tensor of eigenvectors
            evals (torch.Tensor): (K,) tensor of eigenvectors
            num_eig (int): number of eigenvalues to use
            n_feat (int): number of features to compute

        Returns:
            hks (torch.Tensor): (N, n_feat) tensor of HKS features
        """
        abs_ev = torch.sort(torch.abs(evals)).values[:num_eig]

        t_list = np.geomspace(4*np.log(10)/abs_ev[-1], 4*np.log(10)/abs_ev[1], n_feat)
        t_list = torch.from_tensor(t_list.astype(np.float32)).to(device=evecs.device)

        evals_s = abs_ev

        coefs = torch.exp(-t_list[:,None] * evals_s[None,:])  # (num_T,K)

        natural_HKS = np.einsum('tk,nk->nt', coefs, evecs[:,:num_eig].square())

        inv_scaling = coefs.sum(1)  # (num_T)

        return (1/inv_scaling)[None,:] * natural_HKS

    def compute_WKS(self, evecs, evals, num_eig, n_feat):
        """
        Compute the WKS features for each vertex in the mesh.

        Args:
            evecs (torch.Tensor): (N, K) tensor of eigenvectors
            evals (torch.Tensor): (K,) tensor of eigenvectors
            num_eig (int): number of eigenvalues to use
            n_feat (int): number of features to compute

        Returns:
            wks: torch.Tensor: (N, num_E) tensor of WKS features
        """
        abs_ev = torch.sort(torch.abs(evals)).values[:num_eig]

        e_min,e_max = np.log(abs_ev[1]),np.log(abs_ev[-1])
        sigma = 7*(e_max-e_min)/n_feat

        e_min += 2*sigma
        e_max -= 2*sigma

        energy_list = torch.linspace(e_min,e_max,n_feat)

        evals_s = abs_ev

        coefs = torch.exp(-torch.square(energy_list[:,None] - torch.log(torch.abs(evals_s))[None,:])/(2*sigma**2))  # (num_E,K)

        natural_WKS = np.einsum('tk,nk->nt', coefs, evecs[:,:num_eig].square())

        inv_scaling = coefs.sum(1)  # (num_E)
        return (1/inv_scaling)[None,:] * natural_WKS


class AblationDiffusionNetBlock(nn.Module):
    """
    Ablation-capable Diffusion block with optional Diffusion / Gradient modules.
    Distinguished from original DiffusionNetBlock.
    """

    def __init__(self,
                 n_channels,
                 mlp_hidden_dims,
                 dropout=True,
                 use_diffusion_module=True,
                 use_gradient_module=True):
        """
        Parameters
        ------------------
        n_channels          : int - feature dimension for input and output
        mlp_hidden_dims     : list[int] - hidden layers of MLP
        dropout             : bool
        use_diffusion_module: bool - enable/disable spectral diffusion
        use_gradient_module : bool - enable/disable gradient features
        """
        super(AblationDiffusionNetBlock, self).__init__()

        self.n_channels = n_channels
        self.use_diffusion_module = use_diffusion_module
        self.use_gradient_module = use_gradient_module

        # ---- Diffusion module (optional) ----
        if self.use_diffusion_module:
            self.diffusion = SpectralDiffusion(n_channels)

        # ---- Gradient module (optional) ----
        if self.use_gradient_module:
            self.grad_module = SpatialGradient(n_channels)

        # ---- Construct MLP input dimension ----
        mlp_in_dim = n_channels              # x_in
        if self.use_diffusion_module:
            mlp_in_dim += n_channels         # x_diffuse
        if self.use_gradient_module:
            mlp_in_dim += n_channels         # grad features

        mlp_layer_sizes = [mlp_in_dim] + mlp_hidden_dims + [n_channels]
        self.mlp = MiniMLP(mlp_layer_sizes,
                           dropout=dropout,
                           activation=nn.ReLU)

    def forward(self, x_in, vertex_areas, evals, evecs, gradX, gradY):
        """
        x_in : (B,N,P)
        """
        B, N, P = x_in.shape

        outputs = [x_in]

        # -------- (1) Diffusion (optional) --------
        if self.use_diffusion_module:
            x_diffuse = self.diffusion(x_in, evals, evecs, vertex_areas)
            outputs.append(x_diffuse)
        else:
            x_diffuse = None

        # -------- (2) Gradient Features (optional) --------
        if self.use_gradient_module:
            source = x_diffuse if self.use_diffusion_module else x_in

            per_batch_grad = []
            for b in range(B):
                gx = torch.mm(gradX[b], source[b])
                gy = torch.mm(gradY[b], source[b])
                per_batch_grad.append(torch.stack((gx, gy), dim=-1))  # (N,P,2)

            batch_grad = torch.stack(per_batch_grad, dim=0)  # (B,N,P,2)
            grad_input = batch_grad.view(B * N, P, 2)
            grad_feats = self.grad_module(grad_input).view(B, N, P)
            outputs.append(grad_feats)

        # -------- (3) MLP on concatenated features --------
        mlp_input = torch.cat(outputs, dim=-1)         # (B,N, M)
        mlp_out = self.mlp(mlp_input.view(B * N, -1)).view(B, N, P)

        # -------- (4) Residual --------
        return x_in + mlp_out


class AblationDiffusionNet(nn.Module):
    """
    Ablation version of DiffusionNet that supports disabling Diffusion or Gradient modules.
    Fully distinguished naming from original DiffusionNet.
    """

    def __init__(self,
                 p_in,
                 p_out,
                 n_channels=128,
                 N_block=4,
                 last_activation=None,
                 mlp_hidden_dims=None,
                 dropout=True,
                 use_diffusion_module=True,
                 use_gradient_module=True):
        super(AblationDiffusionNet, self).__init__()

        self.p_in = p_in
        self.p_out = p_out
        self.n_channels = n_channels
        self.N_block = N_block
        self.last_activation = last_activation
        self.use_diffusion_module = use_diffusion_module
        self.use_gradient_module = use_gradient_module

        if mlp_hidden_dims is None:
            mlp_hidden_dims = [n_channels, n_channels]
        self.mlp_hidden_dims = mlp_hidden_dims
        self.dropout = dropout

        # ---- Input MLP ----
        input_mlp_sizes = [p_in] + mlp_hidden_dims + [n_channels]
        self.input_mlp = MiniMLP(input_mlp_sizes,
                                 dropout=dropout,
                                 activation=nn.ReLU)

        # ---- Blocks ----
        self.blocks = nn.ModuleList([
            AblationDiffusionNetBlock(
                n_channels=n_channels,
                mlp_hidden_dims=mlp_hidden_dims,
                dropout=dropout,
                use_diffusion_module=use_diffusion_module,
                use_gradient_module=use_gradient_module
            ) for _ in range(N_block)
        ])

        # ---- Output layer ----
        self.output_linear = nn.Linear(n_channels, p_out)

    def forward(self, x_in, vertex_areas, evals=None, evecs=None, gradX=None, gradY=None):

        # ------ Manage batch ------
        if len(x_in.shape) == 2:
            x_in = x_in.unsqueeze(0)            # (1,N,p)
            vertex_areas = vertex_areas.unsqueeze(0)
            if evals is not None: evals = evals.unsqueeze(0)
            if evecs is not None: evecs = evecs.unsqueeze(0)
            if gradX is not None: gradX = gradX.unsqueeze(0)
            if gradY is not None: gradY = gradY.unsqueeze(0)
            appended_batch = True
        else:
            appended_batch = False

        B, N, _ = x_in.shape

        # ------ Input MLP ------
        x = self.input_mlp(x_in.view(B * N, -1)).view(B, N, self.n_channels)

        # ------ Blocks ------
        for block in self.blocks:
            x = block(x, vertex_areas, evals, evecs, gradX, gradY)

        # ------ Output layer ------
        x_out = self.output_linear(x)  # (B,N,p_out)

        # ------ Optional activation ------
        if self.last_activation is not None:
            x_out = self.last_activation(x_out)

        # ------ Remove batch if added ------
        if appended_batch:
            x_out = x_out.squeeze(0)

        return x_out


__all__ = ['project_to_basis', 'unproject_from_basis', 'SpectralDiffusion', 'SpatialGradient', 'MiniMLP', 'DiffusionNetBlock', 'DiffusionNet', 'Trainer', 'AblationDiffusionNetBlock', 'AblationDiffusionNet']
