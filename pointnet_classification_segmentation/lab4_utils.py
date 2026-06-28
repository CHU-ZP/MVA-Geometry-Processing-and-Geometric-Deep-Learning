"""Utilities extracted from pointnet_classification_segmentation.ipynb.

This module follows the notebook-first layout used in the reference coursework repo: the notebook keeps the narrative, and this file collects reusable definitions.
"""

import os
import os.path as osp
from glob import glob
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


class FeatTransform(nn.Module):
    # TODO
    def __init__(self, inp_dim=3, hidden_dims1=[64,128,1024], hidden_dims2=[512,256], use_bn=True):
      """
      inp_dim : int - dimension d of the input
      hidden_dims1 : list - hidden layers (inluding d_out) of the first MLP block (defined with nn.Conv1d...)
      hidden_dims2 : list - hidden layers (without d_out) of the second MLP block
      use_bn       : bool - whether to use batchnorm
      """
      super().__init__()

      ## TODO
      self.inp_dim = inp_dim
      self.use_bn = use_bn

      # Block1: shared point-wise MLP (Using nn.Conv1d to define fully connecter layers)
      c_in = inp_dim
      conv_layers=[]
      for c_out in hidden_dims1:
        conv_layers.append(nn.Conv1d(c_in, c_out, kernel_size=1, bias= not use_bn))
        if use_bn:
          conv_layers.append(nn.BatchNorm1d(c_out))
        conv_layers.append(nn.ReLU())
        c_in = c_out

      self.mlp1 = nn.Sequential(*conv_layers)

      #Block2:
      lin_sizes = [hidden_dims1[-1]] + hidden_dims2
      linear_layers=[]
      for i in range(len(lin_sizes)-1):
        linear_layers.append(nn.Linear(lin_sizes[i],lin_sizes[i+1],bias=True))
        if use_bn:
          linear_layers.append(nn.BatchNorm1d(lin_sizes[i+1]))
        linear_layers.append(nn.ReLU())

        self.mlp2 = nn.Sequential(*linear_layers)

        self.final_fc = nn.Linear(lin_sizes[-1], inp_dim * inp_dim)

        nn.init.zeros_(self.final_fc.weight)
        nn.init.zeros_(self.final_fc.bias)


    def forward(self, x):
        """
        x : (B, d, n) - This is standard shape for convolution inputs

        Output
        ------------
        (B, d , d) : output T defined as I + NET(x) for stability
        """
        B,d,n=x.shape
        assert d == self.inp_dim, f"Expected 3d coordinate dim {self.inp_dim}, got {d}"
        # Block 1, from per-point features to global features, dim-1
        f = self.mlp1(x) # (B,d,n)->(B,1024,n)
        g = torch.max(f,dim=2,keepdim=False)[0]

        #Block 2 from glbal features to T martix
        h = self.mlp2(g)
        t = self.final_fc(h)
        T = t.view(B,self.inp_dim,self.inp_dim)


        return T + torch.eye(self.inp_dim, device=T.device).unsqueeze(0)  #Block 2 from glbal features to T martix
        h = self.mlp2(g)
        t = self.final_fc(h)
        T = t.view(B,self.inp_dim,self.inp_dim)


        return T + torch.eye(self.inp_dim, device=T.device).unsqueeze(0)


class PointNetfeat(nn.Module):

    ## TODO
    def __init__(self, inp_dim=3, mlp2_hidden_dims=[64,128,1024], transf_hidden_dims1=[64,128,1024], transf_hidden_dims2=[512,256], use_bn=True):
      """
      inp_dim   : int - dimension d of the input
      mlp2_hidden_dims : list - hidden layers (inluding d_out) of the second MLP block (defined with nn.Conv1d...)
      transf_hidden_dims1 : list - hidden layers (inluding d_out) of the first mlp in each FeatureTransform block
      transf_hidden_dims2 : list - hidden layers (without d_out) of the second mlp in each FeatureTransform block
      use_bn : bool - whether to use batchnorm
      """
      super().__init__()

      # USEFUL INFO
      self.inp_dim = inp_dim
      self.latent_dim1 = mlp2_hidden_dims[0] # size of the second features
      self.latent_dim = mlp2_hidden_dims[-1] # is also the out_dim
      self.use_bn = use_bn

      # transform1 get a T martix
      self.transf1 = FeatTransform(inp_dim, transf_hidden_dims1, transf_hidden_dims2, use_bn)

      # mlp1: 3->64, which is a convention
      conv_layers=[]
      conv_layers.append(nn.Conv1d(inp_dim, transf_hidden_dims1[0], kernel_size=1, bias= not use_bn))
      conv_layers.append(nn.BatchNorm1d(transf_hidden_dims1[0]))
      conv_layers.append(nn.ReLU())
      self.mlp1 = nn.Sequential(*conv_layers)

      # transform2
      self.transf2 = FeatTransform(mlp2_hidden_dims[0], transf_hidden_dims1, transf_hidden_dims2, use_bn)

      # mlp2: 64->1024:  mlp2_hidden_dims=[64,128,1024]
      conv_layers=[]
      c_in = mlp2_hidden_dims[0]
      for c_out in mlp2_hidden_dims[1:]:
        conv_layers.append(nn.Conv1d(c_in, c_out, kernel_size=1, bias= not use_bn))
        if use_bn:
          conv_layers.append(nn.BatchNorm1d(c_out))
        conv_layers.append(nn.ReLU())
        c_in = c_out
      self.mlp2 = nn.Sequential(*conv_layers)



    def forward(self, x):
      """
      Parameters
      --------------
      x : (B, in_dim, n) input batch in convolution format.

      Output
      -------------
      global_feature : (B, d_out) output global feature (d_out=1024)
      T_feat : (B, latent1, latent1) transformation matrix of the second FeatureTransform module
      x_feat : (B, latent1, n) per-point features after being transformed by the second
                FeatureTransform module
      """
      T_input = self.transf1(x)
      x = torch.bmm(T_input, x)
      x = self.mlp1(x)
      T_feat = self.transf2(x)           # (B, 64, 64)
      x_feat = torch.bmm(T_feat, x)
      x_feat = self.mlp2(x_feat)
      global_feature = torch.max(x_feat, dim=2)[0]

      return global_feature, T_feat, x_feat


class PointNetCls(nn.Module):
    ## TODO
    def __init__(self, n_cls, inp_dim=3, cls_hidden_dims=[512,256], mlp2_hidden_dims=[64,128,1024], transf_hidden_dims1=[64,128,1024], transf_hidden_dims2=[512,256], use_bn=True):
        """
        PointNet classification network

        Parameters
        ------------------
            n_cls (int): Number of classes for classification
            inp_dim (int): Dimension of the input points
            cls_hidden_dims (list): List of hidden dimensions for the MLP classifier
            mlp2_hidden_dims (list): List of hidden dimensions for the second MLP block in the feature extractor
            transf_hidden_dims1 (list): List of hidden dimensions for the first MLP in each FeatureTransform block
            transf_hidden_dims2 (list): List of hidden dimensions for the second MLP in each FeatureTransform block
            use_bn (bool): Whether to use batchnorm
        """

        super(PointNetCls, self).__init__()

        self.use_bn = use_bn

        ## TODO

        self.feat = PointNetfeat(inp_dim, mlp2_hidden_dims, transf_hidden_dims1, transf_hidden_dims2, use_bn)

        # classification
        in_dim = mlp2_hidden_dims[-1]
        layers = []
        prev = in_dim
        for h in cls_hidden_dims:
            layers += [nn.Linear(prev, h)]
            if use_bn:
                layers += [nn.BatchNorm1d(h)]
            layers += [nn.ReLU(inplace=True)]
            layers += [nn.Dropout(p=0.3)]
            prev = h
        layers += [nn.Linear(prev, n_cls)]  # No BN/ReLU
        self.cls_head = nn.Sequential(*layers)


    def forward(self, x):
      """
      Parameters
      --------------
      inp : (B, n, d) input batch in standard format.

      Output
      -------------
      out : (B, n_cls) output score for each class
      T_feat : (B, latent1, latent1) transformation matrix of the second FeatureTransform module
      """
      x = x.transpose(2,1) # Put input in convolution format
      global_feature, T_feat, x_feat = self.feat(x)
      logits = self.cls_head(global_feature)
      return logits, T_feat


def random_scale_transl(X):
    """Apply random independent scaling and translation to the point cloud.

    Args:
        X (np.ndarray): Point cloud data, shape (N, 3).

    Returns:
        X_transformed (np.ndarray): Scaled and translated point cloud data, shape (N, 3).
    """
    # Independent scaling for each coordinate axis ∈ [2/3, 3/2]
    scale = np.random.uniform(2/3, 3/2, size=(1, 3))          # shape (1,3)
    # Independent translation for each coordinate axis ∈ [-0.2, 0.2]
    translation = np.random.uniform(-0.2, 0.2, size=(1, 3))   # shape (1,3)
    # Apply the affine transform
    X_transformed = X * scale + translation
    return X_transformed


def random_rotation(X):
    """Apply random rotation to the point cloud (around the y-axis).

    Args:
        X (np.ndarray): Point cloud data, shape (N, 3).

    Returns:
        X_rot (np.ndarray): Rotated point cloud data, shape (N, 3).
    """
    theta = np.random.uniform(0, 2 * np.pi)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    R_y = np.array([
        [cos_t, 0, sin_t],
        [0,     1, 0    ],
        [-sin_t,0, cos_t]
    ], dtype=np.float32)

    X_rot = X @ R_y.T   # (N,3) × (3,3)
    return X_rot


def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    print(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    f.close()
    return (data, label)


def load_h5_files(data_path, files_list_path):
    """Load h5 into memory
    """
    files_list = [Path(line.rstrip()).name for line in open(osp.join(data_path, files_list_path))]
    data = []
    labels = []
    for i in range(len(files_list)):
        data_, labels_ = load_h5(os.path.join(data_path, files_list[i]))
        data.append(data_)
        labels.append(labels_)
    data = np.concatenate(data, axis=0)
    labels = np.concatenate(labels, axis=0)
    return data, labels


class ModelNet40Generator(Dataset):

    def __init__(self, mode, data_dir, files_list, num_classes=40, num_points=1024, rot_aug=False):
        """Initialize the ModelNet40Generator class.

        Args:
            mode (str): 'train' or 'test'.
            data_dir (str): Path to the data directory.
            files_list (str): Path to the files list.
            num_classes (int, optional): ModelNet class. Defaults to 40.
            num_points (int, optional): Input PC. Defaults to 1024.
            rot_aug (bool, optional): Augment data. Defaults to False.
        """

        assert mode.lower() in ('train', 'val', 'test')
        assert files_list in ('train_files.txt', 'test_files.txt')
        self.data, labels = load_h5_files(data_dir, files_list)

        self.mode = mode
        self.num_points = num_points
        self.num_classes = num_classes
        self.num_samples = self.data.shape[0]
        self.rot_aug = rot_aug
        self.labels = np.reshape(labels, (-1,))


    def __len__(self) -> int:
        return self.num_samples


    def __getitem__(self, idx):
        indexes = np.random.permutation(np.arange(self.num_points))[:self.num_points]
        X = self.data[idx, indexes, ...]
        y = self.labels[idx, ...]
        y_categorical = torch.from_numpy(np.eye(self.num_classes, dtype='uint8')[y]).long()

        if self.mode == 'train' and self.rot_aug:
            # X = self.random_scaling(X)
            X = self.random_rotation_3d(X)

        return torch.from_numpy(X).float(), y_categorical.float()


    def random_scaling(self, X):
        """Apply random scaling to the point cloud.

        Args:
            X (np.ndarray): Point cloud data, (N, 3).

        Returns:
            X (np.ndarray): Scaled point cloud data (N, 3).
        """
        return random_scale_transl(X)


    def random_rotation_3d(self, X):
        """Apply random rotation to the point cloud (axis agnostic).

        Args:
            X (np.ndarray): Point cloud data, (N, 3).

        Returns:
            rotated_data (np.ndarray): Rotated point cloud data (N, 3).
        """
        return random_rotation(X)


def classification_loss(preds, gt_labels, T_feat, w_reg):
        """
        Classification loss for PointNet

        Parameters
        --------------
        preds : (B, n_cls) - output of the classifier
        gt_labels : (B,) - ground truth labels
        T_feat : (B, latent1, latent1) - transformation matrix of the second FeatureTransform module
        w_reg : weight of the regularization term

        Output
        -------------
        loss : classification loss L_{cls} + w_reg * L_{reg}
        """
        ce = F.cross_entropy(preds, gt_labels)

        d = T_feat.size(1)
        I = torch.eye(d, device=T_feat.device).unsqueeze(0).expand_as(T_feat)  # (B,d,d)
        reg = torch.mean(torch.norm(T_feat.bmm(T_feat.transpose(2, 1)) - I, dim=(1, 2)))  # Frobenius norm per sample

        return ce + w_reg * reg


class Trainer(object):

    def __init__(self, train_loader, valid_loader, device='cuda',
                 lr=1e-3, weight_decay=1e-4, num_epochs=200,
                 lr_decay_every = 50, lr_decay_rate = 0.5,
                 log_interval=10, save_dir=None, **model_cfg):

        """
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
        model_cfg: (dict) keyword arguments for model
        """
        self.model = self.get_model(model_cfg).to(device)
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
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []

        self.inp_feat = model_cfg.get('inp_feat', 'xyz')
        self.num_eig = model_cfg.get('num_eig', 128)
        if not self.inp_feat in ['xyz', 'xyzn', 'hks', 'wks']:
            raise ValueError('inp_feat must be one of xyz, xyzn, hks, wks')

        self.model.to(self.device)

    def get_model(self, model_cfg):
        if model_cfg['name'].lower() == 'pointnet':
            return PointNetCls(n_cls=model_cfg['n_cls'],
                              inp_dim=model_cfg['inp_dim'],
                              cls_hidden_dims=model_cfg['cls_hidden_dims'],
                              mlp2_hidden_dims=model_cfg['mlp2_hidden_dims'],
                              transf_hidden_dims1=model_cfg['transf_hidden_dims1'],
                              transf_hidden_dims2=model_cfg['transf_hidden_dims2'])
        else:
            raise ValueError('%s must be one of PointNet, PointNet++'%model_cfg['name'])

    def adjust_lr(self):
        lr = self.lr * self.lr_decay_rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def forward_step(self, inp):
        """
        Perform a forward step of the model.

        Args:
            inp (torch.Tensor): (N, D) tensor of input features

        Returns:
            pred (torch.Tensor): (N, p_out) tensor of predicted labels.
        """
        inp = inp.to(self.device)
        preds, trans_feat = self.model(inp)
        return preds, trans_feat


    def get_loss(self, preds, gt_labels, trans_feat):
        loss = classification_loss(preds, gt_labels.argmax(dim=1), trans_feat, 1e-3)
        return loss

    def train_epoch(self):
        train_loss = 0
        train_acc = 0
        for i, (inp_pts, gt_labels) in enumerate(tqdm(self.train_loader)):
            self.optimizer.zero_grad()
            gt_labels = gt_labels.to(self.device)
            preds, trans_feat = self.forward_step(inp_pts)
            loss = self.get_loss(preds, gt_labels, trans_feat)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            pred_labels = torch.max(preds, dim=1).indices
            this_correct = pred_labels.eq(gt_labels.argmax(dim=-1)).sum().item()
            train_acc += this_correct/gt_labels.shape[0]

        return train_loss/len(self.train_loader), train_acc/len(self.train_loader)

    def valid_epoch(self):
        val_loss = 0
        val_acc = 0
        for i, (inp_pts, gt_labels) in enumerate(tqdm(self.valid_loader, leave=False)):
            gt_labels = gt_labels.to(self.device)
            preds, trans_feat = self.forward_step(inp_pts)
            loss = self.get_loss(preds, gt_labels, trans_feat)
            val_loss += loss.item()
            pred_labels = torch.max(preds, dim=1).indices
            this_correct = pred_labels.eq(gt_labels.argmax(dim=-1)).sum().item()
            val_acc += this_correct/gt_labels.shape[0]

        return val_loss/len(self.valid_loader), val_acc/len(self.valid_loader)

    def run(self):
        for epoch in tqdm(range(self.num_epochs)):
            self.model.train()

            if epoch % self.lr_decay_every == 0:
                self.adjust_lr()

            train_ep_loss, train_ep_acc = self.train_epoch()
            self.train_losses.append(train_ep_loss)
            self.train_accs.append(train_ep_acc)

            if epoch % self.log_interval == 0:
                val_loss, val_acc = self.valid_epoch()
                self.val_losses.append(val_loss)
                self.val_accs.append(val_acc)
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'model_latest.pth'))
                print('Epoch: {:03d}, Train Loss: {:.4f}, Train Acc: {:.2f}%, Val Loss: {:.4f}, Val Acc: {:.2f}%'.format(epoch,
                                                                                                                       train_ep_loss,
                                                                                                                       1e2*train_ep_acc,
                                                                                                                       val_loss,
                                                                                                                       1e2*val_acc))
        torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'model_final.pth'))

    def test(self):
        file_final = os.path.join(self.save_dir, 'model_final.pth')
        if not os.path.exists(file_final):
            print('-------------------------------------------------------')
            print('No final weights, switching to last weights')
            print('-------------------------------------------------------')
            file_final = os.path.join(self.save_dir, 'model_latest.pth')
        weights = torch.load(file_final)
        self.model.load_state_dict(weights)
        _, score = self.valid_epoch()
        print('Final Valid Accuracy : {:.2f}%'.format(1e2*score))


class PointNetSeg(nn.Module):
    ## TODO
    def __init__(self, n_cls, inp_dim=3, seg_hidden_dims=[512,256,128], mlp2_hidden_dims=[64,128,1024], transf_hidden_dims1=[64,128,1024], transf_hidden_dims2=[512,256]):
        """
        PointNet classification network

        Parameters
        ------------------
            n_cls (int): Number of classes for classification
            inp_dim (int): Dimension of the input points
            seg_hidden_dims (list): List of hidden dimensions for the segmentation MLP
            mlp2_hidden_dims (list): List of hidden dimensions for the second MLP block in the feature extractor
            transf_hidden_dims1 (list): List of hidden dimensions for the first MLP in each FeatureTransform block
            transf_hidden_dims2 (list): List of hidden dimensions for the second MLP in each FeatureTransform block
        """

        super().__init__()

    def forward(self, x):
      """
      Parameters
      --------------
      inp : (B, n, d) input batch in standard format.

      Output
      -------------
      out : (B, n, n_cls) output score for each class
      T_feat : (B, latent1, latent1) transformation matrix of the second FeatureTransform module
      """
      x = x.transpose(2,1) # Put input in convolution format

      return out, T_feat


class RNAMeshDataset(Dataset):
    """RNA Mesh Dataset
    """

    def __init__(self, root_dir, train, n_classes=260):

        """
            root_dir (string): Directory with all the meshes.
            train (bool): If True, use the training set, else use the test set.
            num_eig (int): Number of eigenvalues to use.
            op_cache_dir (string): Directory to cache the operators.
            n_classes (int): Number of classes.
        """

        self.train = train  # bool
        self.root_dir = root_dir
        self.n_class = n_classes # (includes -1)

        # store in memory
        self.verts_list = []
        self.faces_list = []
        self.labels_list = []  # per-vertex

        # Load the meshes & labels
        if self.train:
            with open(os.path.join(self.root_dir, "train.txt")) as f:
                this_files = [line.rstrip() for line in f]
        else:
            with open(os.path.join(self.root_dir, "test.txt")) as f:
                this_files = [line.rstrip() for line in f]

        print("loading {} files: {}".format(len(this_files), this_files))

        # Load the actual files

        off_path = os.path.join(root_dir, "off")
        label_path = os.path.join(root_dir, "labels")
        for f in this_files:
            off_file = os.path.join(off_path, f)
            label_file = os.path.join(label_path, f[:-4] + ".txt")

            verts, faces = read_mesh(off_file)
            labels = np.loadtxt(label_file).astype(int) + 1 # shift -1 --> 0

            verts = torch.tensor(verts).float()
            faces = torch.tensor(faces)
            labels = torch.tensor(labels)

            # center and unit scale
            verts = normalize_positions(verts)

            self.verts_list.append(verts)
            self.faces_list.append(faces)
            self.labels_list.append(labels)


    def __len__(self):
        return len(self.verts_list)

    def __getitem__(self, idx):
        return self.verts_list[idx], self.faces_list[idx], self.labels_list[idx]


def segmentation_loss(preds, gt_labels, T_feat, w_reg):
        """
        Classification loss for PointNet

        Parameters
        --------------
        preds : (B, n_cls) - output of the segmentation network
        gt_labels : (B, n) - ground truth labels
        T_feat : (B, latent1, latent1) - transformation matrix of the second FeatureTransform module
        w_reg : weight of the regularization term

        Output
        -------------
        loss : segmentation loss L_{seg} + w_reg * L_{reg}
        """
        return loss


class TrainerSeg(object):

    def __init__(self, train_loader, valid_loader, device='cuda',
                 lr=1e-3, weight_decay=1e-4, num_epochs=200,
                 lr_decay_every = 50, lr_decay_rate = 0.5,
                 log_interval=10, save_dir=None, **model_cfg):

        """
        pointNet_cls: (nn.Module) class of the PointNet (or ++) model.
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
        model_cfg: (dict) keyword arguments for model
        """
        self.model = self.get_model(model_cfg).to(device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.loss = torch.nn.CrossEntropyLoss()

        self.lr_decay_every = lr_decay_every
        self.lr_decay_rate = lr_decay_rate
        self.log_interval = log_interval
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []

        self.inp_feat = model_cfg.get('inp_feat', 'xyz')
        self.num_eig = model_cfg.get('num_eig', 128)
        if not self.inp_feat in ['xyz', 'xyzn', 'hks', 'wks']:
            raise ValueError('inp_feat must be one of xyz, xyzn, hks, wks')

        self.model.to(self.device)

    def get_model(self, model_cfg):
        if model_cfg['name'].lower() == 'pointnet':
            return PointNetSeg(n_cls=model_cfg['n_cls'],
                              inp_dim=model_cfg['inp_dim'],
                              seg_hidden_dims=model_cfg['seg_hidden_dims'],
                              mlp2_hidden_dims=model_cfg['mlp2_hidden_dims'],
                              transf_hidden_dims1=model_cfg['transf_hidden_dims1'],
                              transf_hidden_dims2=model_cfg['transf_hidden_dims2'])
        else:
            raise ValueError('%s must be one of PointNet, PointNet++'%model_cfg['name'])

    def adjust_lr(self):
        lr = self.lr * self.lr_decay_rate
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def forward_step(self, inp):
        """
        Perform a forward step of the model.

        Args:
            inp (torch.Tensor): (N, D) tensor of input features

        Returns:
            pred (torch.Tensor): (N, p_out) tensor of predicted labels.
        """
        inp = inp.to(self.device)
        verts = inp[..., :3].to(self.device)
        norms = inp[..., 3:].to(self.device)
        preds, trans_feat = self.model(inp)
        return preds, trans_feat



    def get_loss(self, preds, gt_labels, trans_feat):

        loss = segmentation_loss(preds, gt_labels, trans_feat, 1e-3)
        return loss

    def train_epoch(self):
        train_loss = 0
        train_acc = 0
        for i, (inp_pts, gt_labels) in enumerate(tqdm(self.train_loader)):
            self.optimizer.zero_grad()
            gt_labels = gt_labels.to(self.device)
            preds, trans_feat = self.forward_step(inp_pts)
            loss = self.get_loss(preds[0], gt_labels[0], trans_feat)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()

            pred_labels = torch.max(preds[0], dim=1).indices  # (N,)
            this_correct = pred_labels.eq(gt_labels).sum().item()
            train_acc += this_correct / inp_pts.shape[-2]

        return train_loss/len(self.train_loader), train_acc/len(self.train_loader)

    def valid_epoch(self):
        val_loss = 0
        val_acc = 0
        total = 0
        for i, (inp_pts, gt_labels) in enumerate(self.valid_loader):
            gt_labels = gt_labels.to(self.device)
            preds, trans_feat = self.forward_step(inp_pts)
            loss = self.get_loss(preds[0], gt_labels[0], trans_feat)
            val_loss += loss.item()

            pred_labels = torch.max(preds[0], dim=1).indices
            this_correct = pred_labels.eq(gt_labels).sum().item()
            total += inp_pts.shape[-2]
            val_acc += this_correct

        return val_loss/total, val_acc/total

    def run(self):
        for epoch in tqdm(range(self.num_epochs)):
            self.model.train()

            if epoch % self.lr_decay_every == 0:
                self.adjust_lr()

            train_ep_loss, train_ep_acc = self.train_epoch()
            self.train_losses.append(train_ep_loss)
            self.train_accs.append(train_ep_acc)

            if epoch % self.log_interval == 0:
                val_loss, val_acc = self.valid_epoch()
                self.val_losses.append(val_loss)
                self.val_accs.append(val_acc)
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'model_latest.pth'))
                print('Epoch: {:03d}, Train Loss: {:.4f}, Train Acc: {:.2f}, Val Loss: {:.4f}, Val Acc: {:.2f}%'.format(epoch,
                                                                                                                       train_ep_loss,
                                                                                                                       1e2*train_ep_acc,
                                                                                                                       val_loss,
                                                                                                                       1e2*val_acc))
        torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'model_final.pth'))

    def visualize(self):
        """
        We only test the first two shapes of validation set.
        """
        self.model.eval()
        test_seg_meshes = []

        for i, (inp_pts, inp_faces, gt_labels) in enumerate(self.valid_loader):
            gt_labels = gt_labels.to(self.device)
            preds, trans_feat = self.forward_step(inp_pts)
            pred_labels = torch.max(preds[0], dim=1).indices
            test_seg_meshes.append([TriMesh(inp_pts.squeeze().cpu().numpy(), inp_faces.squeeze().cpu().numpy()),
                                  pred_labels.squeeze().cpu().numpy()])
            if i==1:
                break

        cmap1 = plt.get_cmap("jet")(test_seg_meshes[0][-1] / (146))[:,:3]
        cmap2 = plt.get_cmap("jet")(test_seg_meshes[1][-1] / (146))[:,:3]

        plu.double_plot(test_seg_meshes[0][0], test_seg_meshes[1][0], cmap1, cmap2)

    def test(self):
        file_final = os.path.join(self.save_dir, 'model_final.pth')
        if not os.path.exists(file_final):
            print('-------------------------------------------------------')
            print('No final weights, switching to last weights')
            print('-------------------------------------------------------')
            file_final = os.path.join(self.save_dir, 'model_latest.pth')
        weights = torch.load(file_final)
        self.model.load_state_dict(weights)
        _, score = self.valid_epoch()
        print('Final Valid Segmentation Accuracy : {:.2f}%'.format(1e2*score))


__all__ = ['FeatTransform', 'PointNetfeat', 'PointNetCls', 'random_scale_transl', 'random_rotation', 'load_h5', 'load_h5_files', 'ModelNet40Generator', 'classification_loss', 'Trainer', 'PointNetSeg', 'RNAMeshDataset', 'segmentation_loss', 'TrainerSeg']
