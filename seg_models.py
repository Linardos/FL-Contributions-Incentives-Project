import torch
import torch.nn as nn
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def preprocess_mask_labels(mask: np.ndarray):

    mask_WT = mask.copy()
    mask_WT[mask_WT == 1] = 1 # eticheta 1 = necrotic / non-enhancing tumor core
    mask_WT[mask_WT == 2] = 1 # eticheta 2 = peritumoral edema
    mask_WT[mask_WT == 4] = 1 # eticheta 4 = enhancing tumor core

    mask_TC = mask.copy()
    mask_TC[mask_TC == 1] = 1
    mask_TC[mask_TC == 2] = 0
    mask_TC[mask_TC == 4] = 1

    mask_ET = mask.copy()
    mask_ET[mask_ET == 1] = 0
    mask_ET[mask_ET == 2] = 0
    mask_ET[mask_ET == 4] = 1

    mask = np.stack([mask_WT, mask_TC, mask_ET], axis=1)
    # mask = np.moveaxis(mask, (0, 1, 2, 3), (0, 3, 2, 1)) # mutam axele pentru a putea vizualiza mastile ulterior

    return mask

class DiceLoss(nn.Module):
    """Soft-Dice for multi-label masks (expects logits)."""
    def __init__(self, eps: float = 1e-6):
        super().__init__();  self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        probs = torch.sigmoid(logits)                       # (B,C,D,H,W) ∈ (0,1)
        dims  = tuple(range(2, probs.ndim))                 # sum over D,H,W
        inter = 2 * (probs * targets).sum(dim=dims)         # (B,C)
        union = probs.sum(dim=dims) + targets.sum(dim=dims) + self.eps
        dice  = inter / union                               # (B,C)
        return 1 - dice.mean()                              # scalar


class BCEDiceLoss(nn.Module):
    def __init__(self, dice_w: float = 1.0, bce_w: float = 0.1):
        super().__init__()
        self.dice_w = dice_w;  self.bce_w = bce_w
        self.bce  = nn.BCEWithLogitsLoss()   # raw logits here
        self.dice = DiceLoss()

    def forward(self, logits, targets):
        return self.dice_w * self.dice(logits, targets) + \
               self.bce_w  * self.bce (logits, targets)


class ResDoubleConv(nn.Module):
    """ BN -> ReLU -> Conv3D -> BN -> ReLU -> Conv3D """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.InstanceNorm3d(in_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )
        self.skip = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(out_channels),
        )

    def forward(self, x):
        return self.double_conv(x) + self.skip(x)


class ResDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.MaxPool3d(2, 2),
            ResDoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.encoder(x)


class ResUp(nn.Module):
    def __init__(self, in_channels, out_channels, trilinear=False):
        super().__init__()
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = ResDoubleConv(in_channels + in_channels // 2, out_channels)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=3, stride=2,
                                         padding=1, output_padding=1)
            self.conv = ResDoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        x1 = torch.nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2, diffZ // 2, diffZ - diffZ // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Out(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class ResUNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, n_channels=32):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_channels = n_channels

        self.input_layer = nn.Sequential(
            nn.Conv3d(in_channels, n_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm3d(n_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(n_channels, n_channels, kernel_size=3, stride=1, padding=1)
        )
        self.input_skip = nn.Conv3d(in_channels, n_channels, kernel_size=3, stride=1, padding=1)
        self.enc1 = ResDown(n_channels, 2 * n_channels)
        self.enc2 = ResDown(2 * n_channels, 4 * n_channels)
        self.enc3 = ResDown(4 * n_channels, 8 * n_channels)
        self.bridge = ResDown(8 * n_channels, 16 * n_channels)
        self.dec1 = ResUp(16 * n_channels, 8 * n_channels)
        self.dec2 = ResUp(8 * n_channels, 4 * n_channels)
        self.dec3 = ResUp(4 * n_channels, 2 * n_channels)
        self.dec4 = ResUp(2 * n_channels, n_channels)
        self.out = Out(n_channels, out_channels)

    def forward(self, x):
        x1 = self.input_layer(x) + self.input_skip(x)
        # x1:n -> x2:2n
        x2 = self.enc1(x1)
        # x2:2n -> x3:4n
        x3 = self.enc2(x2)
        # x3:4n -> x4:8n
        x4 = self.enc3(x3)
        # x4:8n -> x5:16n
        bridge = self.bridge(x4)
        mask = self.dec1(bridge, x4)
        mask = self.dec2(mask, x3)
        mask = self.dec3(mask, x2)
        mask = self.dec4(mask, x1)
        mask = self.out(mask)
        return mask

# utils.py  – new version
class LocalUpdateMONAI(object):
    """
    Federated-learning helper that performs `local_ep` epochs of training
    on a MONAI dict-style DataLoader and returns the updated weights.

    Parameters
    ----------
    lr : float
        Optimiser learning-rate.
    local_ep : int
        Number of local epochs.
    trainloader : torch.utils.data.DataLoader
        Yields dictionaries with image modalities and a 'seg' key.
    img_keys : Sequence[str], optional
        Ordered list of modality keys to concatenate into a 5-D tensor
        (B, C, D, H, W).  Default corresponds to BraTS.
    label_key : str, optional
        Dict key that contains the integer mask.  Default: 'seg'.
    """
    def __init__(
        self,
        lr: float,
        local_ep: int,
        trainloader,
        img_keys=("flair", "t1", "t1ce", "t2"),
        label_key="seg",
    ):
        self.lr = lr
        self.local_ep = local_ep
        self.trainloader = trainloader
        self.img_keys = img_keys
        self.label_key = label_key

    @torch.no_grad()
    def _stack_modalities(self, batch):
        """Convert dict-batch → (B, C, D, H, W) + (B, D, H, W) mask."""
        imgs = torch.cat([batch[k] for k in self.img_keys], dim=1)  # C=4
        mask = batch[self.label_key] # (B, 1, D, H, W)
        mask = mask.squeeze(1)                     # drop ch-dim
        # preprocess mask labels
        mask = preprocess_mask_labels(mask.numpy())                
        # convert mask to tensor
        mask = torch.tensor(mask, dtype=torch.float32)
        # print("mask", mask.shape, mask.__class__)
        return imgs.to(device), mask.to(device)

    def update_weights(self, model):
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = BCEDiceLoss().to(device)  # Use BCEDiceLoss for segmentation
        scaler = torch.amp.GradScaler()           # AMP scaler

        epoch_loss = []
        for _ in range(self.local_ep):
            batch_loss = []
            for batch in self.trainloader:             # MONAI yields dicts
                images, labels = self._stack_modalities(batch)
                optimizer.zero_grad()
                with torch.amp.autocast(device_type="cuda"):        # Enable AMP
                    logits = model(images)            # (B, C, D, H, W)
                    # print("logits", logits.shape, "labels", labels.shape)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()         # Scale loss for AMP
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)                # Step optimizer
                scaler.update()                       # Update scaler
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)