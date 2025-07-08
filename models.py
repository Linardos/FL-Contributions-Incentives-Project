import torch
import torch.nn as nn
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# models.py

# class ConvBlock(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         self.block = nn.Sequential(
#             nn.Conv3d(in_ch, out_ch, 3, padding=1),
#             nn.InstanceNorm3d(out_ch),
#             nn.ReLU(inplace=True),
#             nn.Conv3d(out_ch, out_ch, 3, padding=1),
#             nn.InstanceNorm3d(out_ch),
#             nn.ReLU(inplace=True),
#         )

#     def forward(self, x):
#         return self.block(x)

# class UpBlock(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
#         self.conv = ConvBlock(in_ch, out_ch)

#     def forward(self, x1, x2):
#         x1 = self.up(x1)
#         x = torch.cat([x2, x1], dim=1)
#         return self.conv(x)

# class ResUNet3D(nn.Module):
#     def __init__(self, in_channels=4, out_channels=3, base_filters=32):
#         super().__init__()
#         self.enc1 = ConvBlock(in_channels, base_filters)
#         self.enc2 = ConvBlock(base_filters, base_filters * 2)
#         self.enc3 = ConvBlock(base_filters * 2, base_filters * 4)
#         self.bottleneck = ConvBlock(base_filters * 4, base_filters * 8)

#         self.pool = nn.MaxPool3d(2)

#         self.up2 = UpBlock(base_filters * 8, base_filters * 4)
#         self.up1 = UpBlock(base_filters * 4, base_filters * 2)
#         self.up0 = UpBlock(base_filters * 2, base_filters)

#         self.final = nn.Conv3d(base_filters, out_channels, kernel_size=1)
#         self.softmax = nn.Softmax(dim=1)

#     def forward(self, x):
#         e1 = self.enc1(x)
#         e2 = self.enc2(self.pool(e1))
#         e3 = self.enc3(self.pool(e2))
#         b = self.bottleneck(self.pool(e3))
#         d2 = self.up2(b, e3)
#         d1 = self.up1(d2, e2)
#         d0 = self.up0(d1, e1)
#         out = self.final(d0)
#         return self.softmax(out)

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
    def __init__(self, eps: float = 1e-9):
        super(DiceLoss, self).__init__()
        self.eps = eps
        
    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        
        num = targets.size(0)
        probability = torch.sigmoid(logits)
        probability = probability.view(num, -1)
        targets = targets.view(num, -1)
        assert(probability.shape == targets.shape)
        
        intersection = 2.0 * (probability * targets).sum()
        union = probability.sum() + targets.sum()
        dice_score = (intersection + self.eps) / union
        #print("intersection", intersection, union, dice_score)
        return 1.0 - dice_score

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        
    def forward(self, 
                logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        assert(logits.shape == targets.shape)
        dice_loss = self.dice(logits, targets)
        bce_loss = self.bce(logits, targets)
        
        return bce_loss + dice_loss
    

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
                scaler.step(optimizer)                # Step optimizer
                scaler.update()                       # Update scaler
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)


# _____________________ classification for CIFAR under here

class ResNet9(nn.Module):
    def __init__(self):
        super(ResNet9, self).__init__()
        self.prep = self.convbnrelu(channels=3, filters=64)
        self.layer1 = self.convbnrelu(64, 128)
        self.layer_pool = nn.MaxPool2d(2, 2, 0, 1, ceil_mode=False)
        self.layer1r1 = self.convbnrelu(128, 128)
        self.layer1r2 = self.convbnrelu(128, 128)
        self.layer2 = self.convbnrelu(128, 256)
        self.layer3 = self.convbnrelu(256, 512)
        self.layer3r1 = self.convbnrelu(512, 512)
        self.layer3r2 = self.convbnrelu(512, 512)
        self.out_pool = nn.MaxPool2d(kernel_size=4, stride=4, ceil_mode=False)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=512, out_features=10, bias=False)

    def convbnrelu(self, channels, filters):
        layers = []
        layers.append(nn.Conv2d(channels, filters, (3, 3),
                                (1, 1), (1, 1), bias=False))
        layers.append(nn.BatchNorm2d(filters, track_running_stats=False))
        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.prep(x)
        x = self.layer_pool(self.layer1(x))
        r1 = self.layer1r2(self.layer1r1(x)) 
        x = x + r1
        x = self.layer_pool(self.layer2(x))
        x = self.layer_pool(self.layer3(x))
        r3 = self.layer3r2(self.layer3r1(x))
        x = x + r3
        out = self.out_pool(x)
        out = self.flatten(out)
        out = self.linear(out)
        out = out * 0.125

        return out
        
class LocalUpdate(object):

    def __init__(self, lr, local_ep, trainloader):
        self.lr = lr
        self.local_ep = local_ep
        self.trainloader = trainloader

    def update_weights(self, model):

        model.train()
        epoch_loss = []
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss().to(device)
        for iter in range(self.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(device), labels.to(device)
                model.zero_grad()   
                log_probs = model(images)
                loss = criterion(log_probs, labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)