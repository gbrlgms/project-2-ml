"""Image datasets, neural-network layers, and training helpers.

The module contains two image classifiers: :class:`ArtistCNN`, a compact
convolutional network trained from scratch, and :class:`TransferNet`, which
uses frozen ImageNet MobileNetV2 features with a newly trained classification
head. Both expect batches of RGB images represented as tensors in the 0–255
range and return one logit per class.
"""

from copy import deepcopy
from math import pi

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import f1_score
from torch.utils.data import Dataset
from torchvision.transforms import v2

from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

IMG_SIZE = 224

# ---------- Preprocessing / Augmentation layers ----------
def _warp(x, theta):
    """Apply a separate affine transform to every image in a batch.

    ``theta`` is a batch of 2-by-3 affine matrices in normalized image
    coordinates. ``affine_grid`` creates the sampling locations and
    ``grid_sample`` interpolates the input pixels at those locations. The
    reflection padding fills areas exposed at image boundaries.

    Args:
        x: Image batch shaped ``(N, C, H, W)``.
        theta: Affine matrices shaped ``(N, 2, 3)``.

    Returns:
        Warped image batch with the same shape and device as ``x``.
    """
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear",
                         padding_mode="reflection", align_corners=False)


class RandomFlip(nn.Module):
    """Randomly mirror each image horizontally during training.

    A separate Bernoulli decision with probability 0.5 is drawn for each
    batch item. Evaluation mode returns the input unchanged.
    """
    def forward(self, x):
        """Flip selected items along the width axis of an ``NCHW`` batch."""
        if not self.training:
            return x
        mask = torch.rand(x.size(0), device=x.device) < 0.5
        return torch.where(mask[:, None, None, None], x.flip(-1), x)


class RandomRotation(nn.Module):
    """Apply an independent random rotation to each training image.

    ``factor`` determines the maximum rotation as a fraction of a full turn;
    for example, ``0.05`` means angles from -18 to +18 degrees. Evaluation
    mode disables augmentation.
    """
    def __init__(self, factor=0.05):          # 0.05 -> ±18°
        """Store the maximum absolute fraction of a full-turn rotation."""
        super().__init__()
        self.factor = factor

    def forward(self, x):
        """Rotate a batch by sampled angles, leaving it unchanged in eval mode."""
        if not self.training:
            return x
        n = x.size(0)
        angle = (torch.rand(n, device=x.device) * 2 - 1) * self.factor * 2 * pi
        cos, sin = torch.cos(angle), torch.sin(angle)
        theta = torch.zeros(n, 2, 3, device=x.device)
        theta[:, 0, 0], theta[:, 0, 1] = cos, -sin
        theta[:, 1, 0], theta[:, 1, 1] = sin, cos
        return _warp(x, theta)


class RandomZoom(nn.Module):
    """Apply an independent random zoom to each training image.

    Each image receives a scale sampled from ``[1-factor, 1+factor]``.
    Sampling matrices use the reciprocal scale so values above one zoom in.
    Evaluation mode disables augmentation.
    """
    def __init__(self, factor=0.1):
        """Store the maximum proportional zoom in either direction."""
        super().__init__()
        self.factor = factor

    def forward(self, x):
        """Zoom a batch around its center, leaving it unchanged in eval mode."""
        if not self.training:
            return x
        n = x.size(0)
        zoom = 1 + (torch.rand(n, device=x.device) * 2 - 1) * self.factor
        theta = torch.zeros(n, 2, 3, device=x.device)
        theta[:, 0, 0] = 1 / zoom
        theta[:, 1, 1] = 1 / zoom
        return _warp(x, theta)


class Rescaling(nn.Module):
    def __init__(self, scale=1 / 255):
        """Initialize a fixed multiplier used to rescale pixel values."""
        super().__init__()
        self.scale = scale

    def forward(self, x):
        """Multiply every input pixel by the configured scale."""
        return x * self.scale


def conv_block(c_in, c_out):
    """Build one convolution, activation, and downsampling block.

    The 3-by-3 convolution uses padding to preserve height and width. ReLU
    adds nonlinearity, then 2-by-2 max pooling halves each spatial dimension.
    This block is used repeatedly by :class:`ArtistCNN` to build richer
    feature maps while reducing computation.

    Args:
        c_in: Number of input channels.
        c_out: Number of convolution output channels.

    Returns:
        An ``nn.Sequential`` module implementing the block.
    """
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class ArtistCNN(nn.Module):
    """Convolutional artist classifier trained from scratch.

    Layer flow:

    1. ``preprocess`` randomly flips, rotates, and zooms training images, then
       scales their 0–255 pixel values to 0–1. The augmentation modules turn
       off automatically in evaluation mode.
    2. ``features`` applies three ``conv_block`` stages: 3→32, 32→64, and
       64→128 channels. Each stage extracts local patterns with a convolution,
       applies ReLU, and halves image width and height with max pooling.
    3. ``pool`` uses adaptive average pooling to reduce each 128-channel
       feature map to one value, regardless of its spatial dimensions.
    4. ``head`` flattens those values, applies dropout during training, and
       maps the 128 features to class logits with a linear layer.

    The output is logits rather than probabilities; cross-entropy loss
    expects logits and applies the appropriate normalization internally.
    """
    def __init__(self, num_classes=6, dropout=0.4):
        """Construct the network for ``num_classes`` labels.

        Args:
            num_classes: Number of output artist classes.
            dropout: Probability of dropping head activations during training.
        """
        super().__init__()
        self.preprocess = nn.Sequential(
            RandomFlip(),
            RandomRotation(0.05),
            RandomZoom(0.1),
            Rescaling(1 / 255),
        )
        self.features = nn.Sequential(
            conv_block(3, 32),
            conv_block(32, 64),
            conv_block(64, 128),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)    # GlobalAveragePooling2D
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),       # logits (softmax applied in the loss)
        )

    def forward(self, x):                      # x: raw pixels in 0-255
        """Transform an ``NCHW`` RGB batch into per-class logits."""
        x = self.preprocess(x.float())
        x = self.features(x)
        return self.head(self.pool(x))

    @torch.no_grad()
    def predict_proba(self, x):                # softmax for inference / the demo
        """Return class probabilities and switch the model to evaluation mode.

        Gradients are disabled by the decorator. The method leaves the model
        in evaluation mode after the call.
        """
        self.eval()
        return F.softmax(self(x), dim=1)

class EarlyStopping:
    """Track validation loss and retain a snapshot of the best model weights.

    An epoch is an improvement only when its loss is lower than the best loss
    by more than ``min_delta``. Consecutive non-improving epochs increment
    ``wait``; reaching ``patience`` tells the caller to stop training.
    """
    def __init__(self, patience=5, min_delta=0.0, restore_best_weights=True):
        """Initialize early-stopping thresholds and empty best-state tracking.

        Args:
            patience: Number of consecutive non-improving epochs allowed.
            min_delta: Minimum loss decrease counted as an improvement.
            restore_best_weights: Whether :meth:`restore` loads the best state.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float("inf")
        self.best_epoch = 0
        self.best_state = None
        self.wait = 0

    def step(self, val_loss, model, epoch):
        """Record a validation result and report whether training should stop.

        On improvement, a deep copy of the model state is saved so later
        optimizer updates cannot mutate the snapshot. Otherwise the patience
        counter advances.

        Args:
            val_loss: Validation loss for the current epoch.
            model: Model whose state should be snapshotted on improvement.
            epoch: Current epoch number, retained as ``best_epoch``.

        Returns:
            ``True`` when the number of non-improving epochs reaches patience.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.best_state = deepcopy(model.state_dict())
            self.wait = 0
            return False
        self.wait += 1
        return self.wait >= self.patience

    def restore(self, model):
        """Load the saved best weights when restoration is enabled and available."""
        if self.restore_best_weights and self.best_state is not None:
            model.load_state_dict(self.best_state)

class TransferNet(nn.Module):
    """Artist classifier built from frozen pretrained MobileNetV2 features.

    Layer flow:

    1. Inputs are converted to floating point and scaled from 0–255 to 0–1.
    2. ``augment`` applies random horizontal flips, rotations, and zooms in
       training mode only.
    3. ``mean`` and ``std`` buffers normalize RGB channels with ImageNet
       statistics expected by the pretrained backbone.
    4. ``base`` is the feature-extraction portion of ImageNet-pretrained
       MobileNetV2, without its original classifier. Its parameters are frozen
       and its BatchNorm layers stay in evaluation mode.
    5. ``pool`` averages each of the 1280 feature maps to one value. ``head``
       flattens those values, applies dropout, and produces class logits.

    Only the new classification head is trainable by default. The stored
    normalization tensors are buffers, so they move with the model and are
    included in its state dictionary without being optimized.
    """
    def __init__(self, num_classes=6, dropout=0.3):
        """Load the pretrained feature extractor and create a new class head.

        Args:
            num_classes: Number of output artist classes.
            dropout: Drop probability before the output linear layer.
        """
        super().__init__()
        self.augment = nn.Sequential(RandomFlip(), RandomRotation(0.05), RandomZoom(0.1))

        # MobileNetV2's own preprocessing: scale to 0-1, then ImageNet mean/std
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # include_top=False, weights="imagenet"
        self.base = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT).features
        for p in self.base.parameters():          # base.trainable = False
            p.requires_grad = False

        self.pool = nn.AdaptiveAvgPool2d(1)       # pooling="avg"
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(1280, num_classes),         # new output layer (logits)
        )

    def train(self, mode=True):
        """Set training mode while keeping frozen backbone BatchNorm stable."""
        super().train(mode)
        self.base.eval()      # keep the frozen BatchNorm layers on their ImageNet statistics
        return self

    def forward(self, x):                         # x: raw pixels 0-255
        """Normalize images, extract MobileNet features, and return logits."""
        x = x.float() / 255
        x = self.augment(x)
        x = (x - self.mean) / self.std
        return self.head(self.pool(self.base(x)))

    @torch.no_grad()
    def predict_proba(self, x):
        """Return class probabilities and leave the model in evaluation mode."""
        self.eval()
        return F.softmax(self(x), dim=1)

# ---------- training ----------

def run_epoch(net, loader, criterion, device, optimizer=None):
    """Run one training or evaluation pass over a data loader.

    Supplying an optimizer selects training mode: gradients are enabled and
    each batch updates the model. Without an optimizer the model is put in
    evaluation mode and only loss/metrics are computed. Loss is accumulated
    weighted by batch size so a short final batch is handled correctly.

    Args:
        net: Classifier returning class logits.
        loader: Iterable yielding image batches and integer labels.
        criterion: Loss function accepting ``(logits, labels)``.
        device: Device to which images and labels are moved.
        optimizer: Optimizer used for training, or ``None`` for evaluation.

    Returns:
        Tuple ``(mean_loss, macro_f1)`` over all examples in the loader.
    """
    train = optimizer is not None
    net.train(train)
    total_loss, preds, targets = 0.0, [], []
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = net(x)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            preds += logits.argmax(1).cpu().tolist()
            targets += y.cpu().tolist()
    return total_loss / len(loader.dataset), f1_score(targets, preds, average="macro")

def fit(net, optimizer, train_dl, val_dl, criterion, device,
        max_epochs=50, patience=5):
    """Train a model, validate after each epoch, and restore its best weights.

    Each epoch runs a training pass followed by a validation pass, records
    loss and macro F1 for both, and prints a progress line. Early stopping
    monitors validation loss; on exit the best recorded model weights are
    restored. Training ends at ``max_epochs`` if patience is not reached.

    Args:
        net: Model to train.
        optimizer: Optimizer configured for the trainable model parameters.
        train_dl: Training data loader.
        val_dl: Validation data loader.
        criterion: Loss function used for both passes.
        device: Device for model inputs and labels (the model should be placed
            on this device by the caller).
        max_epochs: Maximum number of training epochs.
        patience: Consecutive non-improving validation epochs before stopping.

    Returns:
        Tuple ``(history, best_epoch)``. ``history`` contains lists for
        ``train_loss``, ``val_loss``, ``train_f1``, and ``val_f1``. The epoch
        count starts at one; it is zero if no best state was recorded.
    """
    stopper = EarlyStopping(patience=patience, restore_best_weights=True)
    hist = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}
    for epoch in range(1, max_epochs + 1):
        tr_loss, tr_f1 = run_epoch(net, train_dl, criterion, device, optimizer)
        va_loss, va_f1 = run_epoch(net, val_dl, criterion, device)
        for k, v in zip(hist, (tr_loss, va_loss, tr_f1, va_f1)):
            hist[k].append(v)
        print(f"epoch {epoch:2d} | train loss {tr_loss:.3f} F1 {tr_f1:.3f} | "
              f"val loss {va_loss:.3f} F1 {va_f1:.3f}")
        if stopper.step(va_loss, net, epoch):
            print(f"Early stopping at epoch {epoch}. Best epoch: {stopper.best_epoch}")
            break
    stopper.restore(net)
    return hist, stopper.best_epoch

@torch.no_grad()
def evaluate(net, loader, device, criterion):
    """Compute average loss and predictions for a labeled data loader.

    The model is switched to evaluation mode and gradients are disabled. The
    loss is averaged by example count, while predictions and targets are
    returned as Python integer lists in loader order.

    Args:
        net: Classifier returning logits.
        loader: Iterable yielding image batches and integer labels.
        device: Device to which each batch is moved.
        criterion: Loss function accepting ``(logits, labels)``.

    Returns:
        Tuple ``(mean_loss, targets, predictions)``.
    """
    net.eval()
    preds, targets, loss = [], [], 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = net(x)
        loss += criterion(logits, y).item() * len(y)
        preds += logits.argmax(1).cpu().tolist()
        targets += y.cpu().tolist()
    return loss / len(loader.dataset), targets, preds


# ---------- data ----------
class PaintingDataset(Dataset):
    """Dataset that loads paintings and maps artist names to integer labels.

    Images are resized and center-cropped to ``IMG_SIZE`` square pixels, then
    converted to channel-first uint8 tensors. Pixel values remain in 0–255;
    model preprocessing performs scaling. The dataframe is reset internally
    so dataset indices always address rows from zero to ``len(df)-1``.
    """
    def __init__(self, df, class_to_idx):
        """Prepare image paths, encoded labels, and deterministic transforms.

        Args:
            df: DataFrame with ``path`` and ``artist`` columns.
            class_to_idx: Mapping from artist name to integer class label.
        """
        self.df = df.reset_index(drop=True)
        self.paths = df["path"].tolist()
        self.labels = df["artist"].map(class_to_idx).tolist()
        # keep uint8 0-255: the model's Rescaling layer does the /255
        self.tf = v2.Compose([v2.Resize(IMG_SIZE), v2.CenterCrop(IMG_SIZE), v2.PILToTensor()])

    def __len__(self):
        """Return the number of image rows in this dataset."""
        return len(self.paths)

    def __getitem__(self, i):
        """Load and transform one image, returning its tensor and class index."""
        img = Image.open(self.paths[i]).convert("RGB")
        return self.tf(img), self.labels[i]


# def compare_models(models, X, y, cv, scoring):
#     """Cross-validate each model on the same folds and return a tidy table."""
#     pass


# def plot_errors(y_true, y_pred):
#     """Show where the model goes wrong: a confusion matrix or the residuals."""
#     pass
