
import os
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from glob import glob
import mlflow
from pytorch_lightning.loggers import MLFlowLogger
from torchvision import models
NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
ALPHABET = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']
ALL_CHAR_SET = NUMBER + ALPHABET
ALL_CHAR_SET_LEN = len(ALL_CHAR_SET)
MAX_CAPTCHA = 4

class OCRDataset(Dataset):
    def __init__(self, image_files, transform=None):
        self.image_files = image_files
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def encode(self, char):
        onehot = [0] * ALL_CHAR_SET_LEN
        idx = ALL_CHAR_SET.index(char)
        onehot[idx] += 1
        return onehot

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        img = Image.open(image_file)
        img = img.convert('L')
        label = image_file.split(os.sep)[-1][:-4]
        label_oh = []
        for i in label[:MAX_CAPTCHA]:  # Ensure label length <= 5
            label_oh += self.encode(i)
        # Pad with zeros if label is shorter than MAX_CAPTCHA
        label_oh += [0] * (ALL_CHAR_SET_LEN * (MAX_CAPTCHA - len(label)))
        if self.transform is not None:
            img = self.transform(img)
        return img, np.array(label_oh), label


class OCRDataModule(pl.LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

        self.transform = transforms.Compose([
            transforms.Resize([64, 128]),
            transforms.ToTensor(),
        ])

    def setup(self, stage=None):
        train_image_files = glob(r'.\data\OCR\Train\*')
        test_image_files = glob(r'.\data\OCR\Test\*')

        self.train_dataset = OCRDataset(
            train_image_files,
            transform=self.transform
        )

        self.val_dataset = OCRDataset(
            test_image_files,
            transform=self.transform
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )



data_module = OCRDataModule(
    batch_size=32
)

data_module.setup()

for batch in data_module.train_dataloader():
    images, encoded_labels, labels = batch
    print(images.shape)  # Should be (32, 1, 64, 128)
    print(encoded_labels.shape)  # Should be (32, 180)
    print(labels)  # List of 32 strings
    break

class OCRModel(pl.LightningModule):
    def __init__(self, num_chars=ALL_CHAR_SET_LEN, max_length=MAX_CAPTCHA):
        super().__init__()
        self.save_hyperparameters()
        self.char_set = ALL_CHAR_SET

        # DenseNet backbone
        self.backbone = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)

        # Change first conv for grayscale input
        old_conv = self.backbone.features.conv0
        self.backbone.features.conv0 = nn.Conv2d(
            1,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )

        with torch.no_grad():
            self.backbone.features.conv0.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)

        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_chars * max_length)
        )

        self.inference_transform = transforms.Compose([
            transforms.Resize((64, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

    def forward(self, x):
        out = self.backbone(x)
        return out.view(-1, self.hparams.max_length, self.hparams.num_chars)

    def _decode_indices(self, indices):
        return ["".join(self.char_set[idx] for idx in row.tolist()) for row in indices]

    def _levenshtein(self, s1, s2):
        if s1 == s2:
            return 0
        if len(s1) < len(s2):
            s1, s2 = s2, s1
        previous = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1, start=1):
            current = [i]
            for j, c2 in enumerate(s2, start=1):
                insertions = previous[j] + 1
                deletions = current[j - 1] + 1
                substitutions = previous[j - 1] + (c1 != c2)
                current.append(min(insertions, deletions, substitutions))
            previous = current
        return previous[-1]

    def _shared_step(self, batch, stage="train"):
        x, y, labels = batch
        logits = self(x)  # (B, max_len, num_chars)

        batch_size = x.size(0)

        # Convert one-hot labels to indices
        y = y.view(batch_size, self.hparams.max_length, self.hparams.num_chars)
        target = y.argmax(dim=-1)  # (B, max_len)

        loss = F.cross_entropy(
            logits.view(-1, self.hparams.num_chars),
            target.view(-1)
        )

        pred = logits.argmax(dim=-1)  # (B, max_len)
        probs = torch.softmax(logits, dim=-1)
        confidence = probs.max(dim=-1).values.mean()

        # Core accuracy metrics
        char_acc = (pred == target).float().mean()
        captcha_acc = (pred == target).all(dim=1).float().mean()

        # Per-position accuracy
        pos_accs = {}
        for i in range(self.hparams.max_length):
            pos_accs[f"{stage}_pos{i+1}_acc"] = (pred[:, i] == target[:, i]).float().mean()

        # Sequence-distance metrics
        pred_texts = self._decode_indices(pred.detach().cpu())
        target_texts = list(labels)

        edit_distances = []
        normalized_edit_distances = []
        for pred_text, target_text in zip(pred_texts, target_texts):
            dist = self._levenshtein(pred_text, target_text)
            edit_distances.append(dist)
            normalized_edit_distances.append(dist / max(len(target_text), 1))

        avg_edit_distance = torch.tensor(edit_distances, dtype=torch.float32, device=self.device).mean()
        avg_normalized_edit_distance = torch.tensor(
            normalized_edit_distances, dtype=torch.float32, device=self.device
        ).mean()

        # Main logs
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log(f"{stage}_char_acc", char_acc, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log(f"{stage}_captcha_acc", captcha_acc, prog_bar=True, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log(f"{stage}_avg_edit_distance", avg_edit_distance, prog_bar=False, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log(
            f"{stage}_avg_norm_edit_distance",
            avg_normalized_edit_distance,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(f"{stage}_confidence", confidence, prog_bar=False, on_step=False, on_epoch=True, batch_size=batch_size)

        for name, value in pos_accs.items():
            self.log(name, value, prog_bar=False, on_step=False, on_epoch=True, batch_size=batch_size)

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=3
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }

    def predict_image(self, image_path):
        img = Image.open(image_path).convert("L")
        img = self.inference_transform(img)
        img = img.unsqueeze(0).to(self.device)

        self.eval()
        with torch.no_grad():
            output = self(img)
            pred_indices = output.argmax(dim=-1)

        pred_text = "".join(self.char_set[idx] for idx in pred_indices[0].tolist())
        return pred_text

from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

checkpoint_callback = ModelCheckpoint(
    monitor="val_captcha_acc",
    mode="max",
    save_top_k=1,
    filename="best-captcha-{epoch:02d}-{val_captcha_acc:.4f}"
)

early_stop_callback = EarlyStopping(
    monitor="val_captcha_acc",
    mode="max",
    patience=8
)

model = OCRModel()

mlflow.set_experiment("ocr_using_densenet")

mlflow.set_tracking_uri("file:./mlruns")

mlf_logger = MLFlowLogger(
    experiment_name="ocr_using_densenet",
    tracking_uri="file:./mlruns"
)

trainer = pl.Trainer(
    max_epochs=30,
    logger=mlf_logger,
    callbacks=[checkpoint_callback, early_stop_callback],
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1
)

with mlflow.start_run():
    mlflow.log_param("batch_size", data_module.batch_size)
    mlflow.log_param("max_captcha_length", MAX_CAPTCHA)
    mlflow.log_param("num_characters", ALL_CHAR_SET_LEN)
    mlflow.log_param("backbone", "DenseNet121")
    mlflow.log_param("image_size", "64x128")
    mlflow.log_param("optimizer", "AdamW")
    mlflow.log_param("monitor_metric", "val_captcha_acc")
    mlflow.log_param("tracked_metrics", ",".join([
        "train_loss", "val_loss",
        "train_char_acc", "val_char_acc",
        "train_captcha_acc", "val_captcha_acc",
        "train_avg_edit_distance", "val_avg_edit_distance",
        "train_avg_norm_edit_distance", "val_avg_norm_edit_distance",
        "train_confidence", "val_confidence"
    ]))

    trainer.fit(model, data_module)

    best_model_path = checkpoint_callback.best_model_path
    mlflow.log_param("best_model_path", best_model_path)

    if best_model_path:
        mlflow.log_artifact(best_model_path)

    print("Best model:", best_model_path)

torch.save(model.state_dict(), "captcha_model.pth")
mlflow.log_artifact("captcha_model.pth")

best_model_path = checkpoint_callback.best_model_path
print("Best model:", best_model_path)

model = OCRModel.load_from_checkpoint(best_model_path)

image_file = r".\data\OCR\Test\nfj8.png"
pred_text = model.predict_image(image_file)
print(pred_text)