import logging

import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm

from cubicasa_datasets import CubiCasaDetectionDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DEVICE = "cpu"  # forced off MPS: two crashes in a row during MPS training on this
# machine, including a kernel panic from a WindowServer/GPU watchdog timeout —
# MPS and WindowServer share the GPU on Apple Silicon.
NUM_CLASSES = 3  # background, window, door


def collate_fn(batch):
    return tuple(zip(*batch))


def build_model() -> torch.nn.Module:
    """COCO-pretrained Faster R-CNN, re-headed for window/door detection (paper Sect. 3)."""
    model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    return model


def train(data_dir: str = "data/cubicasa5k", epochs: int = 4, batch_size: int = 2,
          size: int = 512, lr: float = 5e-5,
          checkpoint_path: str = "fasterrcnn_doors_windows.pt") -> None:
    train_loader = DataLoader(CubiCasaDetectionDataset(data_dir, split="train.txt", size=size),
                               batch_size=batch_size, shuffle=True, num_workers=4,
                               collate_fn=collate_fn)

    model = build_model().to(DEVICE)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for images, targets in tqdm(train_loader, desc=f"epoch {epoch} train"):
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        log.info(f"epoch {epoch}: train loss {running_loss / len(train_loader):.4f}")

        torch.save(model.state_dict(), checkpoint_path)
        log.info(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    train()
