import torch
from ultralytics import YOLO
import yaml
import json
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from PIL import Image
import numpy as np
import os
import glob

def create_model(weights_path: str = "/pytorchexample/model.pt"):
    """
    Create and return a YOLO model object initialized with given weights.
    This matches the typical Flower usage where 'create_model' returns a model object,
    not just its state dict.
    """
    yolo_model = YOLO(weights_path)
    return yolo_model


def set_weights(yolo_model, parameters):
    """Set the YOLO model's weights from a list of NumPy arrays."""
    
    state_dict = yolo_model.model.state_dict()

    for (name, old_val), new_val in zip(state_dict.items(), parameters):
        state_dict[name] = torch.tensor(new_val, dtype=old_val.dtype)

    
    yolo_model.model.load_state_dict(state_dict, strict=True)


def get_weights(yolo_model):
    """Return the model's parameters as a list of NumPy arrays."""
    return [val.cpu().numpy() for _, val in yolo_model.model.state_dict().items()]


def train(yolo: YOLO, epochs: int, lr: float, device: str, dataset_path: str,seed: int):
    train_losses = []


    def record_loss(trainer):
        loss_value = trainer.loss.detach().cpu().item()
        train_losses.append(loss_value)

    
    yolo.add_callback("on_train_epoch_end", record_loss)
    """Train the YOLO model."""
    results = yolo.train(
        data=dataset_path,
        epochs=epochs,
        lr0=lr,
        device=device,
        imgsz=640,
        single_cls=True,
        warmup_epochs=0.0,
        close_mosaic=0,
        seed=seed
    )
    return train_losses


def test(yolo_model: YOLO, device: str, cid: int):
    """
    Evaluate YOLO model on a client dataset determined by 'cid'.
    The function returns a (loss, {"map50": val}) tuple.
    """
    dataset_path = f"/WehbeDocker/YOLOV8N/datasets/client_{cid}/data.yaml"

    metrics = yolo_model.val(
        data=dataset_path,
        device=str(device),
        imgsz=640,
    )

    rd = metrics.results_dict
    map50 = rd.get("metrics/mAP50(B)", 0.0)  # example YOLOv8 metric key
    loss_box = rd.get("val/box_loss", 0.0)
    loss_cls = rd.get("val/cls_loss", 0.0)
    total_loss = loss_box + loss_cls

    return total_loss, {"map50": map50}
