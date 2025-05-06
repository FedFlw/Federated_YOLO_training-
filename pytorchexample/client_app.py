"""pytorchexample: A Flower / PyTorch YOLO detection client."""

import os
import glob
import yaml
import torch
from ultralytics import YOLO
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, RecordSet

from pytorchexample.task import (
    create_model,
    get_weights,
    set_weights,
    train,
    test
)

class YOLODetectionClient(NumPyClient):
    def __init__(
        self,
        yolo_model: YOLO,
        local_epochs: int,
        client_state: RecordSet,
        learning_rate: float,
        partition_id: int,
        seed: int
    ):
        self.client_state = client_state
        self.net = yolo_model  
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.cid = partition_id
        self.seed = seed

        # Move model to device if available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

        print(f"[Client {self.cid}] Device: {self.device}")

    def fit(self, parameters, config):
        """Train a model locally using the global parameters as a starting point."""
       
        set_weights(self.net, parameters)
        local_ckpt_path = f"model_{self.cid}.pt"
        ckpt_dict = {"model": self.net.model}  
        torch.save(ckpt_dict, local_ckpt_path)
        yolo_obj = YOLO(local_ckpt_path)




       
        dataset_path = f"/WehbeDocker/YOLOV8N/datasets/client_{self.cid}/data.yaml"
        dataset_path_forcounting = f"/WehbeDocker/YOLOV8N/datasets/client_{self.cid}/train/images"
        
        jpg_files = [file for file in os.listdir(dataset_path_forcounting) if file.lower().endswith('.jpg')]
        num_images = len(jpg_files)

        lossetrain = train(
            yolo=yolo_obj,
            epochs=self.local_epochs,
            lr=self.learning_rate,
            device=self.device,
            dataset_path=dataset_path,
            seed=self.seed,
        )
        if len(lossetrain) >= 2:
            diff = lossetrain[0] - lossetrain[-1]
        else:
            diff = lossetrain[0]


        
        updated_weights = get_weights(yolo_obj)

        
        return updated_weights, int(num_images*diff), {}

    def evaluate(self, parameters, config):
        """Evaluate global parameters on the local validation/test set."""
        
        set_weights(self.net, parameters)
        local_ckpt_path = f"model_{self.cid}.pt"
        ckpt_dict = {"model": self.net.model}  # YOLO expects 'model' or 'ema'
        torch.save(ckpt_dict, local_ckpt_path)
        yolo_obj = YOLO(local_ckpt_path)

        
        loss, metrics = test(
            yolo_model=yolo_obj,
            device=self.device,
            cid=self.cid
        )

        
        num_test_samples = 16

        # Return (loss, num_examples, dict_of_metrics)
        return loss, num_test_samples, metrics


def client_fn(context: Context):
    """Factory to create a YOLODetectionClient for each client."""
    # 1) Create a YOLO model
    yolo_model = create_model("/WehbeDocker/YOLOV8N/pytorchexample/best.pt")

    # 2) Read configs from context
    local_epochs = context.run_config["local-epochs"]
    learning_rate = context.run_config["learning-rate"]
    partition_id = context.node_config["partition-id"]
    seed = context.run_config["seed"] 
    client_state = context.state  

    
    return YOLODetectionClient(
        yolo_model=yolo_model,
        local_epochs=local_epochs,
        client_state=client_state,
        learning_rate=learning_rate,
        partition_id=partition_id,
        seed = seed,
    ).to_client()


# Build the Flower ClientApp
app = ClientApp(client_fn=client_fn)
