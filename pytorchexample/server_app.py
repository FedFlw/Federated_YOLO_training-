"""pytorchexample: A Flower / PyTorch YOLO detection server."""

import os
import time
import torch
from typing import List, Tuple
from flwr.common import Context, Metrics, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from copy import deepcopy

from pytorchexample.task import create_model, get_weights, set_weights


def map50_weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Weighted average aggregator for 'map50'. 
    Each tuple is (num_examples, {"map50": value, ...}).
    """
    total_map50 = 0.0
    total_examples = 0
    for num_examples, m in metrics:
        if "map50" in m:
            total_map50 += num_examples * m["map50"]
            total_examples += num_examples

    if total_examples == 0:
        return {"map50": 0.0}
    return {"map50": total_map50 / total_examples}


def server_fn(context: Context):
    """Construct components defining server behavior."""
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    fraction_eval = context.run_config["fraction-evaluate"]
    server_device = context.run_config["server-device"]

    start_time = time.time()
    yolo_model = create_model("/WehbeDocker/YOLOV8N/pytorchexample/best.pt")
    init_weights = get_weights(yolo_model)  
    parameters = ndarrays_to_parameters(init_weights)

    def on_conclude(server_round, global_parameters, client_manager):
        final_arrays = parameters_to_ndarrays(global_parameters)
        yolo_model = create_model("/WehbeDocker/YOLOV8N/pytorchexample/best.pt")  
        set_weights(yolo_model, final_arrays)
        torch.save(yolo_model.model.state_dict(), "final_agg_model.pt")
        print("Saved final aggregated model to 'final_agg_model.pt'")

    
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_eval,
        min_available_clients=1,
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=map50_weighted_average,
    )
    strategy.on_conclude = on_conclude

    
    config = ServerConfig(num_rounds=num_rounds)

   
    server_app_components = ServerAppComponents(strategy=strategy, config=config)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total simulation time: {elapsed_time / 60:.2f} minutes ({elapsed_time:.2f} seconds)")

    return server_app_components


# Create and run the ServerApp
app = ServerApp(server_fn=server_fn)
