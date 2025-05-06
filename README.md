# Federated_YOLO_training-
This is a repository for doing simulations of training YOLO v8 model in a federated way.

Start by cloning the example project:
├── pytorchexample
│   ├── __init__.py
│   ├── client_app.py   # Defines your ClientApp
│   ├── server_app.py   # Defines your ServerApp
│   └── task.py         # Defines your model, training, testing....
├── pyproject.toml      # Project metadata like dependencies and configs
└── README.md

Install the dependencies defined in `pyproject.toml` as well as the `pytorchexample` package.

You have to prepare the datasets folders as explained in 'https://docs.ultralytics.com/datasets/detect/'
This example might run faster when the `ClientApp`s have access to a GPU. If your system has one, you can make use of it by configuring the `backend.client-resources` component in `pyproject.toml`. If you want to try running the example with GPU right away, use the `local-simulation-gpu`

### In order to run the Simulation Engine, use:
flwr run .

###You can change the configuration setting like:
flwr run . --run-config "num-server-rounds=5 learning-rate=0.05"
