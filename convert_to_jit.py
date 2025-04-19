


import torch
from config import log_path, testing_path, weights_path, dataset_name, result_path, converted_weights_path
from misc import check_mkdir, crf_refine
from device_manager import DeviceManager

# Change "pmd" to the appropriate version when running experiments with other models.
# By default, pmd refers to our best-performing unpruned model.
from pmd import PMDLite
import logging
from datetime import datetime

# Check for available devices and set the device accordingly.
device_manager = DeviceManager()
device_ = device_manager.get_device()

# Configure logging
logging.basicConfig(
    filename=log_path + f'model_export_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
    )

def export_model_to_jit():
    model = PMDLite().to(device_)

    # Load model weights and biases.
    checkpoint = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(checkpoint, strict=False)
    model = model.to(device_)
    model.eval()

    # Trace the model with a random input tensor.   
    example_input = torch.rand(1, 3, 480, 864).to(device_)
    traced_model = torch.jit.trace(model, example_input)

    logging.info("Traced model")
    logging.info(traced_model)
    
    logging.info("Saving model")
    traced_model.save(converted_weights_path)
    logging.info("Running model.forward")
    output = traced_model.forward(example_input)
    logging.info("Traced output: %s", output)
    logging.info("Model graph: %s", traced_model.graph)
    logging.info("Model code: %s", traced_model.code)

export_model_to_jit()
