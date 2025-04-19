import torch

class DeviceManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DeviceManager, cls).__new__(cls)
            cls._instance._initialize_device()
        return cls._instance

    def _initialize_device(self):
        if torch.cuda.is_available():
            # Change this to the device ordinal of the GPU
            # If the device is cuda:x, device_ids should be [x].
            device_ids = [0]
            torch.cuda.set_device(device_ids[0])
            self.device = torch.device("cuda")
            #torch.cuda.set_device(0)  # Set to the first CUDA device
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")  # For Apple Silicon (Metal Performance Shaders)
        else:
            self.device = torch.device("cpu")  # Fallback to CPU

    def get_device(self):
        return self.device
