import sys
from types import ModuleType
import torch
import numpy as np
import pandas as pd
from gluonts.dataset.common import ListDataset
from lag_llama.gluon.estimator import LagLlamaEstimator

# Create dummy gluonts.torch.modules.loss if missing
MODULE_PATH = 'gluonts.torch.modules.loss'
def _create_dummy_module(path):
    parts = path.split('.')
    parent = None
    current = ''
    for part in parts:
        current = f"{current}.{part}" if current else part
        if current not in sys.modules:
            module = ModuleType(current)
            sys.modules[current] = module
            if parent:
                setattr(sys.modules[parent], part, module)
        parent = current
    return sys.modules[path]

_dummy = _create_dummy_module(MODULE_PATH)

# Define stub loss classes to satisfy checkpoint loading
class DistributionLoss:
    def __init__(self, *args, **kwargs): pass
    def __call__(self, *args, **kwargs): return 0.0
    def __getattr__(self, name): return lambda *args, **kwargs: None

class NegativeLogLikelihood(DistributionLoss):
    pass

setattr(_dummy, 'DistributionLoss', DistributionLoss)
setattr(_dummy, 'NegativeLogLikelihood', NegativeLogLikelihood)


def lag_llama_get_model(
    ckpt_path: str,
    context_length: int,
    prediction_length: int,
    device: torch.device,
    num_samples: int,
    use_rope_scaling: bool = False
) -> object:
    """
    Load a Lag-Llama predictor from checkpoint.

    Returns:
        predictor: a GluonTS predictor ready for .predict()
    """
    # 1) load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_kwargs = ckpt["hyper_parameters"]["model_kwargs"]

    # 2) configure rope scaling if requested
    rope_scaling = None
    if use_rope_scaling:
        factor = max(1.0, (context_length + prediction_length) / model_kwargs["context_length"])
        rope_scaling = {"type": "linear", "factor": factor}

    # 3) build estimator
    estimator = LagLlamaEstimator(
        ckpt_path=ckpt_path,
        prediction_length=prediction_length,
        context_length=context_length,
        input_size=model_kwargs["input_size"],
        n_layer=model_kwargs["n_layer"],
        n_embd_per_head=model_kwargs["n_embd_per_head"],
        n_head=model_kwargs["n_head"],
        scaling=model_kwargs["scaling"],
        time_feat=model_kwargs["time_feat"],
        rope_scaling=rope_scaling,
        batch_size=1,
        num_parallel_samples=num_samples,
        device=device,
    )

    # 4) create components
    transformation = estimator.create_transformation()
    lightning_module = estimator.create_lightning_module()
    predictor = estimator.create_predictor(transformation, lightning_module)
    # import pdb; pdb.set_trace()
    # [name for name, _ in lightning_module.model.named_modules() if name.endswith('.attn')]
    return predictor, lightning_module


def lag_llama_model_running(
    predictor: object,
    input_tensor: torch.Tensor,
    freq: str = "h",
    start_date: str = "2020-01-01"
) -> torch.Tensor:
    """
    Run the predictor on a batch of raw tensors.

    Args:
        predictor: GluonTS predictor from get_model()
        input_tensor: tensor of shape (batch_size, context_length)
        freq: e.g. "H" for hourly
        start_date: timestamp string for the first index of each series

    Returns:
        output_tensor: tensor of shape (batch_size, prediction_length)
    """
    # 1) build ListDataset
    list_data = [
        {"start": pd.Timestamp(start_date), "target": row.cpu().numpy()}
        for row in input_tensor
    ]
    dataset = ListDataset(list_data, freq=freq)

    # 2) predict
    forecasts = list(predictor.predict(dataset))

    # 3) collect samples and median
    samples = np.stack([f.samples for f in forecasts], axis=0)
    point_preds = np.median(samples, axis=1)

    # 4) to torch
    return torch.from_numpy(point_preds).to(input_tensor.device)


# Example usage
if __name__ == "__main__":
    DEVICE = torch.device("cuda:0")  # or "cpu"
    CONTEXT_LEN = 64
    PRED_LEN = 16
    NUM_SAMPLES = 100

    # load predictor
    predictor, lightning_module = lag_llama_get_model(
        ckpt_path="lag-llama.ckpt",
        context_length=CONTEXT_LEN,
        prediction_length=PRED_LEN,
        device=DEVICE,
        num_samples=NUM_SAMPLES,
        use_rope_scaling=False,
    )

    # fake input batch
    input_batch = torch.randn(100, CONTEXT_LEN, device=DEVICE)
    # run
    output = lag_llama_model_running(
        predictor,
        input_batch,
        freq="H",
        start_date="2020-01-01",
    ) # should be [100, PRED_LEN]

    print("Output shape:", output.shape)  # should be [100, PRED_LEN]

    import pdb; pdb.set_trace()