import torch
import numpy as np
import pandas as pd
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

def get_moirai_model(
    size="small",
    prediction_length=20,
    context_length=200,
    patch_size="auto",
    num_samples=100,
    target_dim=1,
    feat_dynamic_real_dim=0,
    past_feat_dynamic_real_dim=0,
):
    """
    Returns a MoiraiForecast model initialized with pretrained weights.
    """
    moirai_module = MoiraiModule.from_pretrained(f"Salesforce/moirai-1.0-R-{size}")

    model = MoiraiForecast(
        module=moirai_module,
        prediction_length=prediction_length,
        context_length=context_length,
        patch_size=patch_size,
        num_samples=num_samples,
        target_dim=target_dim,
        feat_dynamic_real_dim=feat_dynamic_real_dim,
        past_feat_dynamic_real_dim=past_feat_dynamic_real_dim,
    )

    return model


def run_moirai_forecast_from_tensors(
    in_tensor: torch.Tensor,
    out_tensor: torch.Tensor,
    model: MoiraiForecast,
    batch_size=32,
    freq="D"
):
    """
    Runs the Moirai model on preprocessed tensors using GluonTS wrapping.
    """
    in_array = in_tensor.numpy()
    out_array = out_tensor.numpy()
    batch_size, in_len = in_array.shape
    _, out_len = out_array.shape

    # Build pandas-based dataset
    dataframes = []
    for i in range(batch_size):
        start_date = pd.Timestamp("2000-01-01") + pd.Timedelta(days=i)
        dates = pd.date_range(start=start_date, periods=in_len + out_len, freq=freq)
        full_series = np.concatenate([in_array[i], out_array[i]])
        df = pd.DataFrame({"target": full_series}, index=dates)
        dataframes.append(df)

    ds = PandasDataset(dataframes)
    train, test_template = split(ds, offset=-out_len)

    test_data = test_template.generate_instances(
        prediction_length=out_len,
        windows=1,
        distance=out_len
    )

    predictor = model.create_predictor(batch_size=batch_size)
    forecasts = list(predictor.predict(test_data.input))
    inputs = list(test_data.input)
    labels = list(test_data.label)

    def pad_and_stack(arrays, pad_value=0.0):
        max_len = max(len(arr) for arr in arrays)
        result = np.full((len(arrays), max_len), pad_value, dtype=np.float32)
        for i, arr in enumerate(arrays):
            result[i, :len(arr)] = arr
        return result

    inp_array = pad_and_stack([inp["target"] for inp in inputs])
    label_array = pad_and_stack([label["target"] for label in labels])
    forecast_array = pad_and_stack([forecast["mean"] for forecast in forecasts])

    attention_layer_names = [
        name for name, _ in model.module.named_modules()
        if name.endswith("self_attn")
    ]

    return forecast_array, attention_layer_names