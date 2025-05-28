import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import re


from utils_io import load_single_synth_data, load_real_data, load_wireless_data
from utils import cluster_embeddings, compute_clustered_cosine_metrics
from utils_analysis import after_check

def is_top_attention_block(name, model_name=None):
    if model_name == "moirai":
        return name.endswith("self_attn")

    if model_name == "lag_llama":
        return name.endswith(".attn")
    
    return (
        (
            ("SelfAttention" in name or "self_attn" in name)
            and not any(part in name for part in [".q", ".k", ".v", ".o", ".relative_attention_bias"])
        )
        or re.fullmatch(r"gpt2\.h\.\d+", name)
    )


def see_model_architecture(model):
    t5_model = model.model  # ChronosModel
    all_names = [name for name, _ in t5_model.named_modules()]
    print("Model architecture:")
    
    for name in all_names:
        if is_top_attention_block(name, model_name="amazon/chronos-t5-small"):
            print(name)

def capture_attention_outputs(model_name, model, model_input, model_output, pred_length=64, module_list_path=None):
    """
    Capture only the *output* of top-level attention blocks (SelfAttention & EncDecAttention).

    Args:
        model: ChronosPipeline instance.
        model_input: 1D or 2D torch.Tensor.
        module_list_path: Optional file path to save module names.

    Returns:
        A tuple containing:
          - A dictionary of attention block outputs, keyed by module name.
          - The model prediction output.
    """
    if model_name == "amazon/chronos-t5-small" or model_name == "amazon/chronos-bolt-small":
        the_model = model.model  # ChronosModel
    elif model_name == "patchtst":
        the_model = model
    elif model_name == "llm4cp_tdd" or model_name == "llm4cp_fdd":
        the_model = model
    elif model_name == "moirai":
        the_model_run = model
        the_model = the_model_run.module
    elif model_name == "lag_llama":
        predictor, lightning_module = model
        the_model = lightning_module.model
        the_model_run = predictor

        
    attention_outputs = {}

    # Save all module names.
    all_names = [name for name, _ in the_model.named_modules()]
    if module_list_path:
        os.makedirs(os.path.dirname(module_list_path), exist_ok=True)
        with open(module_list_path, "w") as f:
            for name in all_names:
                f.write(name + "\n")

    # Define hook to save outputs.
    def save_attention_output(name):
        def hook(module, inputs, output):
            # Some modules may return a tuple; we extract the first element if so.
            attention_outputs[name] = output[0] if isinstance(output, tuple) else output
        return hook

    # Register hooks for the attention modules.
    for name, module in the_model.named_modules():
        if is_top_attention_block(name, model_name):
            module.register_forward_hook(save_attention_output(name))
    # Ensure input has batch dimension.

    if model_input.dim() == 1:
        model_input = model_input.unsqueeze(0)

    # Trigger a forward pass.
    shaded_for_loop = 1
    if model_name == "amazon/chronos-t5-small" or model_name == "amazon/chronos-bolt-small":
        context_pred = model.predict(model_input, prediction_length=pred_length)
    elif model_name == "patchtst":
        model_input = model_input.unsqueeze(-1).to(model.device)
        multi_context_pred = []
        for shaded in range(shaded_for_loop):
            with torch.no_grad():
                context_pred = model(model_input)['prediction_outputs'].squeeze(-1).unsqueeze(1)  # shape: (batch_size, 1, pred_length)
            multi_context_pred.append(context_pred)
        context_pred = torch.cat(multi_context_pred, dim=1)
    elif model_name == "llm4cp_tdd" or model_name == "llm4cp_fdd":
        model_input = model_input.to(model.device)    
        with torch.no_grad():
            context_pred = model(model_input, None, None, None).unsqueeze(1)
            
    elif model_name == "moirai":
        from utils_moirai import run_moirai_forecast_from_tensors
        multi_context_pred = []
        for shaded in range(shaded_for_loop):
            forecast_array, attention_layers = run_moirai_forecast_from_tensors(
                model_input, model_output, model=the_model_run
            )
            context_pred = torch.tensor(forecast_array, dtype=torch.float32).unsqueeze(1)  # shape: (batch_size, 1, pred_length)
            multi_context_pred.append(context_pred)
        context_pred = torch.cat(multi_context_pred, dim=1)
    elif model_name == "lag_llama":
        from utils_lag_llama import lag_llama_model_running
        multi_context_pred = []
        for shaded in range(shaded_for_loop):
            output = lag_llama_model_running(
                predictor,
                model_input,
                freq="h",
                start_date="2020-01-01",
            ) # should be [100, PRED_LEN]
            context_pred = output.unsqueeze(1)  # shape: (batch_size, 1, pred_length)
            multi_context_pred.append(context_pred)
        context_pred = torch.cat(multi_context_pred, dim=1)
    print(f"Shape of context_pred: {context_pred.shape}")
    return attention_outputs, context_pred


import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import math

def plot_attention_pca(attn_outputs, exp_name, postfix, exp_folder):
    """
    Plot PCA of attention layer outputs from selected encoder layers.

    Args:
        attn_outputs (dict): Attention output tensors keyed by module name.
        exp_name (str): Experiment name identifier.
        postfix (str): Postfix to add to output file names.
        exp_folder (str): Base folder for experiment outputs.
    """
    # selected_keys = [name for name in attn_outputs if is_top_attention_block(name)]
    selected_keys = [name for name in attn_outputs]
    num_blocks = len(selected_keys)

    if num_blocks == 0:
        print("No top-level attention blocks found.")
        return

    cols = 6
    rows = math.ceil(num_blocks / cols)
    fig = plt.figure(figsize=(cols * 3, rows * 3))
    
    pca_shape_str = []

    for idx, key in enumerate(selected_keys):
        data = attn_outputs[key]
        y = data.detach().cpu().to(torch.float32).numpy()
        pca_shape_str_x = f"Shape of {key}: {y.shape}, after stacking: {np.vstack(y).shape}"
        pca_shape_str.append(pca_shape_str_x)
        print(pca_shape_str_x)
        y = np.vstack(y)  # Stack the output sequences

        pca = PCA(n_components=3).fit(y)
        y_transformed = pca.transform(y)
        var_ratio = sum(pca.explained_variance_ratio_)

        # d(0.8)
        epsilon = 0.8
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        d_0_8 = np.searchsorted(cumulative_variance, epsilon) + 1

        ax = fig.add_subplot(rows, cols, idx + 1, projection="3d")
        ax.scatter(
            y_transformed[:, 0], y_transformed[:, 1], y_transformed[:, 2],
            s=0.1, alpha=0.3, marker="."
            # s=3, alpha=0.3, marker="."
        )
        print(f'name: {key}, {len(key.split("."))}, d_0_8: {d_0_8}')
        layer_name = f"{key.split('.')[-4]} {key.split('.')[-1]}" if len(key.split('.')) > 4 else key
        ax.set_title(f"{layer_name}\nVar Ratio {var_ratio:.2f}", fontsize=8)
        ax.tick_params(labelsize=6)

    # write PCA shape string to file
    pca_shape_str = "\n".join(pca_shape_str)
    pca_shape_path = os.path.join(exp_folder, "output_architecture", f"pca_shape_{exp_name}{postfix}.txt")
    with open(pca_shape_path, "w") as f:
        f.write(pca_shape_str)
    print(f"PCA shape summary saved to {pca_shape_path}")
    
    # Save the figure
    plt.tight_layout()
    save_dir = os.path.join(exp_folder, "output_plots")
    os.makedirs(save_dir, exist_ok=True)
    plt_path = os.path.join(save_dir, f"{exp_name}{postfix}_pca.png")
    plt.savefig(plt_path, dpi=300)
    plt.close()
    print(f"PCA plot saved to {plt_path}")
    return d_0_8


def save_attention_summary(attn_outputs, exp_name, postfix, exp_folder):
    """
    Save a text summary of attention output tensor shapes.

    Args:
        attn_outputs (dict): Dictionary of attention outputs.
        exp_name (str): Experiment name identifier.
        postfix (str): Postfix to add to output file names.
        exp_folder (str): Base folder for experiment outputs.
    """
    summary_dir = os.path.join(exp_folder, "output_architecture")
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir, f"focused_layers_{exp_name}{postfix}.txt")
    with open(summary_path, "w") as f:
        f.write(f"\nCaptured {len(attn_outputs)} attention layer outputs.\n")
        f.write("Attention Layer Outputs:\n")
        f.write("=====================================\n")
        for name, tensor in attn_outputs.items():
            f.write(f"{name}: shape = {tensor.shape}\n")
    print(f"Attention outputs summary saved to {summary_path}")


def tensor_handler(tensor, ignore_cluster=False):
    """
    Process a tensor to compute clustering metrics.

    Returns:
        best_k: Optimal number of clusters.
        best_sil: Best silhouette score.
        inter_cos: Inter-cluster cosine metric.
    """
    if tensor.is_cuda:
        tensor = tensor.cpu()
    tensor = tensor.to(torch.float32)
    the_embedding = tensor.numpy()

    best_k, best_sil, embedding_y, embedding_key, embedding_cluster = cluster_embeddings(the_embedding)
    inter_cos, _ = compute_clustered_cosine_metrics(
        embedding_y=embedding_y,
        embedding_cluster=embedding_cluster,
        embedding_key=embedding_key,
        center=True,
        ignore_cluster=ignore_cluster, 
    )
    return best_k, best_sil, inter_cos


def record_attention_metrics(attn_outputs, tensor_handler, nmse, mse, d_0_8, exp_name, postfix, exp_folder):
    """
    Computes and records clustering metrics for attention layer outputs and saves them to CSV.

    Args:
        attn_outputs (dict): Dictionary of layer names to attention tensors.
        tensor_handler (function): Function to extract metrics from a tensor.
        nmse: NMSE error between predicted and target outputs.
        exp_name (str): Experiment name identifier.
        postfix (str): Postfix to add to output file names.
        exp_folder (str): Base folder for experiment outputs.

    Returns:
        pd.DataFrame: DataFrame containing all recorded metrics.
    """
    save_dir = os.path.join(exp_folder, "output_isotropy_metric")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{exp_name}{postfix}.csv")

    results = []
    for name, tensor in attn_outputs.items():
        print(f"Processing {name}: shape = {tensor.shape}")
        best_k, best_sil, inter_cos = tensor_handler(tensor, ignore_cluster=False) # with clustering
        best_k_nocluster, best_sil_nocluster, inter_cos_nocluster = tensor_handler(tensor, ignore_cluster=True) # without clustering

        print(f"  Best K: {best_k}, Silhouette Score: {best_sil:.4f}, Inter Cosine: {inter_cos:.4f}")
        results.append({
            "Layer Name": name,
            "Shape": str(tensor.shape),
            "Best K": best_k,
            "Silhouette Score": best_sil,
            "Inter-Cluster Cosine": inter_cos,
            "NMSE": nmse.item() if isinstance(nmse, torch.Tensor) else nmse,
            "d_0_8": d_0_8,
        })

    df = pd.DataFrame(results)
    df.to_csv(save_path, index=False)
    print(f"Metrics saved to: {save_path}")
    return df

import numpy as np
import matplotlib.pyplot as plt

def plot_forecast_with_uncertainty(args, context_in, context_out, context_pred, n_sample=0, filename="forecast_plot.png"):
    """
    Plots the forecast with uncertainty for a single sample and saves it to a PNG file.

    Parameters:
    - context_in: np.ndarray, shape (n_samples, len_time_series)
    - context_out: np.ndarray, shape (n_samples, len_out_time_series)
    - context_pred: np.ndarray, shape (n_samples, 20, len_out_time_series)
    - n_sample: int, index of the sample to plot
    - filename: str, output filename to save the plot
    """
    y_in = context_in[n_sample].to("cpu").numpy()  # Shape: (len_time_series,)
    y_out = context_out[n_sample].to("cpu").numpy()  # Shape: (len_out_time_series,)
    y_pred_samples = context_pred[n_sample].to("cpu").numpy()  # Shape: (20, len_out_time_series)
    print(f"y_in shape: {y_in.shape}, y_out shape: {y_out.shape}, y_pred_samples shape: {y_pred_samples.shape}")
    # if y_pred_sample ndim > y_out ndim, and y_pred_samples.shape[0] == 1, squeeze it
    # If the input has more than one feature, take the first feature only
    if args.model_name == "llm4cp_tdd" or args.model_name == "llm4cp_fdd":
        if y_in.ndim > 1 and y_in.shape[1] > 1:
            y_in = y_in[:, 0]
        if y_out.ndim > 1 and y_out.shape[1] > 1:
            y_out = y_out[:, 0]
        if y_pred_samples.ndim > 1 and y_pred_samples.shape[1] > 1:
            y_pred_samples = y_pred_samples[:, :, 0]
    print(f"real plotting shape: {y_in.shape}, {y_out.shape}, pred shape: {y_pred_samples.shape}")

    # Median and quantiles
    median_pred = np.median(y_pred_samples, axis=0)
    q10 = np.percentile(y_pred_samples, 10, axis=0)
    q90 = np.percentile(y_pred_samples, 90, axis=0)

    # Time axes
    t_in = np.arange(len(y_in))
    t_out = np.arange(len(y_in), len(y_in) + len(y_out))

    nmse = np.mean((median_pred - y_out) ** 2) / np.mean(y_out ** 2)
    mse = np.mean((median_pred - y_out) ** 2)

    # add the last point of y_in to the first y_out and median_pred
    t_out = np.concatenate((t_in[-1:], t_out))
    y_out = np.concatenate((y_in[-1:], y_out))
    median_pred = np.concatenate((y_in[-1:], median_pred))
    q10 = np.concatenate((y_in[-1:], q10))
    q90 = np.concatenate((y_in[-1:], q90))

    # Plot
    plt.figure(figsize=(10, 3))
    plt.plot(t_in, y_in, label="Ground Truth", color='dodgerblue')
    plt.plot(t_out, y_out, color='dodgerblue')
    plt.plot(t_out, median_pred, label="Median Forecast", color='red')
    # plt.fill_between(t_out, q10, q90, color='red', alpha=0.3, label="80% Interval")
    plt.axvline(x=len(y_in)-1, linestyle='--', color='gray')
    # plt.text(t_out[-1], max(max(q90), max(y_out)) * 0.9, f"MSE: {nmse:.3f}", ha='right')
    # plt.title(f"", fontsize=12, weight='bold')
    plt.title(f"Sample {n_sample} - MSE: {mse:.3f}, NMSE: {nmse:.3f}", fontsize=12)
    plt.legend(loc='upper left', frameon=True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

    # also save the 3 series in a npy file of 3 rows
    np.savez(filename.replace(".png", ""), y_in=y_in, y_out=y_out, median_pred=median_pred, t_in=t_in, t_out=t_out, q10=q10, q90=q90)


def run_experiment(exp_idx, pipeline, context_in, context_out, exp_name, exp_folder, args):
    """
    Runs one full experiment iteration, capturing attention outputs,
    plotting PCA of embeddings, computing NMSE and recording clustering metrics.

    Args:
        exp_idx (int): Experiment index (used for file naming).
        pipeline: ChronosPipeline model.
        context_in (torch.Tensor): Input tensor.
        context_out (torch.Tensor): Target output tensor.
        exp_name (str): Base name for the experiment.
        exp_folder (str): Base folder for experiment outputs.

    Returns:
        pd.DataFrame containing recorded metrics.
    """
    # Create a unique postfix for this experiment.
    postfix = f"_exp{exp_idx:02d}"
    print(f"\n--- Running Experiment {exp_idx}{postfix} ---")

    # Save architecture module names and capture attention outputs + context prediction.
    module_list_path = os.path.join(exp_folder, "output_architecture", f"{exp_name}{postfix}.txt")
    print(f"shape of context_in: {context_in.shape}, shape of context_out: {context_out.shape}")

    if args.model_name == "patchtst":
        context_in = context_in.unsqueeze(-1)
        context_out = context_out.unsqueeze(-1)
        print(f"shape of context_in: {context_in.shape}, shape of context_out: {context_out.shape}")
        trained_model, val_loss, collapsed_attentions, stacked_inputs, stacked_targets, stacked_pred = pipeline(context_in, context_out, train=True)
        context_in = stacked_inputs
        context_pred = stacked_pred
        context_out = stacked_targets.cpu()
        # attn_outputs = collapsed_attentions
        pipeline = trained_model
    elif args.model_name == "moirai":
        from utils_moirai import get_moirai_model
        pipeline = get_moirai_model(
            size="small",
            prediction_length=context_out.shape[1],
            context_length=context_in.shape[1],
        )
    elif args.model_name == "lag_llama":
        from utils_lag_llama import lag_llama_get_model
        predictor, lightning_module = lag_llama_get_model(
            ckpt_path="lag-llama.ckpt",
            context_length=context_in.shape[1],
            prediction_length=context_out.shape[1],
            device=args.device,
            num_samples=1,
            use_rope_scaling=False,
        )
        pipeline = (predictor, lightning_module)

    if args.model_name == 'lag_llama':
        batch_size = 2
    else:
        batch_size = 4
    
    if True:
        num_batches = math.ceil(context_in.shape[0] / batch_size)
        attn_outputs = {}
        context_pred = []
   
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min((batch_idx + 1) * batch_size, context_in.shape[0])
            print(f"Processing batch {batch_idx + 1}/{num_batches}: rows {batch_start} to {batch_end - 1}")

            batch_in = context_in[batch_start:batch_end]
            batch_out = context_out[batch_start:batch_end]
            batch_attn_outputs, batch_context_pred = capture_attention_outputs(
                args.model_name,
                pipeline, batch_in, batch_out,
                pred_length=context_out.shape[1], 
                module_list_path=module_list_path if batch_idx == 0 else None,
            )
            
            for key, value in batch_attn_outputs.items():
                if key in attn_outputs:
                    attn_outputs[key] = torch.cat((attn_outputs[key], value), dim=0)
                else:
                    attn_outputs[key] = value
            context_pred.append(batch_context_pred)

        context_pred = torch.cat(context_pred, dim=0)
    
    
    print(f"Captured {len(attn_outputs)} attention layer outputs.")

    # Save attention outputs summary to file.
    save_attention_summary(attn_outputs, exp_name, postfix, exp_folder)

    # Plot and save PCA visualization.
    d_0_8 = plot_attention_pca(attn_outputs, exp_name, postfix, exp_folder)

    # Compute NMSE using the median of context_pred.
    context_pred_median = torch.median(context_pred, dim=1).values
    print("Shape of the pred tensor:", context_pred.shape, "Shape of the output tensor:", context_out.shape, "Shape of the median tensor:", context_pred_median.shape)
    
    # context_pred_median to cpu
    context_pred_median = context_pred_median.cpu()
    context_out = context_out.cpu()
    nmse = torch.mean((context_pred_median - context_out) ** 2) / torch.mean(context_out ** 2)
    mse = torch.mean((context_pred_median - context_out) ** 2)
    print(f"MSE for experiment {exp_idx}: {mse}, NMSE is {nmse}")

    # plot
    num_samples = 10
    for i in range(num_samples):
        if not os.path.exists(os.path.join(exp_folder, "data_showcase")):
            os.makedirs(os.path.join(exp_folder, "data_showcase"))
        plot_forecast_with_uncertainty(
            args, context_in, context_out, context_pred, n_sample=i,
            filename=os.path.join(exp_folder, f"data_showcase/forecast_plot_{exp_name}{postfix}_sample{i}.png")
        )
    
    # Compute and record clustering metrics.
    df_metrics = record_attention_metrics(attn_outputs, tensor_handler, nmse, mse, d_0_8, exp_name, postfix, exp_folder)
    return df_metrics


def main(args):
    # Load model pipeline.
    print("Loading pipeline...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    if args.model_name == "amazon/chronos-t5-small":
        from chronos import ChronosPipeline
        pipeline = ChronosPipeline.from_pretrained(
            # "amazon/chronos-t5-large",
            args.model_name,
            device_map="cuda",
            torch_dtype=torch.bfloat16,
        )
        if args.see_model:
            see_model_architecture(pipeline)
            return
    elif args.model_name == "amazon/chronos-bolt-small":
        from chronos import BaseChronosPipeline
        pipeline = BaseChronosPipeline.from_pretrained(
            "amazon/chronos-bolt-small",
            device_map="cuda",  # use "cpu" for CPU inference and "mps" for Apple Silicon
            torch_dtype=torch.bfloat16,
        )
    elif args.model_name == "patchtst":
        from utils_model import patchtst_forward
        # val_loss, attention_embeddings = patchtst_forward(input_tensor, output_tensor, train=True)
        pipeline = patchtst_forward
    elif args.model_name == "llm4cp_tdd":
        model_path = "./neurips_wireless_data/tdd_U2U_LLM4CP.pth"
        model = torch.load(model_path, map_location=device).to(device)
        pipeline = model
    elif args.model_name == "llm4cp_fdd":
        model_path = "./neurips_wireless_data/fdd_U2D_LLM4CP.pth"
        model = torch.load(model_path, map_location=device).to(device)
        pipeline = model
    elif args.model_name == "moirai":
        pipeline = None
    elif args.model_name == "lag_llama":
        pipeline = None
    print("Loading synthetic data...")
    file = args.data_file
    

    if args.model_name == 'llm4cp_tdd' or args.model_name == 'llm4cp_fdd':
        dd = args.model_name.split('_')[-1]
        result = load_wireless_data(setup=dd, n_samples=args.num_rows)
        exp_folder = f"{args.result_all_folder}/wireless_{args.model_name.split('/')[-1]}_{args.num_rows}rows"
    elif args.real_data is not None:
        if args.real_data == "standard":
            result = load_real_data(setup="standard", n_samples=args.num_rows)
        else:
            result = load_real_data(setup="harder", n_samples=args.num_rows)
        exp_folder = f"{args.result_all_folder}/real_{args.noised_input}noise_{args.real_data}_{args.model_name.split('/')[-1]}_{args.num_rows}rows"
        
        short = False if args.real_data == "standard" else True
        wireless_tdd = load_wireless_data(setup="tdd", n_samples=args.num_rows, first_dim=True, short=short)
        wireless_fdd = load_wireless_data(setup="fdd", n_samples=args.num_rows, first_dim=True, short=short)
        for k, v in wireless_tdd.items():
            result[k] = v
        for k, v in wireless_fdd.items():
            result[k] = v
        interesting_kernels = [
            'energy_QUN', 'energy_SA', 
            'nature_rain', 'nature_solar',
            'exchange_rate', 'nn5_weekly', 
            'hospital', 'covid', 
            'car_retail', 'dominick',
        ]
        wireless_kernels = list(wireless_tdd.keys()) + list(wireless_fdd.keys())
        interesting_kernels += wireless_kernels
        result = {k: v for k, v in result.items() if k in interesting_kernels}
    else:
        result = load_single_synth_data(file, rows=args.num_rows, input_len=args.input_len, output_len=args.output_len, random_seed=42)
        interesting_kernels = [
            "DotProduct_0", "DotProduct_1",
            "Seasonality_0.5W", "Seasonality_0.25H",
            "RationalQuadratic_1", "RationalQuadratic_10",
            "RBF_0.1", "RBF_1",
            "WhiteKernel_1", "WhiteKernel_0.1"
        ]
        
        # keep only interesting kernels
        result = {k: v for k, v in result.items() if k in interesting_kernels}
        exp_folder = f"{args.result_all_folder}/synth_{args.noised_input}noise_{args.input_len}in_{args.output_len}out_{args.num_rows}rows_{args.model_name.split('/')[-1]}"

    for kernel, (inp, out) in result.items():
        print(f"{kernel}: {inp.shape}, {out.shape}")
        if args.noised_input > 0: # add noise of percent 20 of original scope
            noise = torch.normal(0, args.noised_input * torch.std(inp), size=inp.shape).to(inp.device)
            inp = inp + noise
            result[kernel] = (inp, out)
            print(f"Added noise to input: {inp.shape}")

    # Run a single experiment for each kernel.
    for kernel, (inp, out) in result.items():
        if args.kernel_only and kernel != args.kernel_only:
            continue
        print(f"Running experiment for kernel: {kernel}")
        # Use a fixed random seed for reproducibility.
        single_kernel_folder = os.path.join(exp_folder, kernel)
        os.makedirs(single_kernel_folder, exist_ok=True)
        print("Creating directory:", single_kernel_folder)
        exp_idx = 1
        run_experiment(exp_idx, pipeline, inp, out, kernel, single_kernel_folder, args)
        after_check(single_kernel_folder)

    print("All experiments completed.")
            


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run multiple experiments with Chronos T5.")
    parser.add_argument("--num_experiments", type=int, default=1, help="Number of experiments to run.")
    parser.add_argument("--input_len", type=int, default=500, help="Length of input sequence.")
    parser.add_argument("--output_len", type=int, default=64, help="Length of output sequence.")
    parser.add_argument("--num_rows", type=int, default=1000, help="Number of rows in the dataset.")
    parser.add_argument("--see_model", action="store_true", help="See the model architecture.")
    parser.add_argument("--model_name", type=str, default="amazon/chronos-t5-small", help="Model name.") 
    parser.add_argument("--data_file", type=str, default="../data/single_kernelsynth_100.arrow", help="Path to the data file.")

    parser.add_argument("--kernel_only", type=str, default=None, help="Specify Kernel to run the experiment on.")
    parser.add_argument("--real_data", type=str, default=None, help="Specify Real Data to run the experiment on.")
    parser.add_argument("--noised_input", type=float, default=0, help="Added noise")
    parser.add_argument("--result_all_folder", type=str, default="results_all5", help="Path to the data file.")
    args = parser.parse_args()
    
    main(args)