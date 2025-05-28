import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from transformers import PatchTSTConfig, PatchTSTForPrediction

def patchtst_forward(input_tensor, output_tensor=None, train=True, epochs=10, batch_size=32, learning_rate=1e-4):
    """
    Processes input through the PatchTST model.

    Args:
        input_tensor (torch.Tensor): Input tensor of shape (batch_size, sequence_length, num_channels).
        output_tensor (torch.Tensor, optional): Target tensor of shape (batch_size, prediction_length, num_channels).
        train (bool): Flag indicating training mode.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for the optimizer.

    Returns:
        tuple: (loss, attention_embeddings)
            - loss (float or None): Computed loss if in training mode; otherwise, None.
            - attention_embeddings (tuple of torch.Tensor): Attention embeddings from the model.
    """
    # Define model configuration
    patch_len = 16 if input_tensor.shape[1] > 16 else input_tensor.shape[1] // 2
    config = PatchTSTConfig(
        num_input_channels=input_tensor.shape[2],
        context_length=input_tensor.shape[1],
        prediction_length=output_tensor.shape[1] if output_tensor is not None else 32,
        patch_length=patch_len,
        stride=8,
        dropout=0.1,
        num_attention_heads=4,
        hidden_size=32,
        num_hidden_layers=2
    )

    # Initialize the model
    model = PatchTSTForPrediction(config)

    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if train:
        # Prepare dataset
        dataset = TensorDataset(input_tensor, output_tensor)
        train_size = int(0.5 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        model.train()
        for epoch in range(epochs):
            for batch_inputs, batch_targets in train_loader:
                batch_inputs = batch_inputs.to(device)
                batch_targets = batch_targets.to(device)
                # import pdb; pdb.set_trace()
                optimizer.zero_grad()
                outputs = model(past_values=batch_inputs, future_values=batch_targets, output_attentions=True)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

        # Evaluate on validation set
        model.eval()
        val_loss = 0.0

        # Initialize containers
        all_attentions = {}
        all_inputs = []
        all_targets = []
        all_preds = []

        with torch.no_grad():
            for batch_inputs, batch_targets in val_loader:
                batch_inputs = batch_inputs.to(device)
                batch_targets = batch_targets.to(device)

                outputs = model(past_values=batch_inputs, future_values=batch_targets, output_attentions=True)
                val_loss += outputs.loss.item() * batch_inputs.size(0)
                
                # Collect attention from each layer
                for i, layer_attn in enumerate(outputs.attentions):
                    key = f"layer.{i+1}"
                    if key not in all_attentions:
                        all_attentions[key] = []
                    all_attentions[key].append(layer_attn.cpu())

                # Collect ground truth and predictions
                all_inputs.append(batch_inputs.cpu())
                all_targets.append(batch_targets.cpu())
                all_preds.append(outputs.prediction_outputs.cpu())  # shape: (batch_size, prediction_length, 1)

        val_loss /= len(val_dataset)

        # Stack everything
        stacked_attentions = {key: torch.cat(tensors, dim=0) for key, tensors in all_attentions.items()}
        collapsed_attentions = {
            key: tensor.reshape(tensor.shape[0], tensor.shape[1], -1) for key, tensor in stacked_attentions.items()
        }
        stacked_inputs = torch.cat(all_inputs, dim=0).squeeze(-1)              # (N, sequence_length)
        stacked_targets = torch.cat(all_targets, dim=0).squeeze(-1)              # (N, prediction_length)
        stacked_pred = torch.cat(all_preds, dim=0).squeeze(-1).unsqueeze(1)      # (N, 1, prediction_length)

        # Final return
        return model, val_loss, collapsed_attentions, stacked_inputs, stacked_targets, stacked_pred

    else:
        # Inference mode
        model.eval()
        input_tensor = input_tensor.to(device)
        with torch.no_grad():
            outputs = model(past_values=input_tensor, output_attentions=True)
            predictions = outputs.prediction_outputs
            attention_embeddings = outputs.attentions
        
        return predictions, attention_embeddings


if __name__ == "__main__":
    # Example usage
    # Assuming input_tensor is of shape (batch_size, sequence_length, num_channels)
    # and output_tensor is of shape (batch_size, prediction_length, num_channels)
    
    # Create example tensors
    import torch

    # Example input tensor of shape (1000, 100, 1)
    input_tensor = torch.randn(1000, 100, 1)

    # Example output tensor of shape (1000, 32, 1)
    output_tensor = torch.randn(1000, 32, 1)

    # Training mode
    val_loss, attention_embeddings = patchtst_forward(input_tensor, output_tensor, train=True)

    # Inference mode
    predictions, attention_embeddings = patchtst_forward(input_tensor, train=False)
    import pdb; pdb.set_trace()