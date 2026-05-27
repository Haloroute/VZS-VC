# Main training loop
import datasets, datetime, math, os, random, torch
import numpy as np

from dataclasses import asdict
from datasets import DatasetBuilder
from functools import partial
from numpy import ndarray
from tensordict import TensorDict
from torch import Tensor
from torch.optim import AdamW
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm.auto import tqdm

from modules import (
    MeanFlowsAdaptedLoss,
    MeanFlowsGenerator
)
from utils.configs import (
    MeanFlowsGeneratorModuleConfig,
    TrainConfig,
    ValidationConfig,
    VieNeuTTSPreprocessedDatasetConfig
)
from utils.dataset import (
    collate_fn,
    inject_train_data,
    inject_val_data
)
from utils.logger import save_checkpoint
from utils.modules import (
    load_generator,
    load_loss_fn
)


# Train the model from scratch using the preprocessed dataset
def train_model():
    # Load the configurations
    dataset_config = VieNeuTTSPreprocessedDatasetConfig()
    model_config = MeanFlowsGeneratorModuleConfig()
    train_config = TrainConfig()
    validation_config = ValidationConfig()

    # Load the preprocessed dataset using the specified configuration
    dataset = datasets.load_dataset(
        dataset_config.path,
        streaming=True
    )
    train_dataset, val_dataset = dataset[dataset_config.train_split].with_format("torch"), dataset[dataset_config.val_split].with_format("torch")
    # train_dataset = train_dataset.shuffle(seed=dataset_config.seed)
    # val_dataset = val_dataset.shuffle(seed=dataset_config.seed)
    print("Preprocessed dataset loaded and shuffled successfully.")

    # Get the number of training and validation samples
    dataset_builder: DatasetBuilder = datasets.load_dataset_builder(dataset_config.path)
    n_train_samples: int = dataset_builder.info.splits[dataset_config.train_split].num_examples
    n_val_samples: int = dataset_builder.info.splits[dataset_config.val_split].num_examples
    print(f"Number of training samples: {n_train_samples}")
    print(f"Number of validation samples: {n_val_samples}")

    # Create DataLoaders for training and validation
    n_workers = os.cpu_count() - 1 if os.cpu_count() is not None else 0 # Use all available CPU cores minus one for DataLoader workers
    collate_fn_wrapper = partial(collate_fn, config=dataset_config)
    train_loader = StatefulDataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        num_workers=n_workers,
        collate_fn=collate_fn_wrapper,
        pin_memory=True
    )
    val_loader = StatefulDataLoader(
        val_dataset,
        batch_size=validation_config.batch_size,
        num_workers=n_workers,
        collate_fn=collate_fn_wrapper,
        pin_memory=True
    )
    print("DataLoaders created successfully.")

    # Create the model and load the pretrained modules
    model: MeanFlowsGenerator = load_generator(train_config.device)
    loss_fn: MeanFlowsAdaptedLoss = load_loss_fn(train_config.device)
    model = torch.compile(model, dynamic=True)
    loss_fn = torch.compile(loss_fn, dynamic=True)
    print("Model and loss function loaded successfully.")

    # Setup the optimizer, EMA model, random seed, and other training components
    optimizer = AdamW(
        model.parameters(),
        lr=train_config.lr,
        betas=train_config.beta,
        weight_decay=train_config.weight_decay
    )
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(decay=train_config.ema_decay))
    torch.manual_seed(train_config.seed)
    np.random.seed(train_config.seed)
    random.seed(train_config.seed)

    print("Optimizer and EMA model set up successfully.")

    # Training loop
    print("Starting training loop...")
    print(f"Model configuration: {asdict(model_config)}")
    print(f"Training configuration: {asdict(train_config)}")

    # Iterate over epochs
    for epoch in range(1, train_config.n_epochs + 1):
        # Training phase
        print(f"Epoch {epoch}/{train_config.n_epochs}")
        model.train()
        loss_fn.train()
        train_loss = 0.0

        # Iterate over the training DataLoader with a progress bar
        try:
            for i, batch in (t := tqdm(enumerate(train_loader), desc="Training", total=math.ceil(n_train_samples / train_config.batch_size), leave=False)):
                # Move the batch to the specified device (GPU or CPU)
                batch: TensorDict = batch.to(train_config.device, non_blocking=True)
                # batch is a TensorDict containing:
                # "target": pre_vq_padded, # (N, T, D_codec)
                # "content": content_padded, # (N, T_content, D_content)
                # "pitch": pitch_padded, # (N, T_pitch)
                # "amplitude": amplitude_padded, # (N, T_amplitude)
                # "timbre": acoustic_padded, # (N, T_timbre, D_timbre)
                # "target_length": pre_vq_length, # (N,)
                # "content_length": content_length, # (N,)
                # "pitch_length": pitch_length, # (N,)
                # "amplitude_length": amplitude_length, # (N,)
                # "timbre_length": acoustic_length # (N,)

                # Zero the gradients
                optimizer.zero_grad()

                # Use autocast for mixed precision training if enabled in the configuration
                with torch.amp.autocast(device_type=train_config.device, dtype=torch.bfloat16, enabled=train_config.amp_enable):
                    # Update the batch with training data such as sampled r and t timesteps and conditioning dropout
                    batch = inject_train_data(batch, train_config)

                    # Forward pass and loss computation
                    # loss = loss_fn(model, **batch)
                    loss = loss_fn(
                        model=model,
                        target=batch['target'],
                        epsilon=batch['epsilon'],
                        content=batch['content'],
                        pitch=batch['pitch'],
                        amplitude=batch['amplitude'],
                        timbre=batch['timbre'],
                        start_time=batch['start_time'],
                        end_time=batch['end_time'],
                        target_length=batch['target_length'],
                        content_length=batch['content_length'],
                        pitch_length=batch['pitch_length'],
                        amplitude_length=batch['amplitude_length'],
                        timbre_length=batch['timbre_length'],
                        drop_cond=batch['drop_cond']
                    )

                # Backpropagation and optimization step
                loss.backward()
                optimizer.step()
                ema_model.update_parameters(model)

                # Accumulate the total loss for this epoch (multiply by batch size to get the sum of losses for all samples in the batch)
                train_loss += loss.item() * train_config.batch_size

                # Set the description of the progress bar to show the current average loss
                t.set_postfix({
                    "loss": train_loss / ((i + 1) * train_config.batch_size)
                })

        except Exception as e:
            print(f"An error occurred during training: {e}")
            print("Saving checkpoint before exiting...")

            # Get the current time and format it as YYMMDD-HHMMSS for the checkpoint filename
            timestamp = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
            checkpoint_filename = f"checkpoint_epoch_{epoch}_step_{i}_error_{timestamp}.pth"
            checkpoint_path = os.path.join(train_config.checkpoint_folder, checkpoint_filename)
            save_checkpoint(
                checkpoint_path,
                model, optimizer, train_loader, epoch,
                step=i, loss=train_loss / ((i + 1) * train_config.batch_size)
            )

            print(f"Checkpoint saved for epoch {epoch} at step {i}.")
            raise e

        # Calculate and print the average training loss for this epoch
        avg_train_loss = train_loss / n_train_samples
        print(f"Average training loss: {avg_train_loss:.4f}")

        # Validation phase (optional, can be done every few epochs to save time)
        if epoch % validation_config.validate_every_n_epochs == 0: # Validate every few epochs
            model.eval()
            loss_fn.eval()
            val_loss = 0.0

            # Iterate over the validation DataLoader with a progress bar
            with torch.inference_mode():
                for i, batch in tqdm(enumerate(val_loader), desc="Validation", total=math.ceil(n_val_samples / validation_config.batch_size), leave=False):
                    # Move the batch to the specified device (GPU or CPU)
                    batch: TensorDict = batch.to(validation_config.device, non_blocking=True)

                    # Use autocast for mixed precision validation if enabled in the configuration
                    with torch.amp.autocast(device_type=validation_config.device, dtype=torch.bfloat16, enabled=validation_config.amp_enable):
                        # Update the batch with validation data such as constant r and t timesteps and no conditioning dropout
                        batch = inject_val_data(batch)

                        # Forward pass and loss computation
                        # loss = loss_fn(model, **batch)
                        loss = loss_fn(
                            model=model,
                            target=batch['target'],
                            epsilon=batch['epsilon'],
                            content=batch['content'],
                            pitch=batch['pitch'],
                            amplitude=batch['amplitude'],
                            timbre=batch['timbre'],
                            start_time=batch['start_time'],
                            end_time=batch['end_time'],
                            target_length=batch['target_length'],
                            content_length=batch['content_length'],
                            pitch_length=batch['pitch_length'],
                            amplitude_length=batch['amplitude_length'],
                            timbre_length=batch['timbre_length'],
                            drop_cond=batch['drop_cond']
                        )

                    # Accumulate the total loss for this validation epoch (multiply by batch size to get the sum of losses for all samples in the batch)
                    val_loss += loss.item() * validation_config.batch_size

            # Calculate and print the average validation loss for this epoch
            avg_val_loss = val_loss / n_val_samples
            print(f"Average validation loss: {avg_val_loss:.4f}")

        # Saving phase (save a checkpoint every few epochs)
        if epoch % train_config.save_every_n_epochs == 0:
            save_checkpoint(
                os.path.join(train_config.checkpoint_folder, f"checkpoint_epoch_{epoch}.pth"),
                model, optimizer, train_loader, epoch, 
                loss=avg_train_loss
            )
            print(f"Checkpoint saved for epoch {epoch}.")


def main():
    print("This is the main training script. You can run specific training functions from here if needed.")
    print("Choose an action:")
    print("1. Train a model from scratch")
    print("2. Continue training an existing model")
    # choice = input("Enter the number of your choice: ")
    choice = "1" # For now, we will just train from scratch. You can uncomment the input line above to enable user input for choosing the action.
    if choice == "1":
        print("You chose to train a model from scratch.")
        train_model()
    elif choice == "2":
        print("You chose to continue training an existing model.")
        raise NotImplementedError("Continue training functionality is not implemented yet.")
        # continue_training()
    else:
        print("Invalid choice. Please run the script again and choose a valid option.")

if __name__ == "__main__":
    main()