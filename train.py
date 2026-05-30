# Main training loop
import datasets, datetime, math, os, random, torch, wandb

import numpy as np
import torch.nn as nn

from dataclasses import asdict
from datasets import DatasetBuilder
from functools import partial
from tensordict import TensorDict
from torch import Tensor
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from modules import VoiceGenerator
from utils.configs import (
    TrainConfig,
    ValidationConfig,
    VieNeuTTSPreprocessedDatasetConfig,
    VoiceGeneratorModuleConfig
)
from utils.dataset import collate_fn
from utils.logger import load_checkpoint, save_checkpoint
from utils.metrics import calculate_accuracy
from utils.modules import load_generator


# Train the model from scratch using the preprocessed dataset
def train_model(checkpoint_path: str | None = None, previous_run_id: str | None = None):
    # Load the configurations
    dataset_config = VieNeuTTSPreprocessedDatasetConfig()
    model_config = VoiceGeneratorModuleConfig()
    train_config = TrainConfig()
    validation_config = ValidationConfig()

    # Setup WandB run to track this script.
    if previous_run_id is not None:
        print(f"Continuing from previous wandb run ID: {previous_run_id}")
        run = wandb.init(
            # Set the wandb entity where your project will be logged (generally your team name).
            entity="topaz-and-numpy",
            # Set the wandb project where this run will be logged.
            project="VZS-VC",
            # Set the wandb run ID to continue logging to the same run.
            id=previous_run_id,
            # Track hyperparameters and run metadata.
            config={
                "model_config": asdict(model_config),
                "train_config": asdict(train_config),
                "validation_config": asdict(validation_config),
                "dataset_config": asdict(dataset_config)
            },
            resume="must"
        )
    else:
        run = wandb.init(
            # Set the wandb entity where your project will be logged (generally your team name).
            entity="topaz-and-numpy",
            # Set the wandb project where this run will be logged.
            project="VZS-VC",
            # Track hyperparameters and run metadata.
            config={
                "model_config": asdict(model_config),
                "train_config": asdict(train_config),
                "validation_config": asdict(validation_config),
                "dataset_config": asdict(dataset_config)
            }
        )
    run.define_metric("epoch")
    run.define_metric("train/loss", step_metric="epoch")
    run.define_metric("train/accuracy", step_metric="epoch")
    run.define_metric("val/loss", step_metric="epoch")
    run.define_metric("val/accuracy", step_metric="epoch")

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
    collate_fn_wrapper = partial(collate_fn, config=dataset_config)
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        num_workers=train_config.n_workers,
        collate_fn=collate_fn_wrapper,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=validation_config.batch_size,
        num_workers=validation_config.n_workers,
        collate_fn=collate_fn_wrapper,
        pin_memory=True
    )
    print("DataLoaders created successfully.")

    # Create the model and load the pretrained modules
    model: VoiceGenerator = load_generator(train_config.device)
    loss_fn: nn.CrossEntropyLoss = nn.CrossEntropyLoss()
    model = torch.compile(model, dynamic=True, fullgraph=True)
    print("Model and loss function loaded successfully.")
    print(f"Model parameters count: {sum(p.numel() for p in model.parameters())}")

    # Setup the optimizer, EMA model, random seed, and other training components
    optimizer = AdamW(
        model.parameters(),
        lr=train_config.lr,
        betas=train_config.beta,
        weight_decay=train_config.weight_decay
    )
    scheduler = LinearLR(
        optimizer,
        start_factor=train_config.start_factor,
        total_iters=train_config.n_warmup_epochs
    )
    scaler = GradScaler(device=train_config.device, enabled=train_config.amp == "fp16") # Use GradScaler for mixed precision training (fp16) and disable it for bf16 or fp32
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(decay=train_config.ema_decay))
    torch.manual_seed(train_config.seed)
    np.random.seed(train_config.seed)
    random.seed(train_config.seed)

    print("Optimizer and EMA model set up successfully.")

    # Load checkpoint if a path is provided
    if checkpoint_path is not None:
        print(f"Loading checkpoint from {checkpoint_path}...")
        _, start_epoch, start_loss, start_accuracy = load_checkpoint(
            checkpoint_path,
            model, ema_model, optimizer, scheduler, scaler
        )
        print(f"Checkpoint loaded successfully. Resuming from epoch {start_epoch}, loss {start_loss}, accuracy {start_accuracy}%.")

        # Update the learning rate of the optimizer
        for param_group in optimizer.param_groups:
            if 'initial_lr' in param_group and 'lr' in param_group:
                param_group['lr'] *= train_config.lr / (param_group['initial_lr'] + 1e-8)
                param_group['initial_lr'] = train_config.lr

        # Update the learning rate of the scheduler
        if hasattr(scheduler, 'base_lrs'):
            scheduler.base_lrs = [train_config.lr for _ in scheduler.base_lrs]

    else:
        start_epoch = 0
        print("No checkpoint provided. Starting training from scratch.")

    # Training loop
    print("Starting training loop...")
    print(f"Model configuration: {asdict(model_config)}")
    print(f"Training configuration: {asdict(train_config)}")

    # Iterate over epochs
    for epoch in range(start_epoch + 1, train_config.n_epochs + 1):
        # Training phase
        print(f"Epoch {epoch}/{train_config.n_epochs}")
        model.train()
        ema_model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        # Iterate over the training DataLoader with a progress bar
        try:
            for i, batch in (t := tqdm(enumerate(train_loader), desc="Training", total=math.ceil(n_train_samples / train_config.batch_size), leave=False)):
                # Move the batch to the specified device (GPU or CPU)
                batch: TensorDict = batch.to(train_config.device, non_blocking=True)
                # batch is a TensorDict containing:
                # "content": content_padded, # (N, T_content, D_content)
                # "pitch": pitch_padded, # (N, T_pitch)
                # "amplitude": amplitude_padded, # (N, T_amplitude)
                # "timbre": acoustic_padded, # (N, T_timbre, D_timbre)
                # "target": pre_vq_padded, # (N, T, D_codec)

                # "content_length": content_length, # (N,)
                # "pitch_length": pitch_length, # (N,)
                # "amplitude_length": amplitude_length, # (N,)
                # "timbre_length": acoustic_length # (N,)
                # "target_length": pre_vq_length, # (N,)

                # Zero the gradients
                optimizer.zero_grad()

                # Use autocast for mixed precision training if enabled in the configuration (fp16/bf16) and disable it for fp32
                with autocast(device_type=train_config.device, dtype=torch.float16, enabled=train_config.amp != "fp32"):
                    # Forward pass and loss computation
                    output: Tensor = model(
                        content=batch['content'],
                        pitch=batch['pitch'],
                        amplitude=batch['amplitude'],
                        timbre=batch['timbre'],

                        content_length=batch['content_length'],
                        pitch_length=batch['pitch_length'],
                        amplitude_length=batch['amplitude_length'],
                        timbre_length=batch['timbre_length'],

                        target=batch['target'],
                        target_length=batch['target_length']
                    ) # (N, N_bins, T, D_codec)
                    loss: Tensor = loss_fn(output, batch['target']) # CrossEntropyLoss expects (N, N_bins, T, D_codec) for the input and (N, T, D_codec) for the target
                    n_correct, n_total = calculate_accuracy(output, batch['target'])

                # Backpropagation and optimization step
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient clipping to prevent exploding gradients
                scaler.step(optimizer)
                scaler.update() 
                ema_model.update_parameters(model)

                # Accumulate the total loss for this epoch (multiply by batch size to get the sum of losses for all samples in the batch)
                train_loss += loss.item() * train_config.batch_size
                train_correct += n_correct
                train_total += n_total

                # Set the description of the progress bar to show the current average loss
                t.set_postfix({
                    "loss": f"{loss.item():.5f}",
                    "acc": f"{n_correct / (n_total + 1e-8) * 100:.3f}%",
                    "avg_loss": f"{train_loss / ((i + 1) * train_config.batch_size + 1e-8):.5f}",
                    "avg_acc": f"{train_correct / (train_total + 1e-8) * 100:.3f}%"
                })

        except Exception as e:
            print(f"An error occurred during training: {e}")
            print("Saving checkpoint before exiting...")

            # Get the current time and format it as YYMMDD-HHMMSS for the checkpoint filename
            timestamp = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
            checkpoint_filename = f"checkpoint_e_{epoch:03d}_s_{i:05d}_error_{timestamp}.pth"
            checkpoint_path = os.path.join(train_config.checkpoint_folder, checkpoint_filename)
            save_checkpoint(
                checkpoint_path,
                model, ema_model, optimizer, scheduler, scaler, epoch,
                loss=train_loss / ((i + 1) * train_config.batch_size),
                accuracy=train_correct / (train_total + 1e-8) * 100
            )
            run.save(checkpoint_path)

            print(f"Checkpoint saved for epoch {epoch} at step {i}.")
            raise e

        # Calculate and print the average training loss for this epoch
        avg_train_loss = train_loss / (n_train_samples + 1e-8)
        avg_train_acc = train_correct / (train_total + 1e-8) * 100
        print(f"Average training loss: {avg_train_loss:.5f}")
        print(f"Average training accuracy: {avg_train_acc:.3f}%")
        run.log({"epoch": epoch, "train/loss": avg_train_loss, "train/accuracy": avg_train_acc})

        # Validation phase (optional, can be done every few epochs to save time)
        if epoch % validation_config.validate_every_n_epochs == 0: # Validate every few epochs
            ema_model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0

            # Iterate over the validation DataLoader with a progress bar
            with torch.inference_mode():
                for i, batch in (t := tqdm(enumerate(val_loader), desc="Validation", total=math.ceil(n_val_samples / validation_config.batch_size), leave=False)):
                    # Move the batch to the specified device (GPU or CPU)
                    batch: TensorDict = batch.to(validation_config.device, non_blocking=True)

                    # Use autocast for mixed precision validation if enabled in the configuration (fp16/bf16) and disable it for fp32
                    with autocast(device_type=validation_config.device, dtype=torch.float16, enabled=validation_config.amp != "fp32"):
                        # Forward pass and loss computation
                        output: Tensor = ema_model(
                            content=batch['content'],
                            pitch=batch['pitch'],
                            amplitude=batch['amplitude'],
                            timbre=batch['timbre'],

                            content_length=batch['content_length'],
                            pitch_length=batch['pitch_length'],
                            amplitude_length=batch['amplitude_length'],
                            timbre_length=batch['timbre_length'],

                            target=batch['target'],
                            target_length=batch['target_length']
                        ) # (N, N_bins, T, D_codec)
                        loss: Tensor = loss_fn(output, batch['target'])
                        n_correct, n_total = calculate_accuracy(output, batch['target'])

                    # Accumulate the total loss for this validation epoch (multiply by batch size to get the sum of losses for all samples in the batch)
                    val_loss += loss.item() * validation_config.batch_size
                    val_correct += n_correct
                    val_total += n_total

                    # Set the description of the progress bar to show the current average loss
                    t.set_postfix({
                        "loss": f"{loss.item():.5f}",
                        "acc": f"{n_correct / (n_total + 1e-8) * 100:.3f}%",
                        "avg_loss": f"{val_loss / ((i + 1) * validation_config.batch_size + 1e-8):.5f}",
                        "avg_acc": f"{val_correct / (val_total + 1e-8) * 100:.3f}%"
                    })

            # Calculate and print the average validation loss for this epoch
            avg_val_loss = val_loss / (n_val_samples + 1e-8)
            avg_val_acc = val_correct / (val_total + 1e-8) * 100
            print(f"Average validation loss: {avg_val_loss:.5f}")
            print(f"Average validation accuracy: {avg_val_acc:.3f}%")
            run.log({"epoch": epoch, "val/loss": avg_val_loss, "val/accuracy": avg_val_acc})

        # Saving phase (save a checkpoint every few epochs)
        if epoch % train_config.save_every_n_epochs == 0:
            # Get the current time and format it as YYMMDD-HHMMSS for the checkpoint filename
            timestamp = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
            checkpoint_filename = f"checkpoint_e_{epoch:03d}_{timestamp}.pth"
            checkpoint_path = os.path.join(train_config.checkpoint_folder, checkpoint_filename)
            save_checkpoint(
                checkpoint_path,
                model, ema_model, optimizer, scheduler, scaler, epoch,
                loss=avg_train_loss,
                accuracy=avg_train_acc
            )
            run.save(checkpoint_path)
            print(f"Checkpoint saved for epoch {epoch}.")

        # Update the learning rate scheduler
        scheduler.step()

    # Finish the wandb run after training is complete
    run.finish()


# Function to continue training from a checkpoint
def continue_training():
    print("This function will load a checkpoint and continue training from where it left off.")

    # Prompt the user to enter the path to the checkpoint file
    checkpoint_path = input("Enter the path to the checkpoint file: ")
    if not os.path.isfile(checkpoint_path):
        print("Invalid checkpoint path. Please make sure the file exists and try again.")
        return
    
    # Prompt the user to enter a previous wandb run ID to continue logging to the same run (optional)
    previous_run_id = input("Enter the previous wandb run ID to continue logging to the same run (optional, press Enter to skip): ")
    if previous_run_id.strip() == "":
        previous_run_id = None

    train_model(checkpoint_path=checkpoint_path, previous_run_id=previous_run_id)


# Main function to run the training script
def main():
    print("This is the main training script. You can run specific training functions from here if needed.")
    print("Choose an action:")
    print("1. Train a model from scratch")
    print("2. Continue training an existing model")
    choice = input("Enter the number of your choice: ")
    if choice == "1":
        print("You chose to train a model from scratch.")
        train_model()
    elif choice == "2":
        print("You chose to continue training an existing model.")
        continue_training()
    else:
        print("Invalid choice. Please run the script again and choose a valid option.")

if __name__ == "__main__":
    main()