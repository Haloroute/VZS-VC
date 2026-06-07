# Main training loop
import datasets, datetime, math, os, random, torch, wandb
import numpy as np

from dataclasses import asdict
from datasets import DatasetBuilder
from functools import partial
from tensordict import TensorDict
from torch import Tensor
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup

from modules import DiscriminatorLoss, GeneratorLoss, VoiceDiscriminator, VoiceGenerator
from utils.configs import (
    TrainConfig,
    ValidationConfig,
    VieNeuTTSPreprocessedDatasetConfig,
    VoiceGeneratorModuleConfig
)
from utils.dataset import collate_fn
from utils.logger import load_checkpoint, save_checkpoint
from utils.modules import load_discriminator, load_generator


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
    run.define_metric("val/orig_loss", step_metric="epoch")
    run.define_metric("val/orig_accuracy", step_metric="epoch")
    run.define_metric("val/ema_loss", step_metric="epoch")
    run.define_metric("val/ema_accuracy", step_metric="epoch")

    # Load the preprocessed dataset using the specified configuration
    dataset = datasets.load_dataset(
        dataset_config.path,
        streaming=dataset_config.streaming
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
    collate_fn_wrapper = partial(collate_fn, dataset_config=dataset_config, train_config=train_config, model_config=model_config)
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=not dataset_config.streaming, # Shuffle only if not streaming (streaming datasets cannot be shuffled in-memory)
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
    discriminator: VoiceDiscriminator = load_discriminator(train_config.device)
    ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(decay=train_config.ema_decay))

    print("Model loaded successfully.")
    print(f"Model parameters count: {sum(p.numel() for p in model.parameters())}")

    # Setup the loss function for the generator and discriminator
    gen_loss_fn = GeneratorLoss(
        lambda_recon=train_config.lambda_recon,
        lambda_adv=train_config.lambda_adv,
        lambda_fm=train_config.lambda_fm
    )
    dis_loss_fn = DiscriminatorLoss()

    # Setup the optimizers for the generator and discriminator
    dis_optimizer = AdamW(
        discriminator.parameters(),
        lr=train_config.lr,
        betas=train_config.beta,
        weight_decay=train_config.weight_decay
    )
    gen_optimizer = AdamW(
        model.parameters(),
        lr=train_config.lr,
        betas=train_config.beta,
        weight_decay=train_config.weight_decay
    )

    # Setup the learning rate schedulers for the generator and discriminator
    dis_scheduler = get_cosine_schedule_with_warmup(
        dis_optimizer,
        num_warmup_steps=train_config.n_warmup_epochs,
        num_training_steps=train_config.n_epochs
    )
    gen_scheduler = get_cosine_schedule_with_warmup(
        gen_optimizer,
        num_warmup_steps=train_config.n_warmup_epochs,
        num_training_steps=train_config.n_epochs
    )

    # Setup the GradScaler for mixed precision training (use fp16 for NVIDIA GPUs, bf16 for AMD GPUs, and disable for CPU)
    gen_scaler = GradScaler(device=train_config.device, enabled=train_config.amp == torch.float16) # Use GradScaler for mixed precision training (fp16) and disable it for bf16 or fp32
    dis_scaler = GradScaler(device=train_config.device, enabled=train_config.amp == torch.float16) # Use GradScaler for mixed precision training (fp16) and disable it for bf16 or fp32

    torch.manual_seed(train_config.seed)
    np.random.seed(train_config.seed)
    random.seed(train_config.seed)

    print("Optimizer and EMA model set up successfully.")

    # Load checkpoint if a path is provided
    if checkpoint_path is not None:
        print(f"Loading checkpoint from {checkpoint_path}...")
        _, start_epoch = load_checkpoint(
            checkpoint_path,
            model, ema_model, discriminator,
            gen_optimizer, dis_optimizer, gen_scheduler, dis_scheduler, gen_scaler, dis_scaler
        )
        print(f"Checkpoint loaded successfully. Resuming from epoch {start_epoch}.")

        # # Update the learning rate of the optimizer
        # for param_group in optimizer.param_groups:
        #     if 'initial_lr' in param_group and 'lr' in param_group:
        #         param_group['lr'] *= train_config.lr / (param_group['initial_lr'] + 1e-8)
        #         param_group['initial_lr'] = train_config.lr

        # # Update the learning rate of the scheduler
        # if hasattr(scheduler, 'base_lrs'):
        #     scheduler.base_lrs = [train_config.lr for _ in scheduler.base_lrs]

    else:
        start_epoch = 0
        print("No checkpoint provided. Starting training from scratch.")

    # Compile the model with torch.compile for potential speed improvements (optional, can be disabled if it causes issues)
    if train_config.compiled:
        model = torch.compile(model, dynamic=True, fullgraph=True)
        discriminator = torch.compile(discriminator, dynamic=True, fullgraph=True)
        print("Enabled torch.compile for the model.")
    if validation_config.compiled:
        ema_model.module = torch.compile(ema_model.module, dynamic=True, fullgraph=True)
        print("Enabled torch.compile for the EMA model.")

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
        total_gen_loss, total_dis_loss, total_samples = 0.0, 0.0, 0

        # Iterate over the training DataLoader with a progress bar
        for i, batch in (t := tqdm(enumerate(train_loader), desc="Training", total=math.ceil(n_train_samples / train_config.batch_size), leave=False)):
            # Move the batch to the specified device (GPU or CPU)
            batch: TensorDict = batch.to(train_config.device, non_blocking=True)
            # batch is a TensorDict containing:
            # "content": content_padded, # (N, T', D_content)
            # "pitch": pitch_padded, # (N, T)
            # "amplitude": amplitude_padded, # (N, T)
            # "mel": mel_padded, # (N, 2T, n_mel_bins)

            # "mask_indices": mask_indices_padded, # (N, T)
            # "content_length": content_lengths, # (N,)
            # "token_length": mel_lengths, # (N,)

            # 1. Train the discriminator on real and generated samples
            # Zero the gradients
            dis_optimizer.zero_grad()

            # Use autocast for mixed precision training if enabled in the configuration (fp16/bf16) and disable it for fp32
            with autocast(device_type=train_config.device, dtype=train_config.amp, enabled=train_config.amp != torch.float32):
                # Forward pass for the generator (use the generator's output as input to the discriminator)
                with torch.no_grad():
                    output: Tensor = model(
                        content=batch['content'],
                        pitch=batch['pitch'],
                        amplitude=batch['amplitude'],
                        mel=batch['mel'],

                        mask_indices=batch['mask_indices'],
                        content_length=batch['content_length'],
                        token_length=batch['token_length'],
                    )

                # Forward pass for the discriminator
                dis_gen, _ = discriminator(output.detach()) # (N, T), list of feature maps [(N, D, T), ...]
                dis_target, _ = discriminator(batch['mel']) # (N, T), list of feature maps [(N, D, T), ...]

                # Discriminator loss function
                dis_loss: Tensor = dis_loss_fn(
                    dis_gen=dis_gen,
                    dis_target=dis_target,
                    mask_indices=batch['mask_indices'],
                    token_length=batch['token_length']
                )

            # Backpropagation and optimization step for the discriminator
            dis_scaler.scale(dis_loss).backward()
            dis_scaler.unscale_(dis_optimizer)
            clip_grad_norm_(discriminator.parameters(), max_norm=train_config.clip_grad_norm)
            dis_scaler.step(dis_optimizer)
            dis_scaler.update()

            # 2. Train the generator to fool the discriminator and reconstruct the target Mel-spectrogram
            # Zero the gradients
            gen_optimizer.zero_grad()

            # Use autocast for mixed precision training if enabled in the configuration (fp16/bf16) and disable it for fp32
            with autocast(device_type=train_config.device, dtype=train_config.amp, enabled=train_config.amp != torch.float32):
                # Forward pass for the generator
                output: Tensor = model(
                    content=batch['content'],
                    pitch=batch['pitch'],
                    amplitude=batch['amplitude'],
                    mel=batch['mel'],

                    mask_indices=batch['mask_indices'],
                    content_length=batch['content_length'],
                    token_length=batch['token_length'],
                ) # (N, 2T, n_mel_bins)

                # Forward pass for the discriminator on the generator's output and the real target Mel-spectrogram
                dis_gen, fmap_gen = discriminator(output) # (N, T), list of feature maps [(N, D, T), ...]
                with torch.no_grad():
                    _, fmap_target = discriminator(batch['mel']) # (N, T), list of feature maps [(N, D, T), ...]

                # Generator loss function (reconstruction + adversarial + feature matching)
                gen_loss: Tensor = gen_loss_fn(
                    target=batch['mel'],
                    output=output,
                    dis_gen=dis_gen,
                    fmap_gen=fmap_gen,
                    fmap_target=fmap_target,
                    mask_indices=batch['mask_indices']
                )

            # Backpropagation and optimization step for the generator
            gen_scaler.scale(gen_loss).backward()
            gen_scaler.unscale_(gen_optimizer)
            clip_grad_norm_(model.parameters(), max_norm=train_config.clip_grad_norm) # Gradient clipping to prevent exploding gradients
            gen_scaler.step(gen_optimizer)
            gen_scaler.update()

            # Update the EMA model with the current generator parameters
            ema_model.update_parameters(model)

            # Accumulate the total loss for this epoch
            gen_loss, dis_loss = gen_loss.item(), dis_loss.item()
            total_gen_loss, total_dis_loss = total_gen_loss + gen_loss, total_dis_loss + dis_loss
            total_samples += 1

            # Set the description of the progress bar to show the current average loss
            t.set_postfix({
                "gen": f"{gen_loss:.5f}",
                "dis": f"{dis_loss:.5f}",
                "avg_gen": f"{total_gen_loss / (total_samples + 1e-8):.5f}",
                "avg_dis": f"{total_dis_loss / (total_samples + 1e-8):.5f}"
            })

        # Calculate and print the average training loss for this epoch
        avg_gen_loss = total_gen_loss / (total_samples + 1e-8)
        avg_dis_loss = total_dis_loss / (total_samples + 1e-8)

        print(f"Average training loss: {avg_gen_loss:.5f} (Generator), {avg_dis_loss:.5f} (Discriminator)")
        run.log({"epoch": epoch, "train/gen_loss": avg_gen_loss, "train/dis_loss": avg_dis_loss})

        # Validation phase (optional, can be done every few epochs to save time)
        if epoch % validation_config.validate_every_n_epochs == 0: # Validate every few epochs
            # Validation on EMA model
            ema_model.eval()
            total_gen_loss, total_dis_loss, total_samples = 0.0, 0.0, 0

            # Iterate over the validation DataLoader with a progress bar
            with torch.inference_mode():
                for i, batch in (t := tqdm(enumerate(val_loader), desc="Validation", total=math.ceil(n_val_samples / validation_config.batch_size), leave=False)):
                    # Move the batch to the specified device (GPU or CPU)
                    batch: TensorDict = batch.to(validation_config.device, non_blocking=True)

                    # Use autocast for mixed precision validation if enabled in the configuration (fp16/bf16) and disable it for fp32
                    with autocast(device_type=validation_config.device, dtype=validation_config.amp, enabled=validation_config.amp != torch.float32):
                        # Forward pass for the EMA generator and discriminator
                        output: Tensor = ema_model(
                            content=batch['content'],
                            pitch=batch['pitch'],
                            amplitude=batch['amplitude'],
                            mel=batch['mel'],

                            mask_indices=batch['mask_indices'],
                            content_length=batch['content_length'],
                            token_length=batch['token_length'],
                        ) # (N, 2T, n_mel_bins)
                        dis_gen, fmap_gen = discriminator(output) # (N, T), list of feature maps [(N, D, T), ...]                   
                        dis_target, fmap_target = discriminator(batch['mel']) # (N, T), list of feature maps [(N, D, T), ...]

                        # Generator loss function (reconstruction + adversarial + feature matching) and discriminator loss function
                        gen_loss: Tensor = gen_loss_fn(
                            target=batch['mel'],
                            output=output,
                            dis_gen=dis_gen,
                            fmap_gen=fmap_gen,
                            fmap_target=fmap_target,
                            mask_indices=batch['mask_indices']
                        )
                        dis_loss: Tensor = dis_loss_fn(
                            dis_gen=dis_gen,
                            dis_target=dis_target,
                            mask_indices=batch['mask_indices'],
                            token_length=batch['token_length']
                        )

                    # Accumulate the total loss for this epoch
                    gen_loss, dis_loss = gen_loss.item(), dis_loss.item()
                    total_gen_loss, total_dis_loss = total_gen_loss + gen_loss, total_dis_loss + dis_loss
                    total_samples += 1

                    # Set the description of the progress bar to show the current average loss
                    t.set_postfix({
                        "gen": f"{gen_loss:.5f}",
                        "dis": f"{dis_loss:.5f}",
                        "avg_gen": f"{total_gen_loss / (total_samples + 1e-8):.5f}",
                        "avg_dis": f"{total_dis_loss / (total_samples + 1e-8):.5f}"
                    })

            # Calculate and print the average validation loss for this epoch
            avg_gen_loss = total_gen_loss / (total_samples + 1e-8)
            avg_dis_loss = total_dis_loss / (total_samples + 1e-8)

            print(f"Average validation loss: {avg_gen_loss:.5f} (Generator), {avg_dis_loss:.5f} (Discriminator)")
            run.log({"epoch": epoch, "val/gen_loss": avg_gen_loss, "val/dis_loss": avg_dis_loss})

        # Saving phase (save a checkpoint every few epochs)
        if epoch % train_config.save_every_n_epochs == 0:
            # Get the current time and format it as YYMMDD-HHMMSS for the checkpoint filename
            timestamp = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
            checkpoint_filename = f"checkpoint_e_{epoch:03d}_{timestamp}.pth"
            checkpoint_path = os.path.join(train_config.checkpoint_folder, checkpoint_filename)
            save_checkpoint(
                checkpoint_path,
                model, ema_model, discriminator,
                gen_optimizer, dis_optimizer, gen_scheduler, dis_scheduler, gen_scaler, dis_scaler, epoch
            )
            run.save(checkpoint_path)
            print(f"Checkpoint saved for epoch {epoch}.")

        # Update the learning rate schedulers
        gen_scheduler.step()
        dis_scheduler.step()

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