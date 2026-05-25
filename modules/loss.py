# Loss Functions for Neural Networks
import torch

import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.func import jvp
from .generator import MeanFlowsGenerator


class MeanFlowsAdaptedLoss(nn.Module):
    """Adapted squared L2 loss function for MeanFlows model."""
    def __init__(self, omega: float = 0.2, kappa: float = 0.9, p: float = 1.0, c: float = 1e-3):
        """
        Initializes the MeanFlowsAdaptedLoss.
        Args:
            omega (float): Guidance scale for sample. Default is 0.2.
            kappa (float): Guidance scale for conditioning output. Default is 0.9.
            p (float): Power to which the loss is raised. Default is 1.0.
            c (float): Small constant to prevent division by zero. Default is 1e-3.
        """
        super().__init__()
        self.omega = omega
        self.kappa = kappa
        self.p = p
        self.c = c
        self.loss_fn = nn.MSELoss(reduction='none')

    def forward(
        self,
        model: MeanFlowsGenerator, target: Tensor, epsilon: Tensor,
        content: Tensor, pitch: Tensor, amplitude: Tensor, timbre: Tensor,
        start_timestep: Tensor, end_timestep: Tensor,
        content_length: Tensor | None = None, pitch_length: Tensor | None = None,
        amplitude_length: Tensor | None = None, timbre_length: Tensor | None = None,
        target_length: Tensor | None = None, drop_cond: Tensor | None = None
    ):
        """
        Computes the adapted loss between predictions and targets.

        Args:
            model (MeanFlowsGenerator): The generator model to compute predictions.
            target (Tensor): Target tensor of shape (N, T, D_codec).
            epsilon (Tensor): Noise tensor of shape (N, T, D_codec).

            content (Tensor): Content features of shape (N, T_content, D_content).
            pitch (Tensor): Pitch features of shape (N, T_pitch).
            amplitude (Tensor): Amplitude features of shape (N, T_amplitude).
            timbre (Tensor): Timbre features of shape (N, T_timbre, D_timbre).

            start_timestep (Tensor): Start time step (r) tensor of shape (N,). Note that 0.0 <= r <= t <= 1.0.
            end_timestep (Tensor): End time step (t) tensor of shape (N,). Note that 0.0 <= r <= t <= 1.0.

            content_length (Tensor | None): Lengths of content sequences for masking, shape (N,). Should be None if no masking is required.
            pitch_length (Tensor | None): Lengths of pitch sequences for masking, shape (N,). Should be None if no masking is required.
            amplitude_length (Tensor | None): Lengths of amplitude sequences for masking, shape (N,). Should be None if no masking is required.
            timbre_length (Tensor | None): Lengths of timbre sequences for masking, shape (N,). Should be None if no masking is required.
            target_length (Tensor | None): Lengths of target sequences for masking, shape (N,). Should be None if no masking is required.

            drop_cond (Tensor | None): Optional tensor of shape (N,) indicating which samples in the batch should have their conditioning features dropped for regularization.
            Should be binary (0 or 1) and can be None if no conditioning dropout is applied.

        Returns:
            tuple[Tensor, Tensor]: A tuple containing:
                - output (Tensor): The model's output predictions of shape (N, T, D_codec).
                - loss (Tensor): The computed loss value.
        """
        N, T, D_codec = target.shape

        # Step 1: Create noisy input by adding noise to the target
        t: Tensor = end_timestep.view(-1, 1, 1) # (N, 1, 1)
        zt: Tensor = target * (1 - t) + epsilon * t # (N, T, D_codec)

        with torch.no_grad():
            # Step 2: Conditional pass mixed with unconditional pass (depending on drop_cond)
            u_cond: Tensor = model.forward(
                content=content, pitch=pitch, amplitude=amplitude, timbre=timbre,
                start_timestep=end_timestep, end_timestep=end_timestep,
                zt=zt, content_length=content_length, pitch_length=pitch_length,
                amplitude_length=amplitude_length, timbre_length=timbre_length,
                zt_length=target_length, drop_cond=drop_cond
            ) # (N, T, D_codec)

            # Step 3: Fully unconditional pass (by zeroing out conditioning features)
            if drop_cond is not None:
                fully_drop_cond: Tensor = torch.ones_like(drop_cond) # (N,)
            else:
                fully_drop_cond: Tensor = torch.ones(N, dtype=torch.bool, device=target.device) # (N,)

            u_uncond: Tensor = model.forward(
                content=content, pitch=pitch, amplitude=amplitude, timbre=timbre,
                start_timestep=end_timestep, end_timestep=end_timestep,
                zt=zt, content_length=content_length, pitch_length=pitch_length,
                amplitude_length=amplitude_length, timbre_length=timbre_length,
                zt_length=target_length, drop_cond=fully_drop_cond
            ) # (N, T, D_codec)

            # Step 4: Combine conditional and unconditional outputs to calculate the intermediate velocity
            v_tilde: Tensor = self.omega * (epsilon - target) + self.kappa * u_cond + (1 - self.omega - self.kappa) * u_uncond # (N, T, D_codec)

        # Step 5: Create the model function for JVP
        def model_fn(zt: Tensor, r: Tensor, t: Tensor) -> Tensor:
            return model.forward(
                content=content, pitch=pitch, amplitude=amplitude, timbre=timbre,
                start_timestep=r, end_timestep=t,
                zt=zt, content_length=content_length, pitch_length=pitch_length,
                amplitude_length=amplitude_length, timbre_length=timbre_length,
                zt_length=target_length, drop_cond=drop_cond
            )
        
        # Step 6: Create the tangent vector for JVP (the noise direction)
        tangent_zt: Tensor = v_tilde # (N, T, D_codec)
        tangent_r: Tensor = torch.zeros_like(start_timestep) # (N,)
        tangent_t: Tensor = torch.ones_like(end_timestep) # (N,)

        # Step 7: Compute the JVP to get the model's velocity prediction
        u_primal, u_tangent = jvp(
            func=model_fn,
            primals=(zt, start_timestep, end_timestep),
            tangents=(tangent_zt, tangent_r, tangent_t)
        )

        # Step 8: Compute the target velocity and the raw loss
        diff_timestep: Tensor = (end_timestep - start_timestep).view(-1, 1, 1) # (N, 1, 1)
        u_target: Tensor = v_tilde - diff_timestep * u_tangent # (N, T, D_codec)
        raw_loss: Tensor = self.loss_fn(u_primal, u_target.detach()).mean(dim=-1) # (N, T)

        # Step 9: Apply the adaptation factor to the loss
        w: Tensor = 1.0 / (raw_loss + self.c).pow(self.p) # (N, T)
        adapted_loss: Tensor = w.detach() * raw_loss # (N, T)

        # Step 10: Apply masking and proper reduction to get the final loss value
        if target_length is not None:
            mask: Tensor = torch.arange(T, device=target.device).unsqueeze(0) < target_length.unsqueeze(1) # (N, T)
            adapted_loss = adapted_loss * mask
            final_loss: Tensor = adapted_loss.sum() / mask.sum() # Only average over valid (non-padded) time steps
        else:
            final_loss: Tensor = adapted_loss.mean()

        return final_loss