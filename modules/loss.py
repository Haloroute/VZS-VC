# Loss Functions for Neural Networks
import torch

import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from .generator import MeanFlowsGenerator


class MeanFlowsAdaptedLoss(nn.Module):
    """Adapted squared L2 loss function for MeanFlows model."""
    def __init__(
        self,
        omega: float = 0.2,
        kappa: float = 0.9,
        p: float = 1.0,
        c: float = 1e-3,
        epsilon: float = 2 ** (-25/3)
    ):
        """
        Initializes the MeanFlowsAdaptedLoss.
        Args:
            omega (float): Guidance scale for sample. Default is 0.2.
            kappa (float): Guidance scale for conditioning output. Default is 0.9.
            p (float): Power to which the loss is raised. Default is 1.0.
            c (float): Small constant to prevent division by zero. Default is 1e-3.
            epsilon (float): Small constant to calculate the approximate JVP using differentiation approximation. 
                Default is 2^(-25/3) for Central Finite Difference approximation, and 2^-12 for Forward Finite Difference approximation.
        """
        super().__init__()
        self.omega, self.kappa, self.p, self.c, self.epsilon = omega, kappa, p, c, epsilon
        self.loss_fn = nn.MSELoss(reduction='none')

    def forward(
        self,
        model: MeanFlowsGenerator, target: Tensor, epsilon: Tensor,
        content: Tensor, pitch: Tensor, amplitude: Tensor, timbre: Tensor,
        start_time: Tensor, end_time: Tensor,
        target_length: Tensor,
        content_length: Tensor, pitch_length: Tensor,
        amplitude_length: Tensor, timbre_length: Tensor,
        drop_cond: Tensor
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

            start_time (Tensor): Start time step (r) tensor of shape (N,). Note that 0.0 <= r <= t <= 1.0.
            end_time (Tensor): End time step (t) tensor of shape (N,). Note that 0.0 <= r <= t <= 1.0.

            target_length (Tensor): Length of target sequences for masking, shape (N,).
            content_length (Tensor): Length of content sequences for masking, shape (N,).
            pitch_length (Tensor): Length of pitch sequences for masking, shape (N,).
            amplitude_length (Tensor): Length of amplitude sequences for masking, shape (N,).
            timbre_length (Tensor): Length of timbre sequences for masking, shape (N,).

            drop_cond (Tensor): Binary tensor of shape (N,) indicating which samples in the batch should have their conditioning features dropped for regularization.

        Returns:
            tuple[Tensor, Tensor]: A tuple containing:
                - output (Tensor): The model's output predictions of shape (N, T, D_codec).
                - loss (Tensor): The computed loss value.
        """
        N, T, _ = target.shape
        device = target.device

        # Step 1: Create noisy input by adding noise to the target
        t: Tensor = end_time.view(N, 1, 1) # (N, 1, 1)
        zt: Tensor = target * (1 - t) + epsilon * t # (N, T, D_codec)

        # ==========================================
        # CALL 1: Batched u_cond and u_uncond (no_grad, AMP enabled via train.py)
        # ==========================================
        # Step 2: Prepare batched inputs for u_cond and u_uncond
        batched_content = torch.cat([content, content], dim=0)
        batched_pitch = torch.cat([pitch, pitch], dim=0)
        batched_amplitude = torch.cat([amplitude, amplitude], dim=0)
        batched_timbre = torch.cat([timbre, timbre], dim=0)

        # u_cond and u_uncond both use end_time for both start and end
        batched_end_time = torch.cat([end_time, end_time], dim=0)
        batched_zt = torch.cat([zt, zt], dim=0)

        batched_target_len = torch.cat([target_length, target_length], dim=0)
        batched_content_len = torch.cat([content_length, content_length], dim=0)
        batched_pitch_len = torch.cat([pitch_length, pitch_length], dim=0)
        batched_amplitude_len = torch.cat([amplitude_length, amplitude_length], dim=0)
        batched_timbre_len = torch.cat([timbre_length, timbre_length], dim=0)

        # drop_cond for u_cond is drop_cond, for u_uncond is ones_like(drop_cond)
        fully_drop_cond = torch.ones_like(drop_cond)
        batched_drop_cond = torch.cat([drop_cond, fully_drop_cond], dim=0)

        # Step 3: Compute u_cond and u_uncond in a single batched forward pass with no_grad and AMP enabled
        with torch.no_grad():
            u_cond_uncond = model(
                content=batched_content, pitch=batched_pitch, amplitude=batched_amplitude, timbre=batched_timbre,
                start_time=batched_end_time, end_time=batched_end_time,
                zt=batched_zt, content_length=batched_content_len, pitch_length=batched_pitch_len,
                amplitude_length=batched_amplitude_len, timbre_length=batched_timbre_len,
                zt_length=batched_target_len, drop_cond=batched_drop_cond
            )
            u_cond, u_uncond = u_cond_uncond.chunk(2, dim=0)
            v_tilde = self.omega * (epsilon - target) + self.kappa * u_cond + (1 - self.omega - self.kappa) * u_uncond

        # ==========================================
        # CALL 2: Batched u_plus and u_minus (no_grad, AMP disabled, Force FP32)
        # ==========================================
        # Step 4: Prepare tangent vectors for JVP. We want to compute the JVP with respect to zt, start_time, and end_time.
        tangent_zt = v_tilde
        tangent_r = torch.zeros_like(start_time)
        tangent_t = torch.ones_like(end_time)

        with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=False):
            # Step 5: Force all inputs to float32 for numerical stability in the finite difference approximation
            batched_content_f32 = batched_content.float()
            batched_pitch_f32 = batched_pitch.float()
            batched_amplitude_f32 = batched_amplitude.float()
            batched_timbre_f32 = batched_timbre.float()

            zt_f32 = zt.float()            
            start_time_f32 = start_time.float()
            end_time_f32 = end_time.float()
            tangent_zt_f32 = tangent_zt.float()
            tangent_r_f32 = tangent_r.float()
            tangent_t_f32 = tangent_t.float()

            batched_zt_cfd = torch.cat([zt_f32 + self.epsilon * tangent_zt_f32, zt_f32 - self.epsilon * tangent_zt_f32], dim=0)
            batched_start_cfd = torch.cat([start_time_f32 + self.epsilon * tangent_r_f32, start_time_f32 - self.epsilon * tangent_r_f32], dim=0)
            batched_end_cfd = torch.cat([end_time_f32 + self.epsilon * tangent_t_f32, end_time_f32 - self.epsilon * tangent_t_f32], dim=0)

            batched_drop_cond_cfd = torch.cat([drop_cond, drop_cond], dim=0)

            # Step 6: Compute the Central Finite Difference approximation of the JVP with no_grad and AMP disabled for numerical stability
            u_plus_minus = model(
                content=batched_content_f32, pitch=batched_pitch_f32, amplitude=batched_amplitude_f32, timbre=batched_timbre_f32,
                start_time=batched_start_cfd, end_time=batched_end_cfd,
                zt=batched_zt_cfd, content_length=batched_content_len, pitch_length=batched_pitch_len,
                amplitude_length=batched_amplitude_len, timbre_length=batched_timbre_len,
                zt_length=batched_target_len, drop_cond=batched_drop_cond_cfd
            )
            u_plus, u_minus = u_plus_minus.chunk(2, dim=0)
            u_tangent = (u_plus - u_minus) / (2 * self.epsilon)

        # ==========================================
        # CALL 3: Primal Output (requires_grad, AMP enabled)
        # ==========================================
        # Step 7: Compute the primal output at the original input
        u_primal: Tensor = model(
            content=content, pitch=pitch, amplitude=amplitude, timbre=timbre,
            start_time=start_time, end_time=end_time,
            zt=zt, content_length=content_length, pitch_length=pitch_length,
            amplitude_length=amplitude_length, timbre_length=timbre_length,
            zt_length=target_length, drop_cond=drop_cond
        ) # (N, T, D_codec)

        # Step 8: Compute the target velocity and the raw loss
        diff_timestep: Tensor = (end_time - start_time).view(N, 1, 1) # (N, 1, 1)
        u_target: Tensor = v_tilde - diff_timestep * u_tangent # (N, T, D_codec)
        raw_loss: Tensor = self.loss_fn(u_primal, u_target.detach()).mean(dim=-1) # (N, T)

        # Step 9: Apply the adaptation factor to the loss
        w: Tensor = 1.0 / (raw_loss + self.c).pow(self.p) # (N, T)
        adapted_loss: Tensor = w.detach() * raw_loss # (N, T)

        # Step 10: Apply masking and proper reduction to get the final loss value
        mask: Tensor = (torch.arange(T, device=target.device).unsqueeze(0) < target_length.unsqueeze(1)).float() # (N, T)
        adapted_loss = adapted_loss * mask
        final_loss: Tensor = adapted_loss.sum() / (mask.sum() + self.c) # Only average over valid (non-padded) time steps

        return final_loss