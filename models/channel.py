# models/channel.py
import torch
import torch.nn as nn

class DigitalV2XChannel(nn.Module):
    def __init__(self, erasure_token_index=-1):
        """
        Simulate a digital V2X communication channel with erasure and bit-flip noise.
        erasure_token_index: The index value representing an erased token (default -1).
        """
        super().__init__()
        self.erasure_token_index = erasure_token_index

    def forward(self, indices, erasure_rate=0.0, bit_flip_prob=0.0, num_bits=8):
        """
        indices: Tensor of shape [B, H_latent, W_latent] containing the original codebook indices of Target Y
        erasure_rate: Probability of packet loss (epsilon) in [0.0, 1.0]
        bit_flip_prob: Probability of bit flip in a binary symmetric channel (BSC)
        num_bits: log2(K), number of bits per token (default 8 bits for K=256)
        """
        corrupted_indices = indices.clone()

        # 1. Simulate bit flip (BSC) if probability > 0
        if bit_flip_prob > 0.0:
            bit_mask = torch.rand(*indices.shape, num_bits, device=indices.device) < bit_flip_prob
            flipped_indices = torch.zeros_like(indices)
            for b in range(num_bits):
                bit_b = (corrupted_indices >> b) & 1
                flipped_bit = bit_b ^ bit_mask[..., b].long()
                flipped_indices |= (flipped_bit << b)
            corrupted_indices = flipped_indices

        # 2. Simulate packet loss (BEC)
        if erasure_rate > 0.0:
            rand_matrix = torch.rand(indices.shape, device=indices.device)
            erasure_mask = (rand_matrix < erasure_rate)
            corrupted_indices[erasure_mask] = self.erasure_token_index
        else:
            erasure_mask = torch.zeros(indices.shape, dtype=torch.bool, device=indices.device)

        return corrupted_indices, erasure_mask