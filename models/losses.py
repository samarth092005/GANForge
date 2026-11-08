"""DCGAN Loss functions and optimizer configurations implemented in TensorFlow 2 / Keras.

Implements the standard non-saturating adversarial game proposed by Goodfellow et al. (2014)
and refined by Radford et al. (2015):
- Binary Crossentropy with `from_logits=True` for numerical stability (prevents log(0) underflow)
- Optional one-sided label smoothing for real images (Salimans et al., 2016)
- Independent Adam optimizers with beta_1=0.5 to prevent momentum oscillations
"""
from __future__ import annotations

import tensorflow as tf

from config.settings import (
    BETA_1,
    BETA_2,
    LEARNING_RATE_DISC,
    LEARNING_RATE_GEN,
)

# Standard Binary Cross Entropy computed directly on unactivated linear logits
# Mathematically: log(1 + exp(-logits)) for label=1, and log(1 + exp(logits)) for label=0
_cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def generator_loss(fake_logits: tf.Tensor) -> tf.Tensor:
    """Calculates non-saturating adversarial loss for the Generator.

    The Generator aims to fool the Discriminator into classifying generated images
    as real. Using the non-saturating formulation:
        L_G = E_{z ~ p_z} [ -log(D(G(z))) ]
    which corresponds to binary cross-entropy against a target label of 1.0 (ones_like).

    Args:
        fake_logits: Discriminator output logits for synthetic generated images [batch_size, 1].

    Returns:
        Scalar tf.Tensor containing generator loss.
    """
    # Target: All generated images should be perceived as real (target=1.0)
    return _cross_entropy(tf.ones_like(fake_logits), fake_logits)


def discriminator_loss(
    real_logits: tf.Tensor,
    fake_logits: tf.Tensor,
    label_smoothing: float = 0.0,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Calculates adversarial loss for the Discriminator.

    The Discriminator aims to maximize the probability of correctly identifying real
    samples as real (label=1.0) and synthetic generated samples as fake (label=0.0):
        L_D = -E_{x ~ p_data} [ log D(x) ] - E_{z ~ p_z} [ log(1 - D(G(z))) ]

    Args:
        real_logits: Discriminator output logits for real dataset images [batch_size, 1].
        fake_logits: Discriminator output logits for generated images [batch_size, 1].
        label_smoothing: Optional float in [0.0, 0.2] to soften real targets (e.g. 0.9 instead of 1.0)
                         preventing the Discriminator from becoming overconfident.

    Returns:
        Tuple of (total_loss, real_loss, fake_loss).
    """
    # Real loss: target is 1.0 (or 1.0 - label_smoothing)
    real_targets = tf.ones_like(real_logits) * (1.0 - label_smoothing)
    real_loss = _cross_entropy(real_targets, real_logits)

    # Fake loss: target is strictly 0.0 (never smooth fake labels)
    fake_targets = tf.zeros_like(fake_logits)
    fake_loss = _cross_entropy(fake_targets, fake_logits)

    # Total loss is the unweighted sum of real and fake classification losses
    total_loss = real_loss + fake_loss

    return total_loss, real_loss, fake_loss


def get_optimizers(
    learning_rate_gen: float = LEARNING_RATE_GEN,
    learning_rate_disc: float = LEARNING_RATE_DISC,
    beta_1: float = BETA_1,
    beta_2: float = BETA_2,
) -> tuple[tf.keras.optimizers.Optimizer, tf.keras.optimizers.Optimizer]:
    """Instantiates Adam optimizers tuned specifically for DCGAN training.

    Standard Adam sets beta_1=0.9, which introduces excessive momentum and causes
    training to oscillate or diverge in adversarial games. Following Radford et al.,
    beta_1 is reduced to 0.5 to stabilize adversarial equilibrium.

    Args:
        learning_rate_gen: Learning rate for the Generator (default 0.0002).
        learning_rate_disc: Learning rate for the Discriminator (default 0.0002).
        beta_1: Exponential decay rate for the 1st moment estimates (default 0.5).
        beta_2: Exponential decay rate for the 2nd moment estimates (default 0.999).

    Returns:
        Tuple of (generator_optimizer, discriminator_optimizer).
    """
    gen_opt = tf.keras.optimizers.Adam(
        learning_rate=learning_rate_gen,
        beta_1=beta_1,
        beta_2=beta_2,
        name="generator_adam",
    )
    disc_opt = tf.keras.optimizers.Adam(
        learning_rate=learning_rate_disc,
        beta_1=beta_1,
        beta_2=beta_2,
        name="discriminator_adam",
    )
    return gen_opt, disc_opt
