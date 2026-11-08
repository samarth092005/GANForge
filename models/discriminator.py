"""DCGAN Discriminator architecture implemented in TensorFlow 2 / Keras.

Accepts 64x64x3 RGB images (real or generated) and outputs an unactivated scalar logit
representing the likelihood of the image being authentic versus synthetic.

Adheres to Radford et al. (2015) DCGAN architecture guidelines:
- Strided convolutions (Conv2D) for spatial downsampling (replacing pooling layers)
- LeakyReLU activations throughout intermediate layers (alpha=0.2)
- BatchNormalization on intermediate layers (omitted from first layer per DCGAN paper)
- Dropout regularization to stabilize adversarial min-max dynamics
- Single unactivated linear logit output (no final Sigmoid) for numerically stable cross-entropy with from_logits=True
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models

from config.settings import CHANNELS, IMAGE_SIZE


def build_discriminator(
    image_size: tuple[int, int] = IMAGE_SIZE,
    channels: int = CHANNELS,
    dropout_rate: float = 0.3,
    name: str = "DCGAN_Discriminator",
) -> tf.keras.Model:
    """Builds and compiles the Deep Convolutional GAN Discriminator.

    Args:
        image_size: (height, width) tuple of input images (default 64x64).
        channels: Number of color channels (default 3 for RGB).
        dropout_rate: Dropout probability for regularization (default 0.3).
        name: Name for the Keras Model.

    Returns:
        Compiled tf.keras.Model mapping (batch_size, height, width, channels) -> (batch_size, 1).
    """
    kernel_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    model = models.Sequential(name=name)

    # Input: 64x64xchannels
    input_shape = (image_size[0], image_size[1], channels)
    model.add(layers.Input(shape=input_shape, name="image_input"))

    # 1. Downsample: 64x64x3 -> 32x32x64 (No BatchNorm on first discriminator layer per DCGAN design)
    model.add(
        layers.Conv2D(
            64,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="same",
            kernel_initializer=kernel_init,
            name="conv_32x32",
        )
    )
    model.add(layers.LeakyReLU(negative_slope=0.2, name="leaky_0"))
    model.add(layers.Dropout(dropout_rate, name="dropout_0"))

    # 2. Downsample: 32x32x64 -> 16x16x128
    model.add(
        layers.Conv2D(
            128,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="same",
            use_bias=False,
            kernel_initializer=kernel_init,
            name="conv_16x16",
        )
    )
    model.add(layers.BatchNormalization(momentum=0.9, name="bn_1"))
    model.add(layers.LeakyReLU(negative_slope=0.2, name="leaky_1"))
    model.add(layers.Dropout(dropout_rate, name="dropout_1"))

    # 3. Downsample: 16x16x128 -> 8x8x256
    model.add(
        layers.Conv2D(
            256,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="same",
            use_bias=False,
            kernel_initializer=kernel_init,
            name="conv_8x8",
        )
    )
    model.add(layers.BatchNormalization(momentum=0.9, name="bn_2"))
    model.add(layers.LeakyReLU(negative_slope=0.2, name="leaky_2"))
    model.add(layers.Dropout(dropout_rate, name="dropout_2"))

    # 4. Downsample: 8x8x256 -> 4x4x512
    model.add(
        layers.Conv2D(
            512,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="same",
            use_bias=False,
            kernel_initializer=kernel_init,
            name="conv_4x4",
        )
    )
    model.add(layers.BatchNormalization(momentum=0.9, name="bn_3"))
    model.add(layers.LeakyReLU(negative_slope=0.2, name="leaky_3"))
    model.add(layers.Dropout(dropout_rate, name="dropout_3"))

    # 5. Output Head: Flatten to feature vector and project to 1-dimensional unactivated logit
    model.add(layers.Flatten(name="flatten"))
    model.add(
        layers.Dense(
            1,
            kernel_initializer=kernel_init,
            name="logit_output",
        )
    )

    return model
