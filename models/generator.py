"""DCGAN Generator architecture implemented in TensorFlow 2 / Keras.

Transforms a 100-dimensional continuous latent noise vector z ~ N(0, 1)
into a synthetic 64x64x3 RGB image with values normalized in [-1, 1].

Adheres to Radford et al. (2015) DCGAN architecture guidelines:
- Dense projection with spatial Reshape
- Fractional-strided convolutions (Conv2DTranspose) for upsampling
- BatchNormalization on all intermediate generator layers
- ReLU activations in all generator layers except output
- Tanh activation in output layer to bound pixel intensities to [-1, 1]
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models

from config.settings import CHANNELS, LATENT_DIM


def build_generator(
    latent_dim: int = LATENT_DIM,
    channels: int = CHANNELS,
    name: str = "DCGAN_Generator",
) -> tf.keras.Model:
    """Builds and compiles the Deep Convolutional GAN Generator.

    Args:
        latent_dim: Dimensionality of input latent noise vector (default 100).
        channels: Number of output image color channels (default 3 for RGB).
        name: Name for the Keras Model.

    Returns:
        Compiled tf.keras.Model mapping (batch_size, latent_dim) -> (batch_size, 64, 64, channels).
    """
    # Standard normal weight initialization with stddev=0.02 (DCGAN paper standard)
    kernel_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    model = models.Sequential(name=name)

    # 1. Project latent vector z into a 4x4x512 spatial feature map
    model.add(layers.Input(shape=(latent_dim,), name="latent_input"))
    model.add(
        layers.Dense(
            4 * 4 * 512,
            use_bias=False,
            kernel_initializer=kernel_init,
            name="dense_projection",
        )
    )
    model.add(layers.BatchNormalization(momentum=0.9, name="bn_0"))
    model.add(layers.ReLU(name="relu_0"))
    model.add(layers.Reshape((4, 4, 512), name="reshape_4x4"))

    # 2. Upsample: 4x4x512 -> 8x8x256
    model.add(
        layers.Conv2DTranspose(
            256,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="same",
            use_bias=False,
            kernel_initializer=kernel_init,
            name="conv_transpose_8x8",
        )
    )
    model.add(layers.BatchNormalization(momentum=0.9, name="bn_1"))
    model.add(layers.ReLU(name="relu_1"))

    # 3. Upsample: 8x8x256 -> 16x16x128
    model.add(
        layers.Conv2DTranspose(
            128,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="same",
            use_bias=False,
            kernel_initializer=kernel_init,
            name="conv_transpose_16x16",
        )
    )
    model.add(layers.BatchNormalization(momentum=0.9, name="bn_2"))
    model.add(layers.ReLU(name="relu_2"))

    # 4. Upsample: 16x16x128 -> 32x32x64
    model.add(
        layers.Conv2DTranspose(
            64,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="same",
            use_bias=False,
            kernel_initializer=kernel_init,
            name="conv_transpose_32x32",
        )
    )
    model.add(layers.BatchNormalization(momentum=0.9, name="bn_3"))
    model.add(layers.ReLU(name="relu_3"))

    # 5. Upsample to Final Resolution: 32x32x64 -> 64x64xchannels (Tanh output in [-1, 1])
    model.add(
        layers.Conv2DTranspose(
            channels,
            kernel_size=(4, 4),
            strides=(2, 2),
            padding="same",
            use_bias=False,
            activation="tanh",
            kernel_initializer=kernel_init,
            name="conv_transpose_output_64x64",
        )
    )

    return model
