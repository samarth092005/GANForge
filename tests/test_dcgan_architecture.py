"""Unit tests for DCGAN Generator, Discriminator, Loss functions, and Gradients.

Verifies mathematical validity, layer dimensions, parameter counts,
gradient backpropagation, and loss finiteness without executing training.
"""
from __future__ import annotations

import unittest
import numpy as np
import tensorflow as tf

from config.settings import (
    CHANNELS,
    IMAGE_SIZE,
    LATENT_DIM,
    LEARNING_RATE_DISC,
    LEARNING_RATE_GEN,
)
from models.discriminator import build_discriminator
from models.generator import build_generator
from models.losses import discriminator_loss, generator_loss, get_optimizers


class TestDCGANArchitecture(unittest.TestCase):
    """Test suite for DCGAN Generator, Discriminator, losses, and gradients."""

    @classmethod
    def setUpClass(cls) -> None:
        """Instantiate Generator and Discriminator once for the test suite."""
        cls.generator = build_generator(latent_dim=LATENT_DIM, channels=CHANNELS)
        cls.discriminator = build_discriminator(
            image_size=IMAGE_SIZE, channels=CHANNELS, dropout_rate=0.3
        )
        cls.batch_size = 4

    # ---------------------------------------------------------
    # 1. Generator Tests
    # ---------------------------------------------------------
    def test_generator_input_and_output_shapes(self) -> None:
        """Verify Generator accepts (batch, 100) and produces (batch, 64, 64, 3)."""
        noise = tf.random.normal([self.batch_size, LATENT_DIM])
        generated_images = self.generator(noise, training=False)

        expected_shape = (self.batch_size, IMAGE_SIZE[0], IMAGE_SIZE[1], CHANNELS)
        self.assertEqual(generated_images.shape, expected_shape)

    def test_generator_output_range(self) -> None:
        """Verify Generator output pixels are strictly bounded in [-1.0, 1.0] by Tanh."""
        noise = tf.random.normal([self.batch_size, LATENT_DIM])
        generated_images = self.generator(noise, training=False)

        min_val = tf.reduce_min(generated_images).numpy()
        max_val = tf.reduce_max(generated_images).numpy()

        self.assertGreaterEqual(min_val, -1.0)
        self.assertLessEqual(max_val, 1.0)

    def test_generator_trainable_parameters(self) -> None:
        """Verify Generator has non-zero trainable variables and parameters."""
        trainable_vars = self.generator.trainable_variables
        self.assertGreater(len(trainable_vars), 0)

        total_params = self.generator.count_params()
        self.assertGreater(total_params, 1_000_000)

    # ---------------------------------------------------------
    # 2. Discriminator Tests
    # ---------------------------------------------------------
    def test_discriminator_input_and_output_shapes(self) -> None:
        """Verify Discriminator accepts (batch, 64, 64, 3) and produces (batch, 1) logits."""
        dummy_images = tf.random.uniform(
            [self.batch_size, IMAGE_SIZE[0], IMAGE_SIZE[1], CHANNELS],
            minval=-1.0,
            maxval=1.0,
        )
        logits = self.discriminator(dummy_images, training=False)

        self.assertEqual(logits.shape, (self.batch_size, 1))

    def test_discriminator_produces_unactivated_logits(self) -> None:
        """Verify Discriminator produces real-valued unconstrained logits (not sigmoid)."""
        # Discriminator output layer must have no activation function (linear logits)
        output_layer = self.discriminator.layers[-1]
        self.assertIsInstance(output_layer, tf.keras.layers.Dense)
        self.assertEqual(output_layer.units, 1)
        self.assertEqual(output_layer.activation, tf.keras.activations.linear)

    def test_discriminator_trainable_parameters(self) -> None:
        """Verify Discriminator has non-zero trainable variables and parameters."""
        trainable_vars = self.discriminator.trainable_variables
        self.assertGreater(len(trainable_vars), 0)

        total_params = self.discriminator.count_params()
        self.assertGreater(total_params, 1_000_000)

    # ---------------------------------------------------------
    # 3. Forward Pass & Pipeline Compatibility
    # ---------------------------------------------------------
    def test_end_to_end_forward_pass(self) -> None:
        """Verify generated images can be piped directly into the Discriminator."""
        noise = tf.random.normal([self.batch_size, LATENT_DIM])
        fake_images = self.generator(noise, training=False)
        fake_logits = self.discriminator(fake_images, training=False)

        self.assertEqual(fake_logits.shape, (self.batch_size, 1))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(fake_logits)))

    # ---------------------------------------------------------
    # 4. Gradient Calculation Tests
    # ---------------------------------------------------------
    def test_generator_gradients_computable(self) -> None:
        """Verify non-zero gradients can be backpropagated through Generator."""
        noise = tf.random.normal([self.batch_size, LATENT_DIM])

        with tf.GradientTape() as tape:
            fake_images = self.generator(noise, training=True)
            fake_logits = self.discriminator(fake_images, training=False)
            loss_g = generator_loss(fake_logits)

        gradients_g = tape.gradient(loss_g, self.generator.trainable_variables)

        # Ensure all variables have valid non-None gradients
        self.assertEqual(len(gradients_g), len(self.generator.trainable_variables))
        for grad, var in zip(gradients_g, self.generator.trainable_variables):
            self.assertIsNotNone(grad, f"Gradient for {var.name} was None!")
            self.assertTrue(
                tf.reduce_all(tf.math.is_finite(grad)),
                f"Gradient for {var.name} contained NaN or Inf!",
            )

    def test_discriminator_gradients_computable(self) -> None:
        """Verify non-zero gradients can be backpropagated through Discriminator."""
        real_images = tf.random.uniform(
            [self.batch_size, IMAGE_SIZE[0], IMAGE_SIZE[1], CHANNELS],
            minval=-1.0,
            maxval=1.0,
        )
        noise = tf.random.normal([self.batch_size, LATENT_DIM])
        fake_images = self.generator(noise, training=False)

        with tf.GradientTape() as tape:
            real_logits = self.discriminator(real_images, training=True)
            fake_logits = self.discriminator(fake_images, training=True)
            loss_d, _, _ = discriminator_loss(real_logits, fake_logits)

        gradients_d = tape.gradient(loss_d, self.discriminator.trainable_variables)

        self.assertEqual(len(gradients_d), len(self.discriminator.trainable_variables))
        for grad, var in zip(gradients_d, self.discriminator.trainable_variables):
            self.assertIsNotNone(grad, f"Gradient for {var.name} was None!")
            self.assertTrue(
                tf.reduce_all(tf.math.is_finite(grad)),
                f"Gradient for {var.name} contained NaN or Inf!",
            )

    # ---------------------------------------------------------
    # 5. Loss Functions & Optimizers
    # ---------------------------------------------------------
    def test_loss_functions_return_finite_scalars(self) -> None:
        """Verify generator_loss and discriminator_loss compute finite positive scalars."""
        real_logits = tf.constant([[1.5], [0.8], [2.1], [0.2]])
        fake_logits = tf.constant([[-1.2], [-0.5], [-2.0], [0.1]])

        loss_g = generator_loss(fake_logits)
        loss_d, real_loss, fake_loss = discriminator_loss(real_logits, fake_logits)

        self.assertTrue(tf.math.is_finite(loss_g))
        self.assertTrue(tf.math.is_finite(loss_d))
        self.assertTrue(tf.math.is_finite(real_loss))
        self.assertTrue(tf.math.is_finite(fake_loss))

        self.assertGreater(float(loss_g.numpy()), 0.0)
        self.assertGreater(float(loss_d.numpy()), 0.0)
        self.assertAlmostEqual(
            float(loss_d.numpy()), float((real_loss + fake_loss).numpy()), places=5
        )

    def test_label_smoothing_effect(self) -> None:
        """Verify label smoothing softens target without affecting fake loss."""
        real_logits = tf.constant([[2.0], [2.0]])
        fake_logits = tf.constant([[-2.0], [-2.0]])

        _, real_loss_hard, fake_loss_1 = discriminator_loss(
            real_logits, fake_logits, label_smoothing=0.0
        )
        _, real_loss_smooth, fake_loss_2 = discriminator_loss(
            real_logits, fake_logits, label_smoothing=0.1
        )

        # Label smoothing on real target increases real loss slightly (since target is 0.9 instead of 1.0)
        self.assertGreater(float(real_loss_smooth.numpy()), float(real_loss_hard.numpy()))
        # Fake loss must remain identical
        self.assertAlmostEqual(
            float(fake_loss_1.numpy()), float(fake_loss_2.numpy()), places=5
        )

    def test_optimizers_configuration(self) -> None:
        """Verify Adam optimizers are properly instantiated with DCGAN hyperparameters."""
        gen_opt, disc_opt = get_optimizers(
            learning_rate_gen=LEARNING_RATE_GEN,
            learning_rate_disc=LEARNING_RATE_DISC,
            beta_1=0.5,
            beta_2=0.999,
        )

        self.assertIsInstance(gen_opt, tf.keras.optimizers.Adam)
        self.assertIsInstance(disc_opt, tf.keras.optimizers.Adam)
        self.assertAlmostEqual(float(gen_opt.learning_rate.numpy()), LEARNING_RATE_GEN)
        self.assertAlmostEqual(float(disc_opt.learning_rate.numpy()), LEARNING_RATE_DISC)
        self.assertAlmostEqual(float(gen_opt.beta_1), 0.5)
        self.assertAlmostEqual(float(disc_opt.beta_1), 0.5)


if __name__ == "__main__":
    unittest.main()
