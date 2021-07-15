"""Base Model for all model."""

import tensorflow as tf
import os

MODEL_FILE_NAME = 'model.h5'


class BaseModel(tf.keras.Model):
    def set_config(self, config):
        self.config = config

    def save_pretrained(self, saved_path):
        """save config and weights to file"""
        os.mkdir(saved_path, exist_ok=True)
        self.config.save_pretrained(saved_path)
        self.save_weights(os.path.join(saved_path, MODEL_FILE_NAME))