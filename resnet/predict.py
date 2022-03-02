from typing import Any

import numpy as np
from cog import BasePredictor, Input, Path
from tensorflow.keras.applications.resnet50 import (
    ResNet50,
    decode_predictions,
    preprocess_input,
)
from tensorflow.keras.preprocessing import image as keras_image


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = ResNet50(weights="resnet50_weights_tf_dim_ordering_tf_kernels.h5")

    # Define the arguments and types the model takes as input
    def predict(self, image: Path = Input(description="Image to classify")) -> Any:
        """Run a single prediction on the model"""
        # Preprocess the image
        img = keras_image.load_img(image, target_size=(224, 224))
        x = keras_image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        # Run the prediction
        preds = self.model.predict(x)
        # Return the top 3 predictions
        return decode_predictions(preds, top=3)[0]
