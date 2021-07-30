import cog
from pathlib import Path
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np


class ResNetPredictor(cog.Predictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = ResNet50(weights='resnet50_weights_tf_dim_ordering_tf_kernels.h5')

    # Define the arguments and types the model takes as input
    @cog.input("input", type=Path, help="Image to classify")
    def predict(self, input):
        """Run a single prediction on the model"""
        # Preprocess the image
        img = image.load_img(input, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        # Run the prediction
        preds = self.model.predict(x)
        # Return the top 3 predictions
        return str(decode_predictions(preds, top=3)[0])
