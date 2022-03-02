# resnet

This model classifies images.

## Usage

✋ Note for M1 Mac users: This model uses TensorFlow, which does not currently work on M1 machines using Docker. See [replicate/cog#336](https://github.com/replicate/cog/issues/336) for more information.

---

First, make sure you've got the [latest version of Cog](https://github.com/replicate/cog#install) installed.

Download the pre-trained weights:

```
curl -O https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet50_weights_tf_dim_ordering_tf_kernels.h5
```

Build the image:

```sh
cog build
```

Now you can run predictions on the model:

```sh
cog predict -i image=@cat.png
```