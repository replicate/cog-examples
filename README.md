# Cog example models

This repo contains example machine learning models you can use to try out [Cog](https://github.com/replicate/cog).

Once you've got a working model and want to publish it so others can see it in action, check out [replicate.com/docs](https://replicate.com/docs).

## Examples in this repo

- [hello-world](hello-world) - Takes a string as input and returns a string as output. The simplest possible "model".
- [resnet](resnet) - Classifies images using ResNet. This is a good example of a deep learning model that's small enough to run without a GPU if you need to do that for demos.
- [z-image-turbo](z-image-turbo) - [Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo) image generation model. This is an example of a modern generative model. It requires a GPU, but it's small enough that it's fast to run, so makes for a good demo.
- [blur](blur) - Applies box blur to an input image
- [canary](canary) - Takes a string as input and returns a streaming string output
- [notebook](notebook) - Using a Jupyter Notebook with Cog
- [hello-train](hello-train) - Demonstrates Cog's training API for fine-tuning

