# Hello, Train 🚂

Cog's [training API](https://github.com/replicate/cog/blob/main/docs/training.md) allows you to define a fine-tuning interface for an existing Cog model, so users of the model can bring their own training data to create derivative fune-tuned models. Real-world examples of this API in use include [fine-tuning SDXL with images](https://replicate.com/blog/fine-tune-sdxl) or [fine-tuning Llama 2 with structured text](https://replicate.com/blog/fine-tune-llama-2).

See the [Cog training reference docs](https://github.com/replicate/cog/blob/main/docs/training.md) for more details.

This simple trainable model takes a string as input and returns a string as output.

## Training with Cog

Then you can run it like this:

```console
cog train -i prefix=hello
```

## Creating new fine-tunes with Replicate's API

Check out these guides to learn how to fine-tune models on Replicate:

- [Fine-tune a language model](https://replicate.com/docs/guides/fine-tune-a-language-model)
- [Fine-tune an image model](https://replicate.com/docs/guides/fine-tune-an-image-model)
