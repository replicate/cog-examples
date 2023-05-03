# Blur

This model applies box blur to an input image.

## Usage

First, make sure you've got the [latest version of Cog](https://github.com/replicate/cog#install) installed.

Build the image:

```sh
cog build
```

Now you can run predictions on the model:

```sh
cog predict -i image=@examples/kodim24.png -i blur=4

cog predict -i image=@examples/kodim24.png -i blur=6
```
