# Hello World

This simple model takes a string as input and returns a string as output.

## Usage

First, make sure you've got the [latest version of Cog](https://github.com/replicate/cog#install) installed.

Build the container image:

```sh
cog build
```

Now you can run predictions on the model:

```sh
cog predict -i text=Athena

cog predict -i text=Zeus
```