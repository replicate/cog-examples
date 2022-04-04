# notebook

A simple example using a Jupyter Notebook with Cog

## Usage

First, make sure you've got the [latest version of Cog](https://github.com/replicate/cog#install) installed.

Build the image:

```sh
cog build
```

Run the Jupyter notebook server with Cog:

```sh
cog run -p 8888 --debug jupyter notebook --allow-root --ip=0.0.0.0
```

Copy the notebook URL to your browser. It should look something like this:

```sh
http://127.0.0.1:8888/?token=eedb5f511a60b179d1a1b2c6395f0f20c02a08124bae6896
```

Save any changes you make to your notebook, then export it as a Python script:

```sh
jupyter nbconvert --to script notebook.ipynb # creates notebook.py
```

Now you can run predictions on the model:

```sh
cog predict -i name=Alice
```
