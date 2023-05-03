# Hello World

This simple trainable model takes a string as input and returns a string as output.

## Usage on replicate training api

Kick off a training using the training api

    training = replicate.trainings.create(
      version="username/trainer_name:version_id",
      input={
        "prefix": "hola"
      },
      destination="username/trained_name"
    )

Then you cna use your trained model at https://replicate.com/username/trained_name

