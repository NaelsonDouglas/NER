# GPS Spancat model

The model is trained and served using the [Spacy CI/CD projects](https://spacy.io/usage/projects) standard.

The available project commands are listed in the [project.yml](https://github.com/NaelsonDouglas/NER/blob/main/src/project.yml) file and the available commands are:

- corpus
  -  In short it prepares the raw data to be used as training and validation data. It reads the annotations stored in assets/big_dataset.csv and converts it to the spaCy binary format.
- train
  - It executes the training pipeline on the previously converted raw dataset. During the training phase the best model will be stored at output/model-best and the last trained model is kept at output/model-last.

- serve
  - It serves the model via a REST API. The default port for the API is 3501, but this value can be configured on the project.yml file on the `vars` field. The api code is written at  scr/scripts/api.py. The main endpoint on the API is the `/extract`, which is a POST endpoint and it's body receives a single field on the following format
  ```json
  {
    "title" : "the listing title goes here"
  }
  ```
  This endpoint returns an object containing the extracted info, like make, model, modelnoq etc.


It is possible to execute the entire pipeline with ```python -m spacy project run --force all```. For more details on the spacy project CLI refer to its [documentation](https://spacy.io/usage/projects).