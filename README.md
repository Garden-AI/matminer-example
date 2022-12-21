# matminer-example
Example of Garden based on MatMiner

## Training
Garden is concerned with publishing, sharing, and running models, and it does
not support the training workflow. However, if we use MLFlow as the back end to
the service, running and tracking of training is included at no extra cost.

Nevertheless, scientists won't _have_ to use MLFlow when training their models
for Garden. They can jump right in at the publish step below, using garden-ai
libraries and CLIs.

In this example, the training code is in the [model](model) directory. It conforms to the MLFlow
[Project Directory](https://www.mlflow.org/docs/latest/projects.html#project-directories)
standard.

[MLProject](model/MLProject) file defines the model training workflow. It includes
a reference for the conda or pipenv dependencies, and how to run the different
training steps along with hyperparameters for each step.

### Running a Training Step
I've tested this with conda. You need the following environment variables set
that correspond to our development MLFlow tracking server:
* MLFLOW_TRACKING_URI
* MLFLOW_S3_ENDPOINT_URL
* AWS_ACCESS_KEY_ID
* AWS_SECRET_ACCESS_KEY

```shell
cd model
conda create -n mlflow
conda activate mlflow
python -m pip install "mlflow>=2.0"
mlflow run --experiment-id 5 .
```
This last command will invoke `make_model.py`, which 
1. Loads training data
2. Configures and builds the featurizer
3. Trains a random forest model
4. Publishes the model and the featurizer to the MLFlow model repository.

While currently not exposed, the script is wired to accept a parameter,
`element_property_preset`.

## Publishing the Models
The `make_model.py` script uses the MLflow `log_model` function to attach the 
trained model to the run in the MLFlow tracking server. There is a specific 
version of this function that understands scikit models. We also need to 
publish the featurizer. This is not natively supported by MLflow, but we created
a custom subclass of `PythonModel` that allows us to describe how to read the
Featurizer from its pickle file and then invoke the `featurize_many` method.

We use these two functions to publish the model and featurizer and attach them 
to the run. The SciKit publisher knows how to inspect the model and determine
dependencies required to work with it. These are also published to the 
tracking server as a Conda environment as well as a Pipenv requirements.

We tack the matminer library in as an extra pip install as part of the model
publish. This means it that the conda environment included with the model is 
sufficient to run the prediction pipeline.

For scientists who choose not to use MLFlow for training, we can provide 
wrappers for these functions in the garden-ai libray and the garden CLI.

After the model and featurizer have been published, you can use the MLFlow
tracking server UI to mark a version of these artifacts as _Production_. This 
tagging is used to allow updates to the model and have the prediction code use
the latest version (or stick to a previous version).

## Prediction
The Garden pipeline is located in the root of this repo. I ran `describeModel.py`
to create the Garden object and publish the metadata (currently just stored
as [metadata.json](metadata.json)).

We can use MLFlow to automatically genreate a conda envioronment that can be 
used to run the prediction:
```shell
mlflow models prepare-env --model-uri models:/Matminer/Production --env-manager conda
```

This command can be used to create a conda environment, a pipenv environment, or 
a Docker image. For consistency, I use the conda option.

After activating the generated conda environment you can run a prediction with
the [pipeline.py](pipeline.py) script.

This script attempts to simulate the garden pipeline architecture with three
functions. `featurize_composition` loads the _Production_ version of the 
featurizer. By means of our custom python function class, `MatminerFeaturizer`, 
we can call the `featurize_many` operation on the loaded artifact.
Likewise, we load the trained model from its _Production_ tag.
