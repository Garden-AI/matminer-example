import mlflow
from garden_ai import GardenClient, step, Pipeline
from pymatgen.core import Composition

client = GardenClient()
garden = client.create_garden(authors=['Ward, Logan'],
                              title="Composition featurizer of Ward et al. 2016")
garden.description = "Formation enthalpy predictor"
garden.language = "English"
garden.doi = '10.1038/npjcompumats.2016.28'

client.register_metadata(garden)


@step
def parse_compositions(formulas: list[str]) -> list[Composition]:
    return [Composition(x) for x in formulas]


@step
def featurize_composition(compositions: list[Composition]) -> list:
    model_name = "Matminer Featurizer"
    model_version = "Production"
    logged_featurizer = f"models:/{model_name}/{model_version}"

    # Load featurizer as a pyfunc model
    loaded_featurizer = mlflow.pyfunc.load_model(logged_featurizer)
    return loaded_featurizer.predict(compositions)


@step
def predict(x: list) -> list[float]:
    model_name = "Matminer"
    model_version = "Production"
    logged_model = f"models:/{model_name}/{model_version}"

    # Load model as a SciKitLearn Model.
    loaded_model = mlflow.sklearn.load_model(logged_model)
    return loaded_model.predict(x)


predictor_pipeline = Pipeline(
    title="Formation enthalpy predictor from molecule compositions",
    steps=(parse_compositions, featurize_composition, predict),
    authors=garden.authors,
    contributors=["Galewsky, Ben"]
)

garden.pipelines = [predictor_pipeline]

print(predictor_pipeline(["NaCl"]))
