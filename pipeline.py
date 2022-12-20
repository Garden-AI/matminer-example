from pymatgen.core import Composition
import mlflow


def parse_compositions(formulas):
    return [Composition(x) for x in formulas]


def featurize_composition(compositions):
    model_name = "Matminer Featurizer"
    model_version = "Production"
    logged_featurizer = f"models:/{model_name}/{model_version}"

    # Load featurizer as a pyfunc model
    loaded_featurizer = mlflow.pyfunc.load_model(logged_featurizer)
    return loaded_featurizer.predict(compositions)


def predict(x):
    model_name = "Matminer"
    model_version = "Production"
    logged_model = f"models:/{model_name}/{model_version}"

    # Load model as a SciKitLearn Model.
    loaded_model = mlflow.sklearn.load_model(logged_model)
    return loaded_model.predict(x)


print(
    predict(
        featurize_composition(
            parse_compositions(["NaCl", "H2O"])
        )
    )
)
