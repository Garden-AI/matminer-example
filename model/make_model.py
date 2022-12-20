from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers import composition as cf
from matminer.datasets import load_dataset
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
import pickle as pkl
import mlflow
import mlflow.sklearn

from MatminerFeaturizer import MatminerFeaturizer


def load_data():
    # Load a test dataset from matminer
    data = load_dataset('flla')
    print('Loaded {} rows with {} columns:'.format(len(data), len(data.columns)),
          data.columns.tolist())

    # Get only the minimum energy structure at each composition
    data['composition'] = data['structure'].apply(lambda x: x.composition)
    data['integer_formula'] = data['composition'].apply(lambda x: x.get_integer_formula_and_factor()[0])

    data.sort_values('e_above_hull', ascending=True, inplace=True)
    data.drop_duplicates('integer_formula', keep='first', inplace=True)
    print('Reduced dataset to {} unique compositions.'.format(len(data)))

    data.reset_index(inplace=True, drop=True)
    data.to_pickle('data.pkl')
    mlflow.log_artifact("data.pkl")

    return data


def create_featurizer(data, element_property_preset):
    # Create the featurizer, which will take the composition as input
    mlflow.log_param("element_property_preset", element_property_preset)
    featurizer = MultipleFeaturizer([
          cf.Stoichiometry(),
          cf.ElementProperty.from_preset(element_property_preset),
          cf.ValenceOrbital(props=['frac']),
          cf.IonProperty(fast=True)
    ])

    # Compute the features
    featurizer.set_n_jobs(1)
    X = featurizer.featurize_many(data['composition'])

    with open('featurizer.pkl', 'wb') as fp:
        pkl.dump(featurizer, fp)

    return X


def fit(X, data):
    # Make the model
    model = Pipeline([
        ('imputer', SimpleImputer()),
        ('model', RandomForestRegressor())
    ])
    model.fit(X, data['formation_energy_per_atom'])
    print('Trained a RandomForest model')

    mlflow.sklearn.log_model(model, "model",
                             extra_pip_requirements=["pymatgen", "matminer"],
                             registered_model_name="Matminer")


if __name__ == "__main__":
    with mlflow.start_run():
        data = load_data()
        x = create_featurizer(data, element_property_preset="magpie")

        artifacts = {"featurizer": "featurizer.pkl"}
        mlflow.pyfunc.log_model(
            artifact_path="featurizer",
            python_model=MatminerFeaturizer(),
            artifacts=artifacts,
            registered_model_name="Matminer Featurizer"
        )

        fit(x, data)
