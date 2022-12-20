import mlflow.pyfunc
import pickle as pkl


class MatminerFeaturizer(mlflow.pyfunc.PythonModel):
    def __init__(self):
        self.featureizer = None

    def load_context(self, context):
        with open(context.artifacts["featurizer"], 'rb') as fp:
            self.featureizer = pkl.load(fp)

    def predict(self, context, model_input):
        return self.featureizer.featurize_many(model_input)
