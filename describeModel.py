import json

from garden_ai import GardenClient

client = GardenClient()
garden = client.create_garden(authors=['Ward, Logan'],
                              title="Composition featurizer of Ward et al. 2016")
garden.description = "Formation enthalpy predictor"
garden.language = "English"
garden.doi = '10.1038/npjcompumats.2016.28'

client.register_metadata(garden)

