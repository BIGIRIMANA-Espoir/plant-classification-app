
from roboflow import Roboflow
rf = Roboflow(api_key="3xFkq9W0Y5retNvseGk7")
project = rf.workspace("espoirs-workspace-wtosj").project("plant-classification-hnafy")
version = project.version(1)
dataset = version.download("folder")
                