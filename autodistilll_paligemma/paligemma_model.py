import os
from dataclasses import dataclass

import torch

import supervision as sv
from autodistill.detection import CaptionOntology, DetectionTargetModel

HOME = os.path.expanduser("~")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class Model(DetectionTargetModel):
    def __init__(self, model_name):
        # load model
        # if only one variant is available, remove the "model_name" argument
        self.model = ...

    def predict(self, input:str, confidence=0.5) -> sv.Detections:
        pass

    def train(self, dataset_yaml, epochs=300):
        pass