import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import supervision as sv
import torch
from autodistill.detection import CaptionOntology, DetectionBaseModel
from autodistill.helpers import load_image
from inference.models.paligemma.paligemma import PaliGemma


def from_pali_gemma(
    response: str,
    resolution_wh: Tuple[int, int],
    class_list: Optional[List[str]] = None,
) -> sv.Detections:
    _SEGMENT_DETECT_RE = re.compile(
        r"(.*?)"
        + r"<loc(\d{4})>" * 4
        + r"\s*"
        + "(?:%s)?" % (r"<seg(\d{3})>" * 16)
        + r"\s*([^;<>]+)? ?(?:; )?",
    )

    width, height = resolution_wh
    xyxy_list = []
    class_name_list = []

    while response:
        m = _SEGMENT_DETECT_RE.match(response)
        if not m:
            break

        gs = list(m.groups())
        before = gs.pop(0)
        name = gs.pop()
        y1, x1, y2, x2 = [int(x) / 1024 for x in gs[:4]]
        y1, x1, y2, x2 = map(round, (y1 * height, x1 * width, y2 * height, x2 * width))

        content = m.group()
        if before:
            response = response[len(before) :]
            content = content[len(before) :]

        xyxy_list.append([x1, y1, x2, y2])
        class_name_list.append(name.strip())
        response = response[len(content) :]

    xyxy = np.array(xyxy_list)
    class_name = np.array(class_name_list)

    if class_list is None:
        class_id = None
    else:
        class_id = np.array([class_list.index(name) for name in class_name])

    return sv.Detections(xyxy=xyxy, class_id=class_id, data={"class_name": class_name})


HOME = os.path.expanduser("~")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class PaliGemma(DetectionBaseModel):
    ontology: CaptionOntology

    def __init__(self, ontology: CaptionOntology):
        self.model = PaliGemma()
        self.ontology = ontology

    def predict(self, input: str, confidence: int = 0.5) -> sv.Detections:
        image = load_image(input)

        prompt = f"detect " + ";".join(self.ontology.classes)

        response = self.model.predict(image, prompt)[0]

        detections = from_pali_gemma(response, image.size, self.ontology.classes)
        detections = detections[detections.confidence > confidence]

        return detections
