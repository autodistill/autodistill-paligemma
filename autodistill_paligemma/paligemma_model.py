import os
from dataclasses import dataclass
import json
from PIL import Image
import supervision as sv
import torch
from autodistill.detection import (
    CaptionOntology,
    DetectionBaseModel,
    DetectionTargetModel,
)
from autodistill.helpers import load_image
from torch.utils.data import Dataset
import random
from inference.models.paligemma.paligemma import LoRAPaliGemma
from inference.models.paligemma.paligemma import PaliGemma as InferencePaliGemma
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from PIL import Image
from tqdm import tqdm
import random
import json
from peft import LoraConfig, get_peft_model
from torch import optim

HOME = os.path.expanduser("~")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Data(Dataset):
    def __init__(self, jsonl_location):
        with open(jsonl_location, "r") as json_file:
            json_list = list(json_file)

        jsons = [json.loads(json_str) for json_str in json_list]
        random.shuffle(jsons)
        self.jsons = jsons

    def __getitem__(self, index):
        return self.jsons[index]

    def __len__(self):
        return len(self.jsons)

    def shuffle(self):
        random.shuffle(self.jsons)


@dataclass
class PaliGemma(DetectionBaseModel):
    ontology: CaptionOntology

    def __init__(
        self,
        ontology: CaptionOntology,
        model_id: str = "paligemma-3b-mix-224",
        lora_model: bool = False,
    ):
        if lora_model:
            self.model = LoRAPaliGemma(
                model_id,
                huggingface_token=os.environ.get("HF_ACCESS_TOKEN"),
                api_key=os.environ.get("ROBOFLOW_API_KEY"),
            )
        else:
            self.model = InferencePaliGemma(model_id)

        self.ontology = ontology

    def predict(self, input: str, confidence: int = 0.5) -> sv.Detections:
        image = load_image(input)

        prompt = f"detect " + ";".join(self.ontology.classes)

        response = self.model.infer(image, prompt=prompt)[0]

        detections = sv.from_lmm(
            "paligemma", response, image.size, self.ontology.classes
        )
        detections = detections[detections.confidence > confidence]

        return detections


@dataclass
class PaliGemmaTrainer(DetectionTargetModel):
    def __init__(self, model_id="google/paligemma-3b-pt-224"):
        device = DEVICE
        print(device)
        cache_dir = "./paligemma"

        if not os.environ.get("HF_ACCESS_TOKEN"):
            raise ValueError("Please set the HF_ACCESS_TOKEN environment variable")

        processor = AutoProcessor.from_pretrained(
            model_id, token=os.environ.get("HF_ACCESS_TOKEN"), cache_dir=cache_dir
        )
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id,
            revision="float16",
            device_map=device,
            cache_dir=cache_dir,
            token=os.environ.get("HF_ACCESS_TOKEN"),
            torch_dtype=torch.float16,
        ).eval()

        self.processor = processor
        self.model = model
        self.device = device

    def predict(self, input: str, confidence=0.5) -> sv.Detections:
        image = load_image(input)

        tokens = self.processor(
            text=["detect"],
            images=[image],
            return_tensors="pt",
            padding="longest",
        )
        tokens = tokens.to(self.device)

        with torch.no_grad():
            response = self.model.generate(**tokens)

        detections = sv.from_lmm(
            "paligemma", response, image.size, self.ontology.classes
        )
        detections = detections[detections.confidence > confidence]

        return detections

    def train(self, dataset):
        def collate_fn(examples):
            images = [
                Image.open(f"{dataset}/dataset/{example['image']}").convert("RGB")
                for example in examples
            ]
            tokens = self.processor(
                text=[example["prefix"] for example in examples],
                suffix=[example["suffix"] for example in examples],
                images=images,
                return_tensors="pt",
                padding="longest",
            )
            tokens = tokens.to(self.device)
            return tokens

        config = LoraConfig(
            r=12,
            lora_alpha=12,
            target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "linear"],
            task_type="CAUSAL_LM",
            lora_dropout=0.05,
            bias="none",
            inference_mode=False,
            use_rslora=True,
            init_lora_weights="gaussian",
            revision="float16",
        )
        peft_model = get_peft_model(self.model, config).to(self.device).train().cuda()
        peft_model.print_trainable_parameters()

        train_dataset = Data(f"{dataset}/dataset/_annotations.train.jsonl")
        val_dataset = Data(f"{dataset}/dataset/_annotations.valid.jsonl")

        TRAIN_EXAMPLES = len(train_dataset)
        BATCH_SIZE = 3
        LEARNING_RATE = 0.0004
        TRAIN_STEPS = TRAIN_EXAMPLES // BATCH_SIZE
        NUM_EPOCHS = 2
        with torch.amp.autocast("cuda", torch.float16):
            lora_layers = filter(lambda p: p.requires_grad, peft_model.parameters())
            optimizer = optim.SGD(lora_layers, lr=LEARNING_RATE)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, NUM_EPOCHS * TRAIN_STEPS + 1, eta_min=LEARNING_RATE / 10
            )
            for epoch in tqdm(range(NUM_EPOCHS), desc="EPOCHS"):
                train_dataset.shuffle()
                iterator = iter(train_dataset)
                progress_bar = tqdm(range(TRAIN_STEPS), desc="STEPS")
                for step in range(1, TRAIN_STEPS + 1):
                    with torch.no_grad():
                        examples = []
                        for _ in range(BATCH_SIZE):
                            examples.append(next(iterator))

                        batch = collate_fn(examples)

                    l = peft_model(**batch)["loss"]
                    l.backward()
                    loss = l.cpu().detach().numpy()

                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

                    progress_bar.update(1)
                    progress_bar.set_description(f"Loss: {loss}")

        peft_model.save_pretrained("paligemma-lora")
        self.processor.save_pretrained("paligemma-lora/")

        self.peft_model = peft_model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
