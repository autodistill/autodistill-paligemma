<div align="center">
  <p>
    <a align="center" href="" target="_blank">
      <img
        width="850"
        src="https://media.roboflow.com/open-source/autodistill/autodistill-banner.png"
      >
    </a>
  </p>
</div>

# Autodistill PaLiGemma Module

This repository contains the code supporting the PaLiGemma base model for use with [Autodistill](https://github.com/autodistill/autodistill).

[PaLiGemma](https://blog.roboflow.com/paligemma-multimodal-vision/), developed by Google, is a computer vision model trained using pairs of images and text. You can label data with PaliGemma models for use in training smaller, fine-tuned models with Autodisitll.

Read the full [Autodistill documentation](https://autodistill.github.io/autodistill/).

## Installation

To use PaLiGemma with autodistill, you need to install the following dependency:

```bash
pip3 install autodistill-paligemma
```

## Quickstart

### Auto-label with an existing model

```python
from autodistill_paligemma import PaliGemma

# define an ontology to map class names to our PaliGemma prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
# then, load the model
base_model = PaliGemma(
    ontology=CaptionOntology(
        {
            "person": "person",
            "a forklift": "forklift"
        }
    )
)

# label a single image
result = PaliGemma.predict("test.jpeg")
print(result)

# label a folder of images
base_model.label("./context_images", extension=".jpeg")
```

### Model fine-tuning (Coming soon)

```python
from autodistill_paligemma import PaLiGemmaTrainer

target_model = PaLiGemmaTrainer()

# train a model
target_model.train("./data/", epochs=200)
```

Note: You cannot yet used fine-tuned models with Autodistill. This feature is coming soon.


## License

The model weights for PaLiGemma are licensed under a custom Google license. To learn more, refer to the [Google Gemma Terms of Use](https://ai.google.dev/gemma/terms).

## 🏆 Contributing

We love your input! Please see the core Autodistill [contributing guide](https://github.com/autodistill/autodistill/blob/main/CONTRIBUTING.md) to get started. Thank you 🙏 to all our contributors!