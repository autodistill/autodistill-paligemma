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

[PaLiGemma](https://blog.roboflow.com/paligemma-multimodal-vision/), developed by Google, is a computer vision model trained using pairs of images and text. You can fine-tune custom PaLiGemma models for use in computer vision workflows with Autodistill PaLiGemma.

Read the full [Autodistill documentation](https://autodistill.github.io/autodistill/).

## Installation

To use PaLiGemma with autodistill, you need to install the following dependency:

```bash
pip3 install autodistill-paligemma
```

## Quickstart

```python
from autodistill_paligemma import PaLiGemma

target_model = PaLiGemma()

# train a model
target_model.train("./context_images_labeled/data.yaml", epochs=200)

# run inference on the new model
pred = target_model.predict("./context_images_labeled/train/images/dog-7.jpg", conf=0.01)

print(pred)
```


## License

The model weights for PaLiGemma are licensed under a custom Google license. To learn more, refer to the [Google Gemma Terms of Use](https://ai.google.dev/gemma/terms).

## 🏆 Contributing

We love your input! Please see the core Autodistill [contributing guide](https://github.com/autodistill/autodistill/blob/main/CONTRIBUTING.md) to get started. Thank you 🙏 to all our contributors!