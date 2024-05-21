import base64
import functools
import html
import io
import os
import re
import subprocess
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple

# Import big vision utilities
import big_vision.datasets.jsonl
import big_vision.sharding
import big_vision.utils
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import sentencepiece
import supervision as sv
import tensorflow as tf
import torch
from autodistill.detection import (CaptionOntology, DetectionBaseModel,
                                   DetectionTargetModel)
from autodistill.helpers import load_image
# Import model definition from big_vision
from big_vision.models.proj.paligemma import paligemma
from big_vision.trainers.proj.paligemma import predict_fns
from inference.models.paligemma.paligemma import PaliGemma
from IPython.core.display import HTML, display
from PIL import Image

# Don't let TF use the GPU or TPUs
tf.config.set_visible_devices([], "GPU")
tf.config.set_visible_devices([], "TPU")

import os
import sys

# Create the ~/.cache/autodistill directory if it doesn't exist
original_dir = os.getcwd()
autodistill_dir = os.path.expanduser("~/.cache/autodistill")
os.makedirs(autodistill_dir, exist_ok=True)

os.chdir(autodistill_dir)

# TPUs with
if "COLAB_TPU_ADDR" in os.environ:
    raise "It seems you are using Colab with remote TPUs which is not supported."

# Fetch big_vision repository if python doesn't know about it and install
# dependencies needed for this notebook.
if not os.path.exists("big_vision_repo"):
    subprocess.run(
        [
            "git",
            "clone",
            "--quiet",
            "--branch=main",
            "--depth=1",
            "https://github.com/google-research/big_vision",
            "big_vision_repo",
        ]
    )

# Append big_vision code to python import path
if "big_vision_repo" not in sys.path:
    sys.path.append(os.path.join(autodistill_dir, "big_vision_repo"))


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


def preprocess_image(image, size=224):
    # Model has been trained to handle images of different aspects ratios
    # resized to 224x224 in the range [-1, 1]. Bilinear and antialias resize
    # options are helpful to improve quality in some tasks.
    image = np.asarray(image)
    if image.ndim == 2:  # Convert image without last channel into greyscale.
        image = np.stack((image,) * 3, axis=-1)
    image = image[..., :3]  # Remove alpha layer.
    assert image.shape[-1] == 3

    image = tf.constant(image)
    image = tf.image.resize(image, (size, size), method="bilinear", antialias=True)
    return image.numpy() / 127.5 - 1.0  # [0, 255]->[-1,1]


def preprocess_tokens(prefix, suffix=None, seqlen=None, tokenizer=None):
    # Model has been trained to handle tokenized text composed of a prefix with
    # full attention and a suffix with causal attention.
    separator = "\n"
    tokens = tokenizer.encode(prefix, add_bos=True) + tokenizer.encode(separator)
    mask_ar = [0] * len(tokens)  # 0 to use full attention for prefix.
    mask_loss = [0] * len(tokens)  # 0 to not use prefix tokens in the loss.

    if suffix:
        suffix = tokenizer.encode(suffix, add_eos=True)
        tokens += suffix
        mask_ar += [1] * len(suffix)  # 1 to use causal attention for suffix.
        mask_loss += [1] * len(suffix)  # 1 to use suffix tokens in the loss.

    mask_input = [1] * len(tokens)  # 1 if its a token, 0 if padding.
    if seqlen:
        padding = [0] * max(0, seqlen - len(tokens))
        tokens = tokens[:seqlen] + padding
        mask_ar = mask_ar[:seqlen] + padding
        mask_loss = mask_loss[:seqlen] + padding
        mask_input = mask_input[:seqlen] + padding

    return jax.tree.map(np.array, (tokens, mask_ar, mask_loss, mask_input))


def postprocess_tokens(tokens, tokenizer=None):
    tokens = tokens.tolist()  # np.array to list[int]
    try:  # Remove tokens at and after EOS if any.
        eos_pos = tokens.index(tokenizer.eos_id())
        tokens = tokens[:eos_pos]
    except ValueError:
        pass
    return tokenizer.decode(tokens)


@dataclass
class PaliGemmaTrainer(DetectionTargetModel):
    def __init__(self, model_path, tokenizer_path):
        backend = jax.lib.xla_bridge.get_backend()
        model_config = ml_collections.FrozenConfigDict(
            {
                "llm": {"vocab_size": 257_152},
                "img": {
                    "variant": "So400m/14",
                    "pool_type": "none",
                    "scan": True,
                    "dtype_mm": "float16",
                },
            }
        )
        model = paligemma.Model(**self.model_config)
        tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)

        # Load params - this can take up to 1 minute in T4 colabs.
        params = paligemma.load(None, model_path, model_config)

        # Define `decode` function to sample outputs from the model.
        decode_fn = predict_fns.get_all(model)["decode"]
        decode = functools.partial(
            self.decode_fn, devices=jax.devices(), eos_token=tokenizer.eos_id()
        )

        # Create a pytree mask of the trainable params.
        def is_trainable_param(name, param):  # pylint: disable=unused-argument
            if name.startswith("llm/layers/attn/"):
                return True
            if name.startswith("llm/"):
                return False
            if name.startswith("img/"):
                return False
            raise ValueError(f"Unexpected param name {name}")

        self.trainable_mask = big_vision.utils.tree_map_with_names(
            is_trainable_param, params
        )

        #
        # If more than one device is available (e.g. multiple GPUs) the parameters can
        # be sharded across them to reduce HBM usage per device.
        mesh = jax.sharding.Mesh(jax.devices(), ("data"))

        self.data_sharding = jax.sharding.NamedSharding(
            mesh, jax.sharding.PartitionSpec("data")
        )

        params_sharding = big_vision.sharding.infer_sharding(
            params, strategy=[(".*", 'fsdp(axis="data")')], mesh=mesh
        )

        # Yes: Some donated buffers are not usable.
        warnings.filterwarnings(
            "ignore", message="Some donated buffers were not usable"
        )

        @functools.partial(jax.jit, donate_argnums=(0,), static_argnums=(1,))
        def maybe_cast_to_f32(params, trainable):
            return jax.tree.map(
                lambda p, m: p.astype(jnp.float32) if m else p, params, trainable
            )

        # Loading all params in simultaneous - albeit much faster and more succinct -
        # requires more RAM than the T4 colab runtimes have by default (12GB RAM).
        # Instead we do it param by param.
        params, treedef = jax.tree.flatten(params)
        sharding_leaves = jax.tree.leaves(params_sharding)
        trainable_leaves = jax.tree.leaves(self.trainable_mask)
        for idx, (sharding, trainable) in enumerate(
            zip(sharding_leaves, trainable_leaves)
        ):
            params[idx] = big_vision.utils.reshard(params[idx], sharding)
            params[idx] = maybe_cast_to_f32(params[idx], trainable)
            params[idx].block_until_ready()
            params = jax.tree.unflatten(treedef, params)

        # Print params to show what the model is made of.
        def parameter_overview(params):
            for path, arr in big_vision.utils.tree_flatten_with_names(params)[0]:
                print(f"{path:80s} {str(arr.shape):22s} {arr.dtype}")

        print(" == Model params == ")
        parameter_overview(params)

        self.tokenizer = tokenizer
        self.model = model

    def predict(self, input: str, confidence=0.5) -> sv.Detections:
        pass

    def train(self, dataset, epochs=300):
        # @title Function to iterate over train and validation examples.
        SEQLEN = 128

        # TODO: Consider data iterators skipping big_vision and tf.data?
        train_dataset = big_vision.datasets.jsonl.DataSource(
            os.path.join(dataset, "dataset/_annotations.train.jsonl"),
            fopen_keys={"image": f"{dataset.location}/dataset"},
        )

        val_dataset = big_vision.datasets.jsonl.DataSource(
            os.path.join(dataset, "dataset/_annotations.valid.jsonl"),
            fopen_keys={"image": f"{dataset.location}/dataset"},
        )

        def train_data_iterator():
            """Never ending iterator over training examples."""
            # Shuffle examples and repeat so one can train for many epochs.
            dataset = train_dataset.get_tfdata().shuffle(1_000).repeat()
            for example in dataset.as_numpy_iterator():
                image = Image.open(io.BytesIO(example["image"]))
                image = preprocess_image(image)

                prefix = example["prefix"].decode().lower()
                suffix = example["suffix"].decode().lower()
                tokens, mask_ar, mask_loss, _ = preprocess_tokens(
                    prefix, suffix, SEQLEN
                )
                label, _, _, _ = preprocess_tokens(suffix, seqlen=SEQLEN)

                yield {
                    "image": np.asarray(image),
                    "text": np.asarray(tokens),
                    "label": np.asarray(label),
                    "mask_ar": np.asarray(mask_ar),
                    "mask_loss": np.asarray(mask_loss),
                }

        def validation_data_iterator():
            """Single iterator over validation examples."""
            for example in val_dataset.get_tfdata(ordered=True).as_numpy_iterator():
                image = Image.open(io.BytesIO(example["image"]))
                image = self.preprocess_image(image)

                prefix = example["prefix"].decode().lower()
                suffix = example["suffix"].decode().lower()
                tokens, mask_ar, _, mask_input = preprocess_tokens(
                    prefix, seqlen=SEQLEN
                )
                label, _, _, _ = preprocess_tokens(suffix, seqlen=SEQLEN)

                yield {
                    "image": np.asarray(image),
                    "text": np.asarray(tokens),
                    "label": np.asarray(label),
                    "mask_ar": np.asarray(mask_ar),
                    "mask_input": np.asarray(mask_input),
                }

        @functools.partial(jax.jit, donate_argnums=(0,))
        def update_fn(params, batch, learning_rate):
            imgs, txts, mask_ar = batch["image"], batch["text"], batch["mask_ar"]

            def loss_fn(params):
                text_logits, _ = self.model.apply(
                    {"params": params}, imgs, txts[:, :-1], mask_ar[:, :-1], train=True
                )
                logp = jax.nn.log_softmax(text_logits, axis=-1)

                # The model takes as input txts[:, :-1] but the loss is defined as predicting
                # next tokens txts[:, 1:]. Additionally, mask_loss[:, 1:] indicates which tokens
                # are part of the loss (e.g. prefix and padded tokens are not included).
                mask_loss = batch["mask_loss"][:, 1:]
                targets = jax.nn.one_hot(txts[:, 1:], text_logits.shape[-1])

                # Compute the loss per example. i.e. the mean of per token pplx.
                # Since each example has a different number of tokens we normalize it.
                token_pplx = jnp.sum(logp * targets, axis=-1)  # sum across vocab_size.
                example_loss = -jnp.sum(
                    token_pplx * mask_loss, axis=-1
                )  # sum across seq_len.
                example_loss /= jnp.clip(
                    jnp.sum(mask_loss, -1), 1
                )  # weight by num of tokens.

                # batch_loss: mean of per example loss.
                return jnp.mean(example_loss)

            loss, grads = jax.value_and_grad(loss_fn)(params)

            # Apply gradients to trainable params using SGD.
            def apply_grad(param, gradient, trainable):
                if not trainable:
                    return param
                return param - learning_rate * gradient

            params = jax.tree_util.tree_map(
                apply_grad, params, grads, self.trainable_mask
            )

            return params, loss

        # Evaluation/inference loop.
        def make_predictions(
            data_iterator,
            *,
            num_examples=None,
            batch_size=4,
            seqlen=SEQLEN,
            sampler="greedy",
        ):
            outputs = []
            while True:
                # Construct a list of examples in the batch.
                examples = []
                try:
                    for _ in range(batch_size):
                        examples.append(next(data_iterator))
                        examples[-1]["_mask"] = np.array(
                            True
                        )  # Indicates true example.
                except StopIteration:
                    if len(examples) == 0:
                        return outputs

                # Not enough examples to complete a batch. Pad by repeating last example.
                while len(examples) % batch_size:
                    examples.append(dict(examples[-1]))
                    examples[-1]["_mask"] = np.array(
                        False
                    )  # Indicates padding example.

                # Convert list of examples into a dict of np.arrays and load onto devices.
                batch = jax.tree.map(lambda *x: np.stack(x), *examples)
                batch = big_vision.utils.reshard(batch, self.data_sharding)

                # Make model predictions
                tokens = self.decode(
                    {"params": params},
                    batch=batch,
                    max_decode_len=seqlen,
                    sampler=sampler,
                )

                # Fetch model predictions to device and detokenize.
                tokens, mask = jax.device_get((tokens, batch["_mask"]))
                tokens = tokens[mask]  # remove padding examples.
                labels = [postprocess_tokens(e["label"]) for e in examples]
                responses = [postprocess_tokens(t) for t in tokens]

                # Append to html output.
                for example, label, response in zip(examples, labels, responses):
                    outputs.append((example["image"], label, response))
                    if num_examples and len(outputs) >= num_examples:
                        return outputs

        BATCH_SIZE = 8
        TRAIN_EXAMPLES = 128
        LEARNING_RATE = 0.01

        TRAIN_STEPS = TRAIN_EXAMPLES // BATCH_SIZE
        EVAL_STEPS = TRAIN_STEPS // 8

        train_data_it = train_data_iterator()

        sched_fn = big_vision.utils.create_learning_rate_schedule(
            total_steps=TRAIN_STEPS + 1,
            base=LEARNING_RATE,
            decay_type="cosine",
            warmup_percent=0.10,
        )

        for step in range(1, TRAIN_STEPS + 1):
            # Make list of N training examples.
            examples = [next(train_data_it) for _ in range(BATCH_SIZE)]

            # Convert list of examples into a dict of np.arrays and load onto devices.
            batch = jax.tree.map(lambda *x: np.stack(x), *examples)
            batch = big_vision.utils.reshard(batch, data_sharding)

            # Training step and report training loss
            learning_rate = sched_fn(step)
            params, loss = update_fn(params, batch, learning_rate)

            loss = jax.device_get(loss)
            print(
                f"step: {step:2d}/{TRAIN_STEPS:2d}   lr: {learning_rate:.5f}   loss: {loss:.4f}"
            )

            if step == 1 or (step % EVAL_STEPS) == 0:
                print(f"Model predictions at step {step}")
                html_out = ""
                for image, _, caption in make_predictions(
                    validation_data_iterator(), num_examples=4, batch_size=4
                ):
                    display(HTML(html_out))

        flat, _ = big_vision.utils.tree_flatten_with_names(params)

        with open("fine-tuned-paligemma-3b-pt-224.f16.npz", "wb") as f:
            np.savez(f, **{k: v for k, v in flat})

        print("Model saved to ./fine-tuned-paligemma-3b-pt-224.f16.npz")

    def predict(self, input:str, confidence=0.5) -> sv.Detections:
        raise NotImplementedError("This method is not implemented yet.")