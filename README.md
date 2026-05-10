# Kazakh OCR: Synthetic Data Generation & Fine-Tuning

This repository provides a complete pipeline for generating synthetic Kazakh printed text data and fine-tuning a TrOCR model for high-accuracy Optical Character Recognition (OCR).

## 📂 Project Structure

* **`SyntheticDataGeneration/`**: Contains the notebook used to create the synthetic dataset and the font files used during generation.
    * **Dataset:** [thekamilya/kazakh-printed-dataset](https://huggingface.co/datasets/thekamilya/kazakh-printed-dataset)
* **`FineTuning/`**: Contains the notebook for fine-tuning the base model on Kazakh text.
    * **Model:** [thekamilya/kazakh-trocr-fine-tuned](https://huggingface.co/thekamilya/kazakh-trocr-fine-tuned)

---

## 🛠 Generation Process

The synthetic dataset was derived from the **issai/kazparc** corpus and processed with the following augmentations to ensure model robustness:

1. **Source:** Text samples were extracted from the ISSAI KazParc dataset.
2. **Color Schemes:** Simulates varied lighting by alternating between "light mode" (black on white) and "dark mode" (white on black).
3. **Dynamic Geometry:** Automatically calculates canvas width based on text length for realistic aspect ratios.
4. **Stylistic Variety:** Randomly selects typefaces with font sizes varying between 40 and 60 points.
5. **Spatial Jitter:** Centers text with random horizontal/vertical offsets to mimic human inconsistency.
6. **Physical Augmentations:**
    * **Rotation:** Tilts images by $\pm5^\circ$ to simulate misaligned scans.
    * **Blur:** Applies Gaussian blur to mimic out-of-focus captures.
    * **Noise:** Injects digital grain to simulate low-light sensor noise.

---

## 📊 Results

The model was fine-tuned using [kazars24/trocr-base-handwritten-ru](https://huggingface.co/kazars24/trocr-base-handwritten-ru) as the base. The table below shows the performance improvement before and after fine-tuning.

| Metric | Before Fine-tuning | After Fine-tuning |
| :--- | :---: | :---: |
| **CER (Character Error Rate)** | 67.1 | **3.7** |
| **Exact Match (%)** | 0.0% | **48.62%** |

### Metric Definitions
* **CER (Character Error Rate):** Measures the distance between predicted and ground truth text normalized by length. **Lower is better.**
* **Exact Match:** The percentage of predictions that perfectly match the ground truth. **Higher is better.**

---

## 🚀 Getting Started

You can use the fine-tuned model directly from the Hugging Face Hub:

```python
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

# Load the fine-tuned Kazakh TrOCR model
processor = TrOCRProcessor.from_pretrained("thekamilya/kazakh-trocr-fine-tuned")
model = VisionEncoderDecoderModel.from_pretrained("thekamilya/kazakh-trocr-fine-tuned")

# Load and prepare image
image = Image.open("your_image.jpg").convert("RGB")
pixel_values = processor(images=image, return_tensors="pt").pixel_values

# Inference
generated_ids = model.generate(pixel_values)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(f"Recognized Text: {generated_text}")



