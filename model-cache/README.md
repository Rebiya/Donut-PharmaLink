---
language: en
tags:
- donut
- vision-encoder-decoder
- medical-ocr
- handwriting-recognition
- document-understanding
license: mit
base_model: naver-clova-ix/donut-base
datasets:
- chinmays18/medical-prescription-dataset
metrics:
- character_accuracy
- word_accuracy
widget:
- text: "Extract text from medical prescription image"
---

# Medical Prescription OCR

## Model Description

This model is a fine-tuned version of [naver-clova-ix/donut-base](https://huggingface.co/naver-clova-ix/donut-base) specifically trained to recognize handwritten medical prescriptions. It combines state-of-the-art OCR capabilities with domain-specific training to accurately extract text from doctor's handwritten notes.

### Key Features
- Specialized for medical prescription text extraction
- Handles various handwriting styles
- Trained with gradual augmentation strategy for robustness
- Includes integrated classification to identify prescription documents

## Intended Use

This model is designed for:
- Research in medical document digitization
- Educational projects in healthcare technology
- Proof-of-concept applications for prescription processing

**Important**: This model is NOT validated for clinical use and should not be used for actual medical diagnosis or prescription verification.

## Training Details

### Architecture
- **Base Model**: NAVER Clova's Donut (Document Understanding Transformer)
- **Type**: Vision Encoder-Decoder model
- **Framework**: PyTorch Lightning

### Training Strategy
The model was trained using a gradual augmentation approach:
1. **Early epochs**: Basic augmentations (slight rotations, brightness adjustments)
2. **Later epochs**: Advanced augmentations (perspective transforms, shadows, motion blur)

This strategy helps the model learn clean patterns first, then adapt to more challenging variations.

### Performance Metrics
- **Character-level Accuracy**: 71%
- **Word-level Accuracy**: 84%

## How to Use

```python
from transformers import DonutProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

# Load the model and processor
processor = DonutProcessor.from_pretrained("chinmays18/medical-prescription-ocr")
model = VisionEncoderDecoderModel.from_pretrained("chinmays18/medical-prescription-ocr")

# Move to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Process an image
image = Image.open("prescription.jpg").convert("RGB")
pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)

# Generate text
task_prompt = "<s_ocr>"
decoder_input_ids = processor.tokenizer(task_prompt, return_tensors="pt").input_ids.to(device)

generated_ids = model.generate(
    pixel_values,
    decoder_input_ids=decoder_input_ids,
    max_length=512,
    num_beams=1,
    early_stopping=True
)

# Decode the generated text
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(generated_text)
```

## Links

- 🔗 **GitHub Repository**: [JonSnow1807/medical-prescription-ocr](https://github.com/JonSnow1807/medical-prescription-ocr)
- 📊 **Dataset**: [chinmays18/medical-prescription-dataset](https://huggingface.co/datasets/chinmays18/medical-prescription-dataset)
- 🤗 **Demo**: Available in the GitHub repository

## Limitations and Biases

1. **Language**: Primarily trained on English prescriptions
2. **Handwriting Styles**: Performance varies with handwriting quality
3. **Medical Terminology**: May struggle with rare drug names or abbreviations
4. **Image Quality**: Best results with clear, well-lit images

## Ethical Considerations

- This model should not be used for actual medical diagnosis or prescription verification
- Always have medical professionals verify any extracted prescription information
- Be aware of patient privacy when processing medical documents

## Citation

If you use this model in your research, please cite:

```bibtex
@misc{shrivastava2024medicalocr,
  author = {Chinmay Shrivastava},
  title = {Medical Prescription OCR},
  year = {2024},
  publisher = {Hugging Face},
  journal = {Hugging Face Model Hub},
  url = {https://huggingface.co/chinmays18/medical-prescription-ocr}
}
```

## Acknowledgments

Base architecture: NAVER Clova AI's Donut team
Training framework: PyTorch Lightning
Dataset inspiration: IAM Handwriting Database


### 3. Update the Tags
Make sure to also update the `datasets` field from `custom-iam-medical` to `chinmays18/medical-prescription-dataset` to properly link to your dataset.