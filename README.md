# PixVerNet: Image Authenticity & Generation

PixVerNet combines an EfficientNetV2-based classifier with a Pix2Pix conditional GAN to distinguish real images from AI-generated ones and to synthesize paired translations. The project is trained on the Kaggle **AI vs Human Generated Images** dataset, enabling side-by-side evaluation of authenticity detection and generation quality.

## Key Features
- **Real vs AI Classification:** EfficientNetV2 backbone, trained on ~11 GB of paired examples for robust binary detection.
- **Paired Dataset Utilization:** Balanced real/AI pairs (many with humans) encourage consistent comparative learning.
- **Pix2Pix Image Generation:** Conditional GAN produces AI counterparts for provided inputs to support visual authenticity studies.

## Dataset
| Feature  | Details |
|----------|---------|
| Source   | Kaggle competition: AI vs Human Generated Images |
| Size     | ~11 GB (paired real/synthetic samples) |
| Content  | Shutterstock-authentic photos matched to state-of-the-art AI renders |
| Diversity| ~33% of real samples feature people; synthetic set spans multiple generators |

## Model Architecture
### Classification
| Component | Detail |
|-----------|--------|
| Backbone  | EfficientNetV2 |
| Task      | Binary (Real vs AI) |
| Loss      | Binary Cross-Entropy |
| Optimizer | AdamW |
| Training  | Mixed precision optional |

### Generation
| Component | Detail |
|-----------|--------|
| Model     | Pix2Pix (Conditional GAN) |
| Generator | U-Net |
| Discriminator | PatchGAN |
| Input     | Real image |
| Output    | AI-generated counterpart |

## Project Structure
```
PixVerNet/
├── classification/
│   ├── train_classifier.py
│   ├── dataset_loader.py
│   ├── model_efficientnetv2.py
│   └── inference.py
├── generation/
│   ├── train_pix2pix.py
│   ├── generator_unet.py
│   ├── discriminator_patchgan.py
│   └── generate.py
├── notebooks/
│   └── exploratory_data_analysis.ipynb
├── utils/
│   ├── transforms.py
│   └── helpers.py
├── README.md
└── requirements.txt
```

## Installation
1. **Clone**
   ```bash
   git clone https://github.com/hardattmangrola/PixVerNet.git
   cd PixVerNet
   ```
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
3. **Download dataset**
   - Grab the Kaggle **AI vs Human Generated Images** dataset.
   - Extract to `./data/ai_vs_human/`.

## Training
- **EfficientNetV2 classifier**
  ```bash
  python classification/train_classifier.py \
      --data_path ./data/ai_vs_human \
      --epochs 20 \
      --batch_size 32
  ```
- **Pix2Pix generator**
  ```bash
  python generation/train_pix2pix.py \
      --data_path ./data/paired \
      --epochs 100 \
      --batch_size 1
  ```

## Inference
- **Classify an image**
  ```bash
  python classification/inference.py --image_path sample.jpg
  ```
- **Generate an AI counterpart**
  ```bash
  python generation/generate.py \
      --image_path sample.jpg \
      --output_dir ./outputs/
  ```

## Results
- EfficientNetV2 classifier reliably separates real vs synthetic images.
- Pix2Pix model produces consistent translations aligned with paired data.
- Workflow supports side-by-side comparison for authenticity verification.

## Future Work
- Expand to multi-class attribution (identify specific generator sources).
- Extend Pix2Pix for style transfer or broader domain adaptation.
- Deploy as a lightweight web app for real-time inference.