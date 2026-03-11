# DiffHash: Text-Guided Adversarial Hashing Attack via Diffusion Model

## Requirements

- Python 3.8+
- PyTorch >= 1.11 (CUDA)
- diffusers >= 0.31
- transformers >= 4.46
- clip (OpenAI CLIP, `pip install git+https://github.com/openai/CLIP.git`)
- lpips
- pytorch-msssim
- eagerpy
- Pillow, numpy, scipy, matplotlib, tqdm

Install dependencies:

```bash
pip install torch torchvision diffusers transformers lpips pytorch-msssim eagerpy tqdm matplotlib scipy
pip install git+https://github.com/openai/CLIP.git
```

## External Models (Download Separately)

The following large files are **not included** in the archive and must be prepared manually:

| Model | Path | Description |
|---|---|---|
| Stable Diffusion v2.1 | `./stable/` | HuggingFace `runwayml/stable-diffusion` (download via `diffusers`) |
| CSQ Hash Models | `./hashing_model/` | Pre-trained CSQ ResNet50 for 16/32/64-bit |
| NUS-WIDE Images | `./data/NUS-WIDE/images/` | Raw NUS-WIDE images |

## Quick Start: 32-bit CSQ Experiment

```bash
cd DiffAttack
python main.py \
    --pretrained_diffusion_path "./stable" \
    --images_root "./data/NUS-WIDE/" \
    --annotations_file "./data/NUS-WIDE/annotations_2100_new.json" \
    --target_annotations_file "./data/NUS-WIDE/annotations_2100_target.json" \
    --hash_model "./hashing_model/NUS-WIDE_CSQ_ResNet50_32.pth" \
    --text_to_hash_model_path "./best_text_to_hash_model_CSQ_NUS-WIDE_5000.pth" \
    --database_hash_path "./data_path/database_code_NUS-WIDE_CSQ_ResNet50_32.txt" \
    --database_label_path "./data/NUS-WIDE/database_label.txt" \
    --save_dir "./output_32" \
    --dataset nuswide \
    --diffusion_steps 20 \
    --start_step 15 \
    --iterations 30 \
    --res 224 \
    --guidance 2.5
```

This runs the full targeted adversarial attack on NUS-WIDE with 32-bit CSQ hash model. Key parameters:

- `--diffusion_steps 20`: DDIM sampling steps
- `--start_step 15`: Step at which latent optimization begins
- `--iterations 30`: Number of optimization iterations
- `--guidance 2.5`: Classifier-free guidance scale

Results (adversarial images, per-sample t-MAP, retrieval visualizations) are saved to `./output_32/`.


