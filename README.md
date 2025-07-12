# knowledge_distillation

# 🔍 Deep Image Sharpening using a Lightweight Student Model (PyTorch)

This project demonstrates an end-to-end deep learning pipeline to enhance blurry images using a custom-built convolutional neural network (CNN). Designed for video conferencing and low-bandwidth conditions, the system trains a lightweight "student" model capable of running at high frame rates with strong perceptual quality.

## 🚀 Highlights

- 🧠 Lightweight CNN architecture for fast image sharpening
- 🔁 Automated dataset generation from high-resolution images
- 📊 Structural Similarity Index (SSIM) for image quality evaluation
- ⚡ Fast inference with PyTorch — optimized for 30–60 FPS+
- 📦 Modular design for training, inference, and evaluation

## 🎯 Project Objective

Enhance image clarity during video conferencing by simulating blurry conditions and restoring sharpness through a student-teacher knowledge distillation framework. This approach supports real-time video enhancement and works well even under degraded bandwidth scenarios.

## 🗂️ Folder Structure

| Folder / File                    | Description                                               |
|----------------------------------|-----------------------------------------------------------|
| `archive/`                       | Original high-quality images used to generate training pairs |
| `datasets/train/low/`           | Simulated blurry (downscaled & upscaled) images           |
| `datasets/train/high/`          | Ground truth sharp images                                 |
| `datasets/test/`                | Input image for sharpening and corresponding output       |
| `models/`                       | Saved PyTorch student model (.pth)                        |
| `train_student.py`              | Script to train the student model                         |
| `inference.py`                  | Runs sharpening on a test image                           |
| `evaluate_ssim.py`              | Computes SSIM between sharpened and ground-truth images   |
| `create_dataset_from_archive.py`| Automatically generates paired blurry/sharp images        |
| `model_student.py`              | CNN architecture for the student model                    |
| `dataset.py`                    | Custom PyTorch Dataset loader class                       |

## ⚙️ Setup Instructions

### Download all the required files using this google drive link sorry for the inconvenience
```bash
https://drive.google.com/file/d/15nGCmY2-bZXlhPfnk7TpleaPAY7kyucD/view?usp=sharing
```

### Set up a virtual environment & install dependencies:
```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision opencv-python pillow scikit-image tqdm
```

---

## 📁 Dataset Preparation

- Add your high-quality images to the `archive/` folder.

- Run the script to create blurry/sharp training pairs:
```bash
python create_dataset_from_archive.py
```

This will populate:

- `datasets/train/low/` — blurred images  
- `datasets/train/high/` — ground truth sharp images

---

## 🏋️‍♂️ Training the Student Model

Run the training script:
```bash
python train_student.py
```

This will train the `StudentSharpenModel` and save it to:
- `models/student_model.pth`

---

## 🔎 Image Inference

To sharpen a test image:

- Place your blurry test image at:  
  `datasets/test/input.jpg`

- Run the inference script:
```bash
python inference.py
```

This will produce:
- `datasets/test/output_sharpened.jpg`

---

## 📊 SSIM Evaluation

To compare your sharpened image against a high-quality version:

- Ensure both input and ground-truth images exist in `datasets/test/`

- Run:
```bash
python evaluate_ssim.py
```

Example Output:
```
🔍 SSIM Score: 0.9912
```

---

## 📌 Notes

- You can adapt this project for real-time webcam sharpening or video frame enhancement.
- The model is designed for low-latency scenarios like video calls or edge deployment.

---

## ✅ Requirements

- Python 3.7+
- PyTorch
- torchvision
- OpenCV (cv2)
- Pillow
- scikit-image
- tqdm
# Also check this video i recommend to check after 15 mins 
