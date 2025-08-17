# 🦥 Unsloth Fine-tune Environment (Docker)

This repository provides a **Dockerized environment** for fine-tuning models using [Unsloth](https://github.com/unslothai/unsloth).  
It includes:  
- A **Dockerfile** that installs all required dependencies  
- A **Python script** that demonstrates fine-tuning a model

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/shungyan/Unsloth-Finetune-App.git
cd Unsloth-Finetune-App
```
### 2. Modify the finetune.py with your model and SFT trainer configuration.

<img width="417" height="192" alt="image" src="https://github.com/user-attachments/assets/80f9fe4e-821b-4644-98b4-02ac1ca58f6b" />
<br>
<br>
<img width="363" height="489" alt="image" src="https://github.com/user-attachments/assets/be207df8-e773-48b4-9e11-7df60fd2b734" />


### 3. Build the Docker Image
```bash
docker build -t unsloth-finetune .
```

### 4. Run the Docker Container
```bash
docker run -it --gpus all \
  --name unsloth-finetune \
  -v $(pwd):/workspace \
  unsloth-finetune
```

🖥️ Requirements

NVIDIA GPU with CUDA support 

Docker with GPU support enabled

Python script (finetune.py) with your training configurations or model. 


✅ Tested Environment

GPU: NVIDIA RTX 1650 Ti (4GB VRAM)

OS: Local PC with Docker + GPU support

Model: Due to VRAM limitations, only smaller models can run.

Tested successfully with:

unsloth/Phi-3-mini-4k-instruct

⚠️ Larger models may run out of memory on GPUs with 4GB VRAM.
