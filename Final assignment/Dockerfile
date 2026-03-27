# FROM pytorch/pytorch:2.10.0-cuda12.6-cudnn9-runtime

# ENV PYTHONDONTWRITEBYTECODE=1
# ENV PYTHONUNBUFFERED=1

# RUN apt-get update && apt-get install -y \
#     python3 \
#     python3-pip \
#     python3-dev \
#     python3-venv \
#     && rm -rf /var/lib/apt/lists/*


# # create virtual environment
# RUN python3 -m venv /opt/venv
# ENV PATH="/opt/venv/bin:$PATH"


# # install dependencies (add any additional dependencies your project requires)
# RUN pip install --no-cache-dir \
#   --index-url https://download.pytorch.org/whl/cu126 \
#   --extra-index-url https://pypi.org/simple \
#   torch torchvision transformers safetensors


# RUN pip install --no-cache-dir \
#     pillow

# WORKDIR /app
# COPY predict.py /app/predict.py
# COPY model.py /app/model.py
# COPY model.pt /app/model.pt

# ENTRYPOINT ["python", "/app/predict.py"]

FROM pytorch/pytorch:2.10.0-cuda12.6-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# create virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# install dependencies
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu126
RUN pip install --no-cache-dir \
    pillow transformers

# # ==========================================
# # BULLETPROOF OFFLINE METHOD: 
# # Download the model and save it locally to /app/mit-b0
# RUN mkdir -p /app/mit-b0
# RUN python3 -c "from transformers import SegformerForSemanticSegmentation; model = SegformerForSemanticSegmentation.from_pretrained('nvidia/mit-b0'); model.save_pretrained('/app/mit-b0')"
# RUN chmod -R 777 /app/mit-b0
# # ==========================================

#  #=========================================
# Download the FULLY trained model on Cityscapes and save it locally to /app/segformer-b0-cityscapes
RUN mkdir -p /app/segformer-b1-cityscapes
RUN python3 -c "from transformers import SegformerForSemanticSegmentation; model = SegformerForSemanticSegmentation.from_pretrained('nvidia/segformer-b1-finetuned-cityscapes-1024-1024'); model.save_pretrained('/app/segformer-b1-cityscapes')"
RUN chmod -R 777 /app/segformer-b1-cityscapes
# #=========================================

WORKDIR /app
COPY predict.py /app/predict.py
COPY model.py /app/model.py
COPY model.pt /app/model.pt

ENTRYPOINT ["python", "/app/predict.py"]