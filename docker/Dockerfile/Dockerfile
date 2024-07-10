FROM nvcr.io/nvidia/pytorch:22.08-py3

RUN apt-get update

RUN pip install torch==1.13.0+cu117 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install torchaudio==0.13.0+cu117 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install torchvision==0.14.0+cu117 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install pip --upgrade
RUN pip install neptune-client
RUN pip install wandb
RUN pip install pip install git+https://github.com/huggingface/transformers
RUN pip install transformers[torch]
RUN pip install datasets
RUN pip install huggingface_hub
RUN pip install torchsummary
RUN pip install numba==0.45.1 --ignore-installed
RUN pip install julius==0.2.7 --ignore-installed
RUN pip install pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
RUN pip install librosa==0.6.0 --ignore-installed
RUN pip install resampy==0.2.0 --ignore-installed
RUN pip install torch-audiomentations
RUN apt-get install git-lfs