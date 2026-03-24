import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
# 2.10.0+cpu
# None
# False

# -----------------------------
# uninstall torch only cpu:  pip uninstall torch torchvision torchaudio -y
# install torch cuda: pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
# Take this link on Page: https://pytorch.org/get-started/locally/
# ----------------------------------
# Run tensorboard: tensorboard --logdir runs