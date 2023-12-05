import torch
from torchinfo import summary
import torch.nn as nn

from model import Squeezeformer

BATCH_SIZE = 4
SEQ_LENGTH = 500
INPUT_SIZE = 80

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Squeezeformer(
    num_classes=10,
).to(device)

inputs = torch.FloatTensor(BATCH_SIZE, SEQ_LENGTH, INPUT_SIZE).to(device)
input_lengths = torch.IntTensor([500, 450, 400, 350]).to(device)


# summary(model, [torch.Tensor((SEQ_LENGTH, INPUT_SIZE)), torch.Tensor((BATCH_SIZE, ))], batch_size=BATCH_SIZE)
model_stats = summary(model, input_data=[inputs, input_lengths], col_names=["input_size",
                    "output_size",
                    "num_params",
                    "params_percent",
                    "kernel_size",
                    "mult_adds",
                    "trainable"], depth=7)

# Specify the file path for the output text file
output_file_path = "model_summary_full.txt"

# Open the file in write mode and write the string to it
with open(output_file_path, 'w') as file:
    file.write(str(model_stats))

outputs, output_lengths = model(inputs, input_lengths)