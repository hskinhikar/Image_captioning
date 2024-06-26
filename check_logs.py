import torch
from transformers import TrainingArguments

# Load the training arguments by giving path of the training_args.bin file within the directory for the trained model
training_args = torch.load(r'C:\Users\shiri\Desktop\Dataverze\image-captioning-output_1k\training_args.bin', map_location='cpu')

# Convert the TrainingArguments object to a dictionary for better readability
training_args_dict = training_args.to_dict()

# Display the training arguments
for key, value in training_args_dict.items():
    print(f"{key}: {value}")



