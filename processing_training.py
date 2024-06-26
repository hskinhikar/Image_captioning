#script to partially train a model


import os
import datasets
from transformers import VisionEncoderDecoderModel, AutoFeatureExtractor,AutoTokenizer

#disabling weights and biases
os.environ["WANDB_DISABLED"] = "true"

#downloading NLTK tokenizer data
import nltk
try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    nltk.download("punkt", quiet=True)


#intialize Vision-Text model:
#specify models for vision encoder and text decoder

image_encoder_model ="google/vit-base-patch16-224-in21k"
text_decade_model = "gpt2"

#load pretrained Vision-Text model
model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
    image_encoder_model, text_decade_model)

#setting up feature extractor for images
from transformers import ViTFeatureExtractor
#FeatureExtractor is used extract features
feature_extractor=ViTFeatureExtractor.from_pretrained(
    'google/vit-base-patch16-224-in21k',
    do_resize=True,
    size=224,
    do_normalize=True,
    image_mean=[0.485, 0.456, 0.406], # mean for 3 channels (RGB)
    image_std=[0.229, 0.224, 0.225] #std for 3 channels (RGB)
)

#Tokenizer is used to tokenize and encode text features
tokenizer=AutoTokenizer.from_pretrained(text_decade_model)


# GPT2 only has bos/eos tokens but not decoder_start/pad tokens
#pad tokens are to make each sentence same, GPT 2 doesnt have any of 
#this but already knows a End Of Sentence tokens so we can use this instead
tokenizer.pad_token = tokenizer.eos_token

# update the model config
model.config.eos_token_id = tokenizer.eos_token_id
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

#set output directory
output_dir = "vit-gpt-model"
#save model and components
model.save_pretrained(output_dir)
feature_extractor.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)



#Data loading and preparation

#going to use sample dataset from https://huggingface.co/datasets/ydshieh/coco_dataset_script

#downloaded relevant data using the download_coco1.py script

COCO_DIR = r"C:\Users\shiri\Desktop\Dataverze\coco_dataset"

ds = datasets.load_dataset("ydshieh/coco_dataset_script", "2017", data_dir=COCO_DIR,trust_remote_code=True)

'''
some finetuning which should be redone later depending on results

The directory containing a trained model will have all of these parameters 
can replace that file at any point so theres a chance this isn't neccesary 


from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
     output_dir="./results",
     num_train_epochs=3,
     per_device_train_batch_size=4,
     per_device_eval_batch_size=4,
     warmup_steps=500,
     weight_decay=0.01,
     logging_dir="./logs",
     logging_steps=10,
 )

trainer = Trainer(
     model=model,
     args=training_args,
     train_dataset=ds["train"],
     eval_dataset=ds["validation"]
 )

trainer.train()
'''
from PIL import Image

# text preprocessing step, tokenizes the text captions with padding to ensure all sequances are of same lenght
def tokenization_fn(captions, max_target_length):
    """Run tokenization on captions."""
    labels = tokenizer(captions, 
                      padding="max_length", 
                      max_length=max_target_length,
                      return_tensors="np")
    return labels['input_ids']

# image preprocessing step
# extracts features from images
def feature_extraction_fn(image_paths, check_image=True):
    """
    Run feature extraction on images
    If `check_image` is `True`, the examples that fails during `Image.open()` will be caught and discarded.
    Otherwise, an exception will be thrown.
    """

    if check_image:
        images = []
        to_keep = []
        for image_file in image_paths:
            try:
                img = Image.open(image_file).convert("RGB")
                images.append(img)
                to_keep.append(True)
            except Exception:
                to_keep.append(False)
    else:
        images = [Image.open(image_file).convert("RGB") for image_file in image_paths]

    encoder_inputs = feature_extractor(images=images, return_tensors="np")

    return encoder_inputs.pixel_values

#combines text and image preprocessing steps, returns dictionary containing tokenised labels and pixel values
def preprocess_fn(examples, max_target_length, check_image = True):
    """Run tokenization + image feature extraction"""
    image_paths = examples['image_path']
    captions = examples['caption']    
    
    model_inputs = {}
    # This contains image path column
    model_inputs['labels'] = tokenization_fn(captions, max_target_length)
    model_inputs['pixel_values'] = feature_extraction_fn(image_paths, check_image=check_image)

    return model_inputs

# Check if processed dataset exists otherwise start processing dataset
processed_dataset_path = "processed_dataset"
if os.path.exists(processed_dataset_path):
    processed_dataset = datasets.load_from_disk(processed_dataset_path)
else:
    processed_dataset = ds.map(
        function=preprocess_fn,
        batched=True,
        fn_kwargs={"max_target_length": 128},
        remove_columns=ds['train'].column_names
    )
    processed_dataset.save_to_disk(processed_dataset_path)

# Sample a subset of the dataset
sample_size = 1000  # Number of samples used for training
train_subset = processed_dataset['train'].shuffle(seed=42).select(range(sample_size))
eval_subset = processed_dataset['validation'].shuffle(seed=42).select(range(sample_size))

from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    evaluation_strategy="epoch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    output_dir="./image-captioning-output",
)

import evaluate
metric = evaluate.load("rouge")

import numpy as np

ignore_pad_token_for_loss = True

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    if ignore_pad_token_for_loss:
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    return result

from transformers import default_data_collator

# Instantiate trainer
trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_subset,
    eval_dataset=eval_subset,
    data_collator=default_data_collator,
)

# Train the model
trainer.train()

# Save the trained model and tokenizer
trainer.save_model("./image-captioning-output_1k")
tokenizer.save_pretrained("./image-captioning-output_1k")