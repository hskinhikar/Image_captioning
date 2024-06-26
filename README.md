# Image classification: attempt 1

Base code sourced from: https://ankur3107.github.io/blogs/the-illustrated-image-captioning-using-transformers/ 

First attempt at using a ViT + GPT-2 model trained on COCO dataset to caption images.

Run download_coco_dataset.py file to download the dataset.

Run a virtual environment and install packages in the versions mentioned in the requirements file.

Run processed_testing.py to process data and store locally (code was changed from base code so that it first checks local directory if there is already processed data present).

This script also initialises a ViT vision transformer model and GPT-2 model, trains it on 1000 images and saves the partially trained model locally.

Evaluation metrics can be found by opening the trainer_state.json in notepad.

Presentation on the analysis has also been uploaded.
