# Image captioning

Base code sourced from: https://ankur3107.github.io/blogs/the-illustrated-image-captioning-using-transformers/ 

First attempt at using a ViT + GPT-2 model trained on COCO dataset to caption images. This is purely recreated to understand and experiment with the parameters and models.
Pictures provided by client and performance of clients model was used as a reference point to judge the performance of the current model.
I have provided reports analysing models on loss metrics as well as visual judgement of captions vs images.
The script provided trains the model partially, and this is then compared to a fully trained model sourced from Hugging face (references provided in the documents) as well as the clients' model.
I have used knowledge gained from this project to determine direction of the hskinhikar/bounding_boxes_optimisation_RL project which has been my biggest project as part of my internship with DataVerze, which is still in progress.

Instructions to run the script:
Run download_coco_dataset.py file to download the dataset.
Run a virtual environment and install packages in the versions mentioned in the requirements file.
Run processed_testing.py to process data and store locally (code was changed from base code so that it first checks local directory if there is already processed data present).
This script also initialises a ViT vision transformer model and GPT-2 model, trains it on 1000 images and saves the partially trained model locally.
Evaluation metrics can be found by opening the trainer_state.json in notepad.

