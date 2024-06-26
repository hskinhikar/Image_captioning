#testing partially trained model against a fully trained model

from transformers import pipeline
from PIL import Image
import requests
from io import BytesIO

# Path to the local image
#path_to_image = r"C:\Users\shiri\AppData\Local\Temp\00d7a0df-9f3a-4b87-a2e9-7e07af7a1e55_test2017.zip.e55\test2017\000000000108.jpg"
url_to_image = 'https://images.unsplash.com/photo-1584395630827-860eee694d7b' 


# Load the image from the local file system
#image = Image.open(path_to_image) 

# Load the image from the URL
response = requests.get(url_to_image)
image = Image.open(BytesIO(response.content))

image_to_text_full = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")

caption_full =image_to_text_full(image, max_new_tokens=30)

image_to_text = pipeline("image-to-text", model=r"C:\Users\shiri\Desktop\Dataverze\image-captioning-output_1k")

caption_partial =image_to_text(image, max_new_tokens=30)

# [{'generated_text': 'a soccer game with a player jumping to catch the ball '}]
print(

    f'Fully trained model: {caption_full} whereas model trained on 2000 images says {caption_partial}'
)