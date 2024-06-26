import requests
import os

urls = [
"http://images.cocodataset.org/zips/train2017.zip"
"http://images.cocodataset.org/zips/val2017.zip"
"http://images.cocodataset.org/zips/test2017.zip"
"http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
"http://images.cocodataset.org/annotations/image_info_test2017.zip"
]

output_dir = "coco_dataset"
os.makedirs(output_dir, exist_ok=True)

def download_file(url, output_dir):
    local_filename = os.path.join(output_dir, url.split('/')[-1])
    print(f'Starting download: {url}')
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
    print(f'Download complete: {local_filename}')

for url in urls:
    download_file(url, output_dir)
