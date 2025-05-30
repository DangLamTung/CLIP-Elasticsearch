import os
import json
import matplotlib.pyplot as plt
from PIL import Image
import torch
import clip
from tqdm import tqdm
from PIL import UnidentifiedImageError
import cv2

from elasticsearch import Elasticsearch

client = Elasticsearch(
   "http://localhost:9200",
    api_key="M0ZOMnBwWUJlckNGV1l2MVBtcmk6dmNsVnZfOU52cVpXYm12QjFBZHNLdw==",
)




image_dir = "./images/"

# device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device='cpu')


# image_input = torch.tensor(np.stack(images)) 
text_tokens = clip.tokenize(["statue"], context_length=77, truncate=True) 
with torch.no_grad():
    embeddings = model.encode_text(text_tokens ).cpu().numpy()
# query_vector = clip.embed_text("coffee")

# print(embeddings.tolist()[0])
response = client.search(
	index="test",
	body={
    	"query": {
        	"knn": {
            	"field": "vector",
            	"query_vector": embeddings.tolist()[0],  # The input vector
            	"k": 3
        	}
    	}
	}
)
# print(response["hits"]["hits"][0])

for i in response["hits"]["hits"][:3]:
    print( i["_source"]["filename"])
# images = [Image.open(os.path.join(image_dir, i["_source"]["filename"])) for i in response["hits"]["hits"][:3]]

# plt.plot_images_grid(images, grid_size=(1, 3))