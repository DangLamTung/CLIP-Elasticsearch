from elasticsearch import Elasticsearch

client = Elasticsearch(
   "http://localhost:9200",
    api_key="M0ZOMnBwWUJlckNGV1l2MVBtcmk6dmNsVnZfOU52cVpXYm12QjFBZHNLdw==",
)


# client.indices.create(
#     index="test",
#     body={
#         "mappings": {
#             "properties": {
#                 "vector": {
#                 	  "type": "dense_vector",
#                 	  "dims": 512,
#                    "similarity": "cosine",
#                 },
#                 "filename": {
#                  "type": "keyword",
#             	},
#         	},
#     	},
# 	}
# )


import os
import json
import matplotlib.pyplot as plt
from PIL import Image
import torch
import clip
from tqdm import tqdm
from PIL import UnidentifiedImageError
import cv2
image_dir = "./images/"
 
device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
print(model)
def process_images(start, end, batch_id):
    original_images = []
    images = []
    image_ids = []

    for idx, filename in enumerate(os.listdir(image_dir)):
        # print(filename)
        if not filename.endswith(".jpg") or idx < start or idx >= end:
            continue

        name, _ = os.path.splitext(filename)
        # if name not in descriptions:
        #     continue

        try:
            
            temp = cv2.imread(str(os.path.join(image_dir, filename)).replace("._","")) 
            temp = cv2.cvtColor(temp, cv2.COLOR_BGR2RGB) 
            image = Image.fromarray(temp)
            
        except Exception as e:
            print(e)
        #     continue
        # image = Image.open(os.path.join(image_dir, filename)).convert("RGB")
        original_images.append(image)
        images.append(preprocess(image))
        image_ids.append(os.path.join(image_dir, filename))
    # print(len(images))  
    if not images:
        return

    # Create a tensor from the list of images
    images_tensor = torch.stack(images).to(device)
 
    # Generate embeddings for the images
    with torch.no_grad():
        embeddings = model.encode_image(images_tensor).cpu().numpy()
      
    # elasticsearch_embeddings = [{"id": id, "embedding": embedding.tolist()} for id, embedding in zip(image_ids, embeddings)]

    for img_id, emb in  zip(image_ids, embeddings):
        print( emb.tolist()[0:10],img_id )
        client.index(
         index="test",
         body={
            "vector": emb.tolist(),
            "filename": img_id,
    	    }
        )



    # with open(f'elasticsearch_embeddings_{batch_id}.json', 'w') as f:
    #     json.dump(elasticsearch_embeddings, f)

# Number of images to process at a time
batch_size = 1000

# Total number of images
total_images = len([filename for filename in os.listdir(image_dir) if filename.endswith(".jpg") ])

# Process images in batches
for i in range(0, total_images, batch_size):
    print(f"Processing images {i} to {min(i + batch_size, total_images)}")
    process_images(i, i + batch_size, i // batch_size)



    