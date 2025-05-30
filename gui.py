import streamlit as st
import chromadb

import time
from PIL import Image
import clip

from elasticsearch import Elasticsearch
import torch
client = Elasticsearch(
   "http://localhost:9200",
    api_key="M0ZOMnBwWUJlckNGV1l2MVBtcmk6dmNsVnZfOU52cVpXYm12QjFBZHNLdw==",
)




image_dir = "./images/"

# device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device='cpu')



# print(response["hits"]["hits"][0])




# Streamlit config
st.set_page_config(layout="wide")
st.title("Image search engine")

# Enter search term or provide image
option = st.selectbox('How do you want to search?', ('Search Term', 'Image'))
if option == "Search Term":
    uploaded_file = None
    search_term = st.text_input("Enter search term")
else:
    search_term = None
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        st.image(uploaded_file, width=200)

st.markdown('<hr style="border:1px #00008B; border-style: solid; margin:0;">', 
    unsafe_allow_html=True)


with st.empty():
    if option and (uploaded_file or search_term):
        start = time.time()
        images = []
        with st.spinner('Searching'):
            if option == 'Search Term':
                # query_embeddings = model.embed(search_term)
                text_tokens = clip.tokenize([search_term], context_length=77, truncate=True)
                with torch.no_grad():
                    query_embeddings = model.encode_text(text_tokens ).cpu().numpy()
            else:
                image = Image.open(uploaded_file).convert("RGB")
                images.append(preprocess(image))
                images_tensor = torch.stack(images).to("cpu")
                query_embeddings = model.encode_image(images_tensor).cpu().detach().numpy()
            response = client.search(
            index="test",
            body={
                "query": {
                    "knn": {
                        "field": "vector",
                        "query_vector": query_embeddings.tolist()[0],  # The input vector
                        "k": 3
                    }
                }
            }
        )
        result = []
        score = []
        for i in response["hits"]["hits"][:3]:
            print( i["_source"]["filename"])
            score.append(i['_score'])
            result.append(i["_source"]["filename"])
        end = time.time()

        # metadatas = result["metadatas"][0]
        # distances = result["distances"][0]
        with st.container():
            st.write(f"**Results** ({end-start:.2f} seconds)")
            for id, file_path in enumerate(result):
                left, right = st.columns([0.5, 0.5])
                with left:
                    st.image(Image.open(file_path.replace("._","")), width=500)
                with right:
                    st.markdown(f"""**Id**: {id}  
                        **Distance**: {score[id]}
                    """)
    else:
        st.write("No results to show. Enter a search term above.")