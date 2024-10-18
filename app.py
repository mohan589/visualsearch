# import streamlit as st
from PIL import Image
# from streamlit_cropper import st_cropper
import torch
import chromadb
import clip
from pathlib import Path
import glob
import os
import ssl
from pinecone import Pinecone, ServerlessSpec
import numpy as np
ssl._create_default_https_context = ssl._create_stdlib_context

# st.title('VisualSearchEngine')

# with st.sidebar:
# 	pic = st.file_uploader("Select File to Upload", type=['png', 'jpg', 'jpeg'])
# 	if pic:
# 		img = Image.open(pic)
# 		updated_img = img.resize((300, 400))
# 		cropped_pic = st_cropper(img_file=updated_img, realtime_update=True, aspect_ratio=(1, 1), should_resize_image=True, default_coords=None, box_color='white')
# 		if cropped_pic:
# 			st.write("Preview")
# 			_ = cropped_pic.thumbnail((100,100))
# 			st.image(cropped_pic, output_format="PNG")

# with st.container():
#   st.write("This is inside the container")
  
#   row1 = st.columns(3)
#   row2 = st.columns(3)
  
#   for col in row1 + row2:
#     tile = col.container(height=250)
#     tile.title(":balloon:")

pc = Pinecone(api_key="5f00d642-14fd-4fd3-acb4-60f9976000ea")

pc.create_index(name="visualsearch", dimension=512, spec=ServerlessSpec(cloud='aws', region='us-east-1'))

index = pc.Index("visualsearch")

device = "mps" if torch.backends.mps.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image_path = Path("/Users/mpichikala/personal/images")
images = glob.glob(str(image_path / "*"))

chroma_client = chromadb.Client()
image_collection = chroma_client.get_or_create_collection("visual_search" , metadata={"hnsw:space": "cosine"})

ids_index = 1
image_embedding = []
for i in images:
  image = preprocess(Image.open(i)).unsqueeze(0).to(device)
  with torch.no_grad():
    image_features = model.encode_image(image).cpu().numpy().tolist()
  image_embedding.append(image_features)
  ids_index += 1
  print(f"processing at index {i} and {ids_index}")
  # image_collection.add(ids= str(ids_index + 1), embeddings=image_features, metadatas={"path": i , "name": os.path.basename(i)})
  # Upsert your vector(s)
  index.upsert(
    vectors=[
      {"id": str(ids_index + 1), "values": np.squeeze(image_features).tolist(), "metadata": {"path": i , "name": os.path.basename(i)}}
    ],
    batch_size=100
  )
print(f"=============================")

while True:
  text_input = input("Enter text: ")
  text_embedding = clip.tokenize(text_input).to(device)
  text_features = model.encode_text(text_embedding).detach().cpu().numpy()
  result = image_collection.query(text_features, n_results=2)
  print(result, 'result')
  for i in result['metadatas'][0]:
    print(i['name'])
    print(i['path'])
    img = Image.open(i['path'])
    img.show()