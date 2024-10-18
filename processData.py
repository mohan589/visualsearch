from pathlib import Path
import glob
import os
import ssl
from PIL import Image
import torch
# import chromadb
import clip
from pinecone import Pinecone
import numpy as np
ssl._create_default_https_context = ssl._create_stdlib_context

pc = Pinecone(api_key="5f00d642-14fd-4fd3-acb4-60f9976000ea")

# pc.create_index(name="visualsearch", dimension=512, spec=ServerlessSpec(cloud='aws', region='us-east-1'))

index = pc.Index("visualsearch")

device = "mps" if torch.backends.mps.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image_path = Path("/Users/mpichikala/personal/images")
images = glob.glob(str(image_path / "*"))

# chroma_client = chromadb.Client()
# image_collection = chroma_client.get_or_create_collection("visual_search" , metadata={"hnsw:space": "cosine"})

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
    batch_size=1000
  )

while True:
  text_input = input("Enter text: ")
  text_embedding = clip.tokenize(text_input).to(device)
  text_features = model.encode_text(text_embedding).detach().cpu().numpy()
  # result = image_collection.query(text_features, n_results=2)
  result = index.query(vector=np.squeeze(text_features).tolist(), top_k=2, include_values=True, include_metadata=True)
  # print(result, 'result')
  # for i in result['metadatas'][0]:
  #   print(i['name'])
  #   print(i['path'])
  #   img = Image.open(i['path'])
  #   img.show()
  
  for i in result['matches']:
    img = Image.open(i['metadata']['path'])
    img.show()
