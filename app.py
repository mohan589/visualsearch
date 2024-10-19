from pathlib import Path
import clip
import glob
import os
import ssl
from pinecone import Pinecone
import numpy as np
import torch
import streamlit as st
import pandas as pd
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import cv2
ssl._create_default_https_context = ssl._create_stdlib_context

st.title('VisualSearchEngine')

pc = Pinecone(api_key="5f00d642-14fd-4fd3-acb4-60f9976000ea")
index = pc.Index("visualsearch")

device = "mps" if torch.backends.mps.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image_path = Path("/Users/mpichikala/personal/images")
images = glob.glob(str(image_path / "*"))

global text_input, searchResults, selectedImagePortionVector, resizedImage
text_input = ""
searchResults = []
selectedImagePortionVector = []
resizedImage = ''

with st.sidebar:
  pic = st.file_uploader("Select File to Upload", type=['png', 'jpg', 'jpeg'])
  image = Image.open(pic) if pic else None
  if pic:
    canvas_result = st_canvas(
      fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
      stroke_width=1,
      background_image=image,
      update_streamlit=True,
      width=400,
      height=550,
      drawing_mode="circle",
      key="canvas",
      display_toolbar=False
    )

    if canvas_result.image_data is not None:
      objects = pd.json_normalize(canvas_result.json_data["objects"]) # need to convert obj to str because PyArrow
      if len(objects) > 0:
        right = int(objects.tail(1)['left'].iloc[0]) + int(objects.tail(1)['width'].iloc[0]) + 100
        bottom = int(objects.tail(1)['top'].iloc[0]) + int(objects.tail(1)['height'].iloc[0])
        coordinateTuples = (int(objects.tail(1)['left'].iloc[0]), int(objects.tail(1)['top'].iloc[0]), right, bottom)
        croppedImage = image.crop(coordinateTuples)
        resizedImage = croppedImage
        st.image(croppedImage, caption="Cropped Image", use_column_width=True, width=300)
      st.dataframe(objects)

with st.sidebar:
  text_input = st.text_input("Search Text", "")
  if st.button("Search..."):
    st.write(f"Searching for {text_input}!")
    text_embedding = clip.tokenize(text_input).to(device)
    text_features = model.encode_text(text_embedding).detach().cpu().numpy()
    searchResults = index.query(vector=np.squeeze(text_features).tolist(), top_k=100, include_values=True, include_metadata=True)
  if resizedImage:
    updatedImage = preprocess(resizedImage).unsqueeze(0).to(device)
    with torch.no_grad():
      text_features = model.encode_image(updatedImage).cpu().numpy().tolist()
    searchResults = index.query(vector=np.squeeze(text_features).tolist(), top_k=100, include_values=True, include_metadata=True)
  searchResults = searchResults['matches'] if searchResults and len(searchResults['matches']) > 0 else []

with st.container():
  st.write("This is inside the container")

  row = st.columns(3)
  if searchResults:
    for index, img in enumerate(searchResults):
      with row[index % 3]:
        image = Image.open(img['metadata']['path'])
        new_image = image.resize((300, 200))
        st.image(new_image, caption=img['metadata']['name'])