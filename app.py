from pathlib import Path
import clip
import glob
import os
import ssl
from pinecone import Pinecone
import numpy as np
import torch
import streamlit as st
from PIL import Image
from streamlit_cropper import st_cropper

ssl._create_default_https_context = ssl._create_stdlib_context

st.title('VisualSearchEngine')

pc = Pinecone(api_key="5f00d642-14fd-4fd3-acb4-60f9976000ea")
index = pc.Index("visualsearch")

device = "mps" if torch.backends.mps.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image_path = Path("/Users/mpichikala/personal/images")
images = glob.glob(str(image_path / "*"))

with st.sidebar:
	pic = st.file_uploader("Select File to Upload", type=['png', 'jpg', 'jpeg'])
	if pic:
		img = Image.open(pic)
		updated_img = img.resize((300, 400))
		cropped_pic = st_cropper(img_file=updated_img, realtime_update=True, aspect_ratio=(1, 1), should_resize_image=True, default_coords=None, box_color='white')
		if cropped_pic:
			st.write("Preview")
			_ = cropped_pic.thumbnail((100,100))
			st.image(cropped_pic, output_format="PNG")

global text_input, searchResults
text_input = ""
searchResults = []

with st.sidebar:
  text_input = st.text_input("Search Text", "")
  if st.button("Search..."):
    st.write(f"Searching for {text_input}!")
    text_embedding = clip.tokenize(text_input).to(device)
    text_features = model.encode_text(text_embedding).detach().cpu().numpy()
    searchResults = index.query(vector=np.squeeze(text_features).tolist(), top_k=100, include_values=True, include_metadata=True)
    searchResults = searchResults['matches']

with st.container():
  st.write("This is inside the container")
  
  row = st.columns(3)
  if searchResults:
    for index, img in enumerate(searchResults):
      with row[index % 3]:
        image = Image.open(img['metadata']['path'])
        new_image = image.resize((300, 200))
        st.image(new_image, caption=img['metadata']['name'])