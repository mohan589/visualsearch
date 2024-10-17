import streamlit as st
import numpy as np
from PIL import Image
from streamlit_cropper import st_cropper

st.title('VisualSearchEngine')

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

with st.container():
  st.write("This is inside the container")
  
  row1 = st.columns(3)
  row2 = st.columns(3)
  
  for col in row1 + row2:
    tile = col.container(height=250)
    tile.title(":balloon:")