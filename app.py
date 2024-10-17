import streamlit as st
from PIL import Image
from streamlit_cropper import st_cropper
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

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
    
dataset = load_dataset("Abrumu/Fashion_controlnet_dataset_V3")

model = SentenceTransformer('clip-ViT-B-32')

# Generate embeddings for all images
def embed_image(example):
  print(example, 'example')
  return {'image_embedding': model.encode(example['target'])}

dataset_with_embeddings = dataset.map(embed_image, batched=True)

pinecone = Pinecone(api_key='5f00d642-14fd-4fd3-acb4-60f9976000ea')

# Create an index if not already created
# if 'fashion-images' not in pinecone.list_indexes():
#   pinecone.create_index('fashion-images', dimension=512, spec=ServerlessSpec(cloud='aws', region='us-east-1') )

index = pinecone.Index('fashion-images')

vectors = []
print(dataset_with_embeddings)

for i, record in enumerate(dataset_with_embeddings):
  print(record, 'record')
  vector = {
    'id': f'image-{i}',
    'values': record['image_embedding'],
    'metadata': {'prompt': record['prompt'], 'clip_caption': record['CLIP_captions']}
  }
  vectors.append(vector)

# Upload vectors to Pinecone
index.upsert(vectors)

