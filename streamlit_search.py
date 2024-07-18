import streamlit as st
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
import os

db_path=r"C:\Users\IdeaProjects\search_engine_youtube\image_vdb" #add your db path here

# Initialize Chroma DB client, embedding function, and data loader
client = chromadb.PersistentClient(path=db_path)
embedding_function = OpenCLIPEmbeddingFunction()
data_loader = ImageLoader()

collection = client.get_or_create_collection(
    name='multimodal_collection',
    embedding_function=embedding_function,
    data_loader=data_loader
)

# Display the banner image
banner_image_path = r"C:\Users\nachi\IdeaProjects\search_engine_youtube\img.png"  # Update with the path to your banner image
st.image(banner_image_path, use_column_width=True)


st.title("Vehicle Image Search Engine")

# Search bar
query = st.text_input("Enter your search query:")
parent_path = r"C:\Users\IdeaProjects\search_engine_youtube\vehicle_images\train\images" #add your image folder path here
if st.button("Search"):
    results = collection.query(query_texts=[query], n_results=5,include=["distances"])
    print(results)
    for image_id, distance in zip(results['ids'][0], results['distances'][0]):
        image_path = os.path.join(parent_path, image_id)
        st.image(image_path, caption=os.path.basename(image_path))
        st.write(f"Distance: {distance}")