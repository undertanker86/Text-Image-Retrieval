# Image Retrieval Project

## Overview

This project implements an image retrieval system using both traditional methods and advanced deep learning techniques. The system allows users to input a query image and retrieve similar images from a dataset. The project also includes tools for dataset creation, processing, and management using vector databases for efficient image searches.
![image](https://drive.google.com/file/d/17nVueUnoU3teddWiFlEKEAiusaJw_fNz/view?usp=sharing)

## Features

- **Basic Image Retrieval**: Supports L1, L2, Cosine Similarity, and Correlation Coefficient methods.
- **Advanced Image Retrieval**: Utilizes a pretrained CLIP model for feature extraction and enhanced search accuracy.
- **Vector Database Integration**: Stores and queries image embeddings using a vector database (e.g., ChromaDB).
- **Data Collection Tools**: Crawls images from websites and organizes them into datasets.
- **Data Processing Tools**: Cleans and structures datasets for use in retrieval tasks.

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/undertanker86/Text-Image-Retrieval.git
   cd image-retrieval-project
2. ** Install OpenCLIPEmbeddingFunction for version VectorDatabase and Embeddinng **

## Project Structure

Embedding.py: Embedding extraction using CLIP.
Traditional.py: Traditional image similarity measures.
VectorDatabase.py: Vector database management.
data_collection.py: Tools for dataset creation and processing.
Project_Image_Retrieval.pdf: Documentation and project details.
