import os
import glob
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to read text files
def read_files(folder_path):
    files = glob.glob(os.path.join(folder_path, "*.txt"))
    texts = []
    filenames = []
    for file in files:
        with open(file, 'r', encoding="utf-8") as f:
            texts.append(f.read())
            filenames.append(os.path.basename(file))
    return texts, filenames

# Function to check plagiarism
def check_plagiarism(texts, filenames):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    print("\nPlagiarism Report:\n")
    for i in range(len(filenames)):
        for j in range(i+1, len(filenames)):
            print(f"{filenames[i]} vs {filenames[j]} → Similarity: {similarity_matrix[i, j] * 100:.2f}%")

# Folder containing text files
folder_path = "documents"

# Read files and check plagiarism
texts, filenames = read_files(folder_path)
check_plagiarism(texts, filenames)
