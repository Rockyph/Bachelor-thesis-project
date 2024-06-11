import numpy as np
import torch

import tqdm, wget, os, random, zipfile

def download_and_extract(url, extract_to):
    # Ensure the directory exists
    os.makedirs(extract_to, exist_ok=True)

    # Define the file path
    filepath = os.path.join(extract_to, 'enwik8.zip')
    extracted_file_path = os.path.join(extract_to, 'enwik8')
    
    if not os.path.exists(filepath):
        # Download the file only if it doesn't exist
        print("Downloading dataset ...")
        wget.download(url, filepath)
        print("\nDownloaded.")

    if not os.path.exists(extracted_file_path):
        print(f"Extracting the data ...")
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("Extracted.")
        
    return load_enwik8(extracted_file_path, train_ratio=0.7, val_ratio=0.1)



def load_enwik8(file_path, train_ratio=0.7, val_ratio=0.1):       
    with open(file_path, 'rb') as file:
        data = np.frombuffer(file.read(), dtype=np.uint8)
        
        # No sure if we need this -> but it says that the buffer's default is not writable
        writable_data = np.copy(data)
        
        data_length = len(writable_data)
        train_len = int(data_length * train_ratio)
        val_len = int(train_len * val_ratio)
        
        trX, vaX, teX = np.split(writable_data, [train_len, train_len + val_len])
        return torch.from_numpy(trX), torch.from_numpy(vaX), torch.from_numpy(teX)