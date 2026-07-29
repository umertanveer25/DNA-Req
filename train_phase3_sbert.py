import os, sys, time
import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import torch

def dna_mapping_5class(nfr_type):
    mapping = {'F': 'A', 'US': 'T', 'SE': 'C', 'PE': 'G'}
    return mapping.get(str(nfr_type).strip().upper(), 'N')

if __name__ == '__main__':
    # Force PyTorch to use 16 cores for all matrix operations natively
    torch.set_num_threads(16)
    torch.set_num_interop_threads(16)
    
    df = pd.read_csv('data/Promise_Dataset.csv')
    y_str = df['Type'].apply(dna_mapping_5class).values
    texts = df['Requirement'].tolist()

    label_to_id = {'A': 0, 'T': 1, 'C': 2, 'G': 3, 'N': 4}
    y_id = [label_to_id[lbl] for lbl in y_str]

    print("[+] Initializing pre-trained SBERT model (all-MiniLM-L6-v2) on 16 cores...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("[+] Preparing DataLoader for BatchHardTripletLoss...")
    train_examples = []
    for text, label_id in zip(texts, y_id):
        train_examples.append(InputExample(texts=[text], label=label_id))

    # Using num_workers=0 to avoid Windows multiprocessing serialization bugs, 
    # but torch.set_num_threads(16) ensures matrix math is still fully parallel!
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16, num_workers=0)

    print("[+] Configuring BatchHardTripletLoss (Immune to class imbalance)...")
    train_loss = losses.BatchHardTripletLoss(model=model)

    print("[+] Starting fine-tuning (Domain Adaptation) for 5 epochs on 16 cores...")
    model.fit(train_objectives=[(train_dataloader, train_loss)], 
              epochs=5, 
              warmup_steps=100, 
              show_progress_bar=True)

    out_dir = "models/sbert-promise-finetuned"
    os.makedirs(out_dir, exist_ok=True)
    model.save(out_dir)
    print(f"[+] Fine-tuned model saved successfully to {out_dir}")
