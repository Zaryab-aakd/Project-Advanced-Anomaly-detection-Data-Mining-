import torch
import matplotlib.pyplot as plt
import numpy as np
from src.data_loader import SMDLoader
from src.preprocessing import Preprocessor

def geometric_masking(x, mask_ratio=0.2):
    batch, seq_len, feats = x.shape
    mask = torch.ones_like(x)
    num_masked = int(seq_len * mask_ratio)
    for i in range(batch):
        start_idx = torch.randint(0, seq_len - num_masked, (1,))
        for idx in start_idx:
            mask[i, idx : idx + num_masked, :] = 0  
    return x * mask

def save_plot(model, device):
    print(" Generating result plot with CORRECT scaling...")
    
    # 1. Load Data
    loader = SMDLoader()
    train_data, test_data, _ = loader.load_machine('machine-1-1')
    
    # 2. Preprocess
    preprocessor = Preprocessor(window_size=100)
    
    
    # This aligns the math so the model isn't confused
    preprocessor.fit_transform(train_data) 
    test_windows = preprocessor.transform(test_data)
    
    # 3. Inference
    input_tensor = torch.FloatTensor(test_windows[:200]).to(device)
    model.eval()
    with torch.no_grad():
        reconstructed = model(input_tensor)
        
    # 4. Plot
    input_np = input_tensor.cpu().numpy()
    recon_np = reconstructed.cpu().numpy()
    
    plt.figure(figsize=(15, 6))
    plt.plot(input_np[0, :, 0], label='Original', color='blue', alpha=0.6)
    plt.plot(recon_np[0, :, 0], label='Reconstructed', color='red', linestyle='--')
    plt.title("Anomaly Detection Result (Feature 0)")
    plt.legend()
    
    plt.savefig('results.png')
    plt.close()
    print(" Plot saved as 'results.png'")
