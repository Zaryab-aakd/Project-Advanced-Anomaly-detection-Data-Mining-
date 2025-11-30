import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from src.data_loader import SMDLoader
from src.preprocessing import Preprocessor
from src.models import TransformerGenerator

def evaluate_model():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n --- Starting Model Evaluation ---")

    # 1. Load Test Data & LABELS
    loader = SMDLoader()
    _, test_data, test_labels = loader.load_machine('machine-1-1')

    # 2. Preprocess
    preprocessor = Preprocessor(window_size=100)
    test_windows = preprocessor.fit_transform(test_data)
    
    # 3. Align Labels with Windows
    # If a window covers time t=0 to t=100, we typically use the label at t=100
    window_labels = test_labels[100-1:] 
    
    # Safety clip to match lengths exactly
    min_len = min(len(test_windows), len(window_labels))
    test_windows = test_windows[:min_len]
    window_labels = window_labels[:min_len]

    # 4. Load Model
    model = TransformerGenerator(feat_dim=38).to(DEVICE)
    try:
        model.load_state_dict(torch.load('gen_model.pth'))
        model.eval()
    except FileNotFoundError:
        print(" Model not found. Cannot evaluate.")
        return

    # 5. Calculate Anomaly Scores (MSE)
    input_tensor = torch.FloatTensor(test_windows).to(DEVICE)
    scores = []
    
    # Batch processing to save memory
    batch_size = 256
    with torch.no_grad():
        for i in range(0, len(input_tensor), batch_size):
            batch = input_tensor[i : i + batch_size]
            recon = model(batch)
            # MSE per window
            loss = torch.mean((batch - recon) ** 2, dim=[1, 2])
            scores.extend(loss.cpu().numpy())
            
    # 6. Calculate AUROC
    try:
        auroc = roc_auc_score(window_labels, scores)
        print(f" FINAL TEST AUROC SCORE: {auroc:.4f}")
        if auroc > 0.80:
            print("Result: Excellent ")
        elif auroc > 0.70:
            print(" Result: Good ")
        else:
            print(" Result: Needs Improvement")
    except Exception as e:
        print(f"Could not calculate AUROC: {e}")

    print("-----------------------------------")
