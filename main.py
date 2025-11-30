from src.train import train_model
from src.utils import save_plot 
from src.evaluate import evaluate_model # Import the new evaluator
from src.models import TransformerGenerator
import torch

if __name__ == "__main__":
    print("=== STEP 1: TRAINING ===")
    train_model()
    
    print("\n=== STEP 2: VISUALIZATION ===")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerGenerator(feat_dim=38).to(DEVICE)
    model.load_state_dict(torch.load('gen_model.pth'))
    save_plot(model, DEVICE)
    
    print("\n=== STEP 3: EVALUATION ===")
    evaluate_model()