import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.data_loader import SMDLoader
from src.preprocessing import Preprocessor
from src.models import TransformerGenerator, Discriminator
from src.utils import geometric_masking

def train_model():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f" Training on {DEVICE}")
    
    # Load & Process
    loader = SMDLoader()
    train_data, _, _ = loader.load_machine('machine-1-1')
    preprocessor = Preprocessor(window_size=100)
    train_windows = preprocessor.fit_transform(train_data)
    
    tensor_x = torch.FloatTensor(train_windows).to(DEVICE)
    train_loader = DataLoader(TensorDataset(tensor_x), batch_size=64, shuffle=True)
    
    # Init Models
    feat_dim = train_windows.shape[2]
    gen = TransformerGenerator(feat_dim).to(DEVICE)
    disc = Discriminator(feat_dim).to(DEVICE)
    
    
    # We slow down D so G can catch up
    g_opt = optim.Adam(gen.parameters(), lr=0.0001)
    d_opt = optim.Adam(disc.parameters(), lr=0.00001) # Added an extra zero here
    
    crit_rec = nn.MSELoss()
    crit_adv = nn.BCELoss()
    
    # Loop
    EPOCHS = 30
    for epoch in range(EPOCHS):
        total_g = 0
        total_d = 0
        for batch_idx, (real,) in enumerate(train_loader):
            # Train Disc
            d_opt.zero_grad()
            real_label = torch.ones(real.size(0), 1).to(DEVICE)
            fake_label = torch.zeros(real.size(0), 1).to(DEVICE)
            
            d_real_loss = crit_adv(disc(real), real_label)
            
            masked = geometric_masking(real.clone())
            fake = gen(masked)
            d_fake_loss = crit_adv(disc(fake.detach()), fake_label)
            
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_opt.step()
            total_d += d_loss.item()
            
            # Train Gen
            g_opt.zero_grad()
            rec_loss = crit_rec(fake, real)
            adv_loss = crit_adv(disc(fake), real_label)
            
           
            # Increase Reconstruction weight (10 -> 50)
            # Decrease Adversarial weight (1 -> 0.05)
            g_loss = 50.0 * rec_loss + 0.05 * adv_loss
            
            g_loss.backward()
            g_opt.step()
            total_g += g_loss.item()
            
        print(f"Epoch {epoch+1} | D: {total_d/len(train_loader):.4f} | G: {total_g/len(train_loader):.4f}")
        
    torch.save(gen.state_dict(), 'gen_model.pth')
    print(" Model Saved")

if __name__ == "__main__":
    train_model()
