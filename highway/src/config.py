import torch
import os

class Config:
    # --- YENİ: ÇOKLU ÇEKİRDEK AYARI ---
    NUM_ENVS = min(os.cpu_count(), 8) 

    # Ortam Ayarları
    ENV_NAME = "highway-fast-v0"
    RENDER_MODE = None 

    # Eğitim Hiperparametreleri
    TOTAL_TIMESTEPS = 500000      
    MAX_STEPS = 200                
    
    BATCH_SIZE = 256               
    LR = 5e-4                      
    GAMMA = 0.99                   
    
    EPSILON_START = 1.0
    EPSILON_END = 0.05
    # Adım sayısına göre azalacak
    EPSILON_DECAY_STEPS = TOTAL_TIMESTEPS * 0.4 
    
    MEMORY_CAPACITY = 50000        
    TARGET_UPDATE = 1000            
    HIDDEN_SIZE = 256              
    
    # Dosya Yolları
    MODEL_PATH = "models"
    FILENAME_UNTRAINED = "dqn_highway_untrained.pth"
    FILENAME_HALF = "dqn_highway_half_trained.pth"
    FILENAME_FINAL = "dqn_highway_final.pth"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")