import gymnasium as gym
import highway_env          
import numpy as np
import os
import sys
import time
import shutil
from gymnasium.wrappers import RecordVideo
from moviepy.editor import VideoFileClip, concatenate_videoclips
from src.config import Config
from src.agent import DQNAgent

def simulation():
    
    #1. MODEL SEÇİMİ (Yorum satırını kaldır)
    secilen_model_adi, secilen_model_dosyasi = "untrained", Config.FILENAME_UNTRAINED
    #secilen_model_adi, secilen_model_dosyasi = "half", Config.FILENAME_HALF
    #secilen_model_adi, secilen_model_dosyasi = "final", Config.FILENAME_FINAL

    #2. VİDEO KAYIT MODU
    VIDEO_KAYDET = True     # True: Akıcı kayıt yapar, False: Sadece izletir

    SIMULASYON_SAYISI = 6   
    FPS = 20                
    sleep_time = 1.0 / FPS  
    
    model_path = os.path.join(Config.MODEL_PATH, secilen_model_dosyasi)
    video_ana_klasor = "videos"
    model_video_klasor = os.path.join(video_ana_klasor, secilen_model_adi)
    gecici_video_klasor = os.path.join(model_video_klasor, "temp")

    print(f"\nMOD: {'VİDEO KAYIT (AKICI)' if VIDEO_KAYDET else 'SADECE İZLEME'}")
    print(f"MODEL: {secilen_model_adi}")

    if not os.path.exists(model_path):
        print(f"HATA: {model_path} bulunamadı!")
        return

    if VIDEO_KAYDET:
        if os.path.exists(gecici_video_klasor):
            shutil.rmtree(gecici_video_klasor) 
        os.makedirs(gecici_video_klasor, exist_ok=True)

    # Ortam Ayarları
    env_config = {
        "observation": {"type": "Kinematics"},
        "action": {"type": "DiscreteMetaAction"},
        "lanes_count": 4,
        "duration": 150,
        "vehicles_count": 30,
        "vehicles_density": 1.35,
        "reward_speed_range": [35, 50],
        "lane_change_reward": -1.2,
    }
    
    render_m = 'rgb_array' if VIDEO_KAYDET else 'human'
    env = gym.make(Config.ENV_NAME, render_mode=render_m, config=env_config)
    
    
    env.metadata["render_fps"] = FPS
    
    if VIDEO_KAYDET:
        env = RecordVideo(
            env, 
            video_folder=gecici_video_klasor,
            episode_trigger=lambda x: True,
            disable_logger=True
        )
    
    # Ajanı Hazırla
    obs, _ = env.reset()
    state_dim = np.prod(obs.shape)
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim)
    agent.load(model_path)
    
    try:
        for i in range(SIMULASYON_SAYISI):
            print(f"Tur {i+1}/{SIMULASYON_SAYISI} işleniyor...")
            state, _ = env.reset()
            state = state.flatten()
            done = False
            truncated = False
            
            while not (done or truncated):
                agent.epsilon = 0.0
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                state = next_state.flatten()
                

                if not VIDEO_KAYDET:
                    env.render()
                    time.sleep(sleep_time)
            
    except KeyboardInterrupt:
        print("\nDurduruldu.")
    finally:
        env.close()

    
    # AKICI VİDEO BİRLEŞTİRME
    
    if VIDEO_KAYDET:
        print("\nVideolar akıcı şekilde birleştiriliyor...")
        try:
            files = [os.path.join(gecici_video_klasor, f) for f in os.listdir(gecici_video_klasor) if f.endswith(".mp4")]
            files.sort() 
            
            clips = [VideoFileClip(f) for f in files]
            final_clip = concatenate_videoclips(clips, method="compose") # Compose metodu daha stabildir
            
            final_output = os.path.join(model_video_klasor, f"{secilen_model_adi}_combined.mp4")
            
            final_clip.write_videofile(
                final_output, 
                fps=FPS, 
                codec="libx264", 
                audio=False,
                logger=None 
            )
            
            for clip in clips: clip.close()
            shutil.rmtree(gecici_video_klasor)
            
            print(f"\nBAŞARILI: Akıcı video hazır -> {final_output}")
        except Exception as e:
            print(f"Video birleştirme hatası: {e}")

if __name__ == "__main__":
    simulation()