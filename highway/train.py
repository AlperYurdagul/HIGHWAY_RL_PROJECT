import gymnasium as gym
import highway_env
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import sys  
from src.config import Config
from src.agent import DQNAgent

def train():
    
    os.makedirs(Config.MODEL_PATH, exist_ok=True)

    # --- ORTAM AYARLARI VE CEZALAR ---
    
    env_config = {
        "observation": {"type": "Kinematics"},
        "action": {"type": "DiscreteMetaAction"},
        "lanes_count": 4,          
        "duration": 90,            
        
        
        "vehicles_count": 30,      
        "vehicles_density": 1.35,   
        
        
        "collision_reward": -125.0,  
        
        
        "reward_speed_range": [35,50], 
        "high_speed_reward": 1,       
        
        
        "lane_change_reward": -1.2,      
        "right_lane_reward": 0.0,      
    }

    print(f"Çoklu Çekirdek Eğitimi Başlıyor!")
    print(f"Çekirdek Sayısı: {Config.NUM_ENVS}")
    print(f"Hedef: {Config.TOTAL_TIMESTEPS} Adım")
    print("-" * 50)

    # --- PARALEL ORTAMLARI OLUŞTUR ---
    envs = gym.make_vec(
        Config.ENV_NAME, 
        num_envs=Config.NUM_ENVS, 
        vectorization_mode="async", 
        render_mode=None,
        config=env_config
    )

    # İlk durumu al ve ajanı hazırla
    obs, _ = envs.reset()
    single_obs_shape = np.prod(obs.shape[1:]) 
    n_actions = envs.single_action_space.n

    agent = DQNAgent(single_obs_shape, n_actions)

    # --- AŞAMA 1: UNTRAINED ---
    agent.save(os.path.join(Config.MODEL_PATH, Config.FILENAME_UNTRAINED))
    print(f"Başlangıç Modeli Kaydedildi: {Config.FILENAME_UNTRAINED}")

    rewards_history = []
    episode_rewards = np.zeros(Config.NUM_ENVS)
    
    global_step = 0
    
    # --- EĞİTİM DÖNGÜSÜ ---
    while global_step < Config.TOTAL_TIMESTEPS:
        
        
        states = obs.reshape(Config.NUM_ENVS, -1)

        
        epsilon = Config.EPSILON_END + (Config.EPSILON_START - Config.EPSILON_END) * \
                  np.exp(-1. * global_step / Config.EPSILON_DECAY_STEPS)
        agent.epsilon = epsilon

        
        actions = []
        for i in range(Config.NUM_ENVS):
            actions.append(agent.select_action(states[i]))
        
        
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        
        next_states = next_obs.reshape(Config.NUM_ENVS, -1)
        dones = terminations | truncations

        
        for i in range(Config.NUM_ENVS):
            agent.remember(states[i], actions[i], rewards[i], next_states[i], dones[i])
            episode_rewards[i] += rewards[i]
            
            if dones[i]:
                rewards_history.append(episode_rewards[i])
                episode_rewards[i] = 0

        
        agent.replay()

        obs = next_obs
        global_step += Config.NUM_ENVS 

        
        if global_step % Config.TARGET_UPDATE < Config.NUM_ENVS:
            agent.update_target_network()

        
        
        if global_step % 100 < Config.NUM_ENVS:
            avg_rew = np.mean(rewards_history[-50:]) if len(rewards_history) > 0 else 0
            progress = (global_step / Config.TOTAL_TIMESTEPS) * 100
            
            # Tek satırda sürekli güncellenen çıktı (Carriage Return \r)
            # Eğer terminalde alt alta yazsın istersen 'end="\r"' kısmını silip normal print yapabilirsin.
            sys.stdout.write(f"\rİlerleme: %{progress:.1f} | Adım: {global_step}/{Config.TOTAL_TIMESTEPS} | Ort. Ödül: {avg_rew:.2f} | Epsilon: {epsilon:.3f}")
            sys.stdout.flush()

        # --- AŞAMA 2: HALF-TRAINED ---
        if global_step >= Config.TOTAL_TIMESTEPS // 2 and \
           global_step < (Config.TOTAL_TIMESTEPS // 2 + Config.NUM_ENVS * 2):
             path = os.path.join(Config.MODEL_PATH, Config.FILENAME_HALF)
             if not os.path.exists(path): 
                 agent.save(path)
                 print(f"\nYarı-Yol Modeli Kaydedildi: {Config.FILENAME_HALF}")

    # --- AŞAMA 3: FINAL ---
    print("\n" + "-"*50)
    agent.save(os.path.join(Config.MODEL_PATH, Config.FILENAME_FINAL))
    print(f"Final Modeli Kaydedildi: {Config.FILENAME_FINAL}")
    
    envs.close()
    
    # --- GRAFİK ÇİZME ---
    plt.figure(figsize=(10, 5))
    if len(rewards_history) > 0:
        # 'Ham Veri' -> 'Raw Data'
        plt.plot(rewards_history, alpha=0.3, label='Raw Data')
        
        window = 50
        if len(rewards_history) > window:
            smooth = np.convolve(rewards_history, np.ones(window)/window, mode='valid')
            # 'Ortalama' -> 'Moving Average'
            plt.plot(smooth, color='red', label='Moving Average')
    
    # Başlık ve Eksen İsimleri İngilizceye Çevrildi
    plt.title(f"Training Progress ({Config.NUM_ENVS} Cores)")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.legend()
    
    plt.savefig("training_graph.png")
    print("\nGraph saved: training_graph.png")

if __name__ == "__main__":
    train()