# ============================================================================
# TREINAMENTO DE ROBÔ AUTÔNOMO COM DEEP Q-LEARNING (DQN)
# ============================================================================
# Este script implementa um agente de aprendizado por reforço profundo que
# treina um robô móvel a navegar autonomamente evitando obstáculos usando
# apenas sensores sonar como entrada.
# ============================================================================

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from typing import List
import random
from collections import deque

from simulation import Robot, World
from visualization import Visualizer


# ============================================================================
# REDE NEURAL Q-NETWORK
# ============================================================================
# Arquitetura da rede neural que aproxima a função Q(s,a)
# Entrada: 8 valores dos sensores sonar normalizados [0-1]
# Saída: 3 valores Q para cada ação (frente, esquerda, direita)
# ============================================================================
class SimpleQNetwork(nn.Module):
    """Rede neural feedforward para aproximar valores Q."""
    
    def __init__(self, input_size=8, hidden_size=64, output_size=3):
        super(SimpleQNetwork, self).__init__()
        # Camada 1: 8 entradas (sensores) -> 64 neurônios
        self.fc1 = nn.Linear(input_size, hidden_size)
        # Camada 2: 64 -> 64 neurônios (camada oculta)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # Camada 3: 64 -> 3 saídas (valores Q para cada ação)
        self.fc3 = nn.Linear(hidden_size, output_size)
        # Função de ativação ReLU para não-linearidade
        self.relu = nn.ReLU()
        
    def forward(self, x):
        """Propagação forward: sensores -> valores Q."""
        x = self.relu(self.fc1(x))  # Primeira camada + ativação
        x = self.relu(self.fc2(x))  # Segunda camada + ativação
        x = self.fc3(x)              # Camada de saída (sem ativação)
        return x


# ============================================================================
# AGENTE DQN (DEEP Q-NETWORK)
# ============================================================================
# Implementa o algoritmo DQN com:
# - Experience Replay: armazena experiências passadas para treinar
# - Target Network: rede separada para estabilizar o treinamento
# - Epsilon-Greedy: balanceia exploração vs. exploração
# ============================================================================
class DQNAgent:
    """Agente de aprendizado por reforço usando Deep Q-Network."""
    
    def __init__(self, state_size=8, action_size=3, lr=0.001):
        # Dimensões do problema
        self.state_size = state_size      # 8 sensores sonar
        self.action_size = action_size    # 3 ações possíveis
        
        # Experience Replay: memória de experiências (s, a, r, s', done)
        self.memory = deque(maxlen=10000)  # Armazena últimas 10k experiências
        
        # Epsilon-Greedy: controla exploração vs. exploração
        self.epsilon = 1.0          # Começa 100% exploração (aleatório)
        self.epsilon_min = 0.01     # Mínimo 1% exploração
        self.epsilon_decay = 0.995  # Decai 0.5% por episódio
        self.learning_rate = lr
        
        # Duas redes neurais (técnica DQN para estabilidade)
        self.q_network = SimpleQNetwork(state_size, 64, action_size)      # Rede principal (treina)
        self.target_network = SimpleQNetwork(state_size, 64, action_size) # Rede alvo (fixa)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Inicializa target network com mesmos pesos da q_network
        self.update_target_network()
        
    def update_target_network(self):
        """Copia pesos da Q-network para a Target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """Armazena experiência na memória para replay posterior."""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Seleciona ação usando estratégia epsilon-greedy.
        
        Com probabilidade epsilon: ação aleatória (exploração)
        Caso contrário: melhor ação segundo Q-network (exploração)
        """
        # Exploração: ação aleatória
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # Exploração: melhor ação segundo a rede neural
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())
    
    def replay(self, batch_size=32):
        """Treina a rede neural usando experiências passadas (Experience Replay).
        
        Algoritmo DQN:
        1. Amostra batch aleatório da memória
        2. Calcula Q(s,a) atual usando q_network
        3. Calcula Q-target = r + γ * max Q(s',a') usando target_network
        4. Minimiza erro entre Q atual e Q-target
        5. Decai epsilon (reduz exploração)
        """
        # Precisa de memória suficiente para amostrar
        if len(self.memory) < batch_size:
            return
        
        # Amostra batch aleatório de experiências
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])
        
        # Q(s,a) atual: valor Q da ação tomada
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # max Q(s',a'): melhor valor Q do próximo estado (usando target network)
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        
        # Q-target = r + γ * max Q(s',a') (Equação de Bellman)
        # Se done=True, Q-target = r (sem futuro)
        target_q_values = rewards + (0.99 * next_q_values * ~dones)
        
        # Calcula perda (MSE entre Q atual e Q-target)
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Backpropagation: atualiza pesos da rede
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decai epsilon: reduz exploração ao longo do tempo
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# ============================================================================
# TREINADOR DE REDE NEURAL COM REINFORCEMENT LEARNING
# ============================================================================
# Gerencia o loop de treinamento:
# - Executa episódios de simulação
# - Coleta experiências (estado, ação, recompensa)
# - Treina o agente DQN
# - Salva modelos periodicamente
# ============================================================================
class NeuralRLTrainer:
    """Gerencia o processo de treinamento do robô com DQN."""
    
    def __init__(self, num_episodes: int = 200, episode_duration: float = 30.0, 
                 max_velocity: float = 300.0):
        # Parâmetros de treinamento
        self.num_episodes = num_episodes          # Total de episódios
        self.episode_duration = episode_duration  # Duração máxima por episódio
        self.dt = 0.1                            # Timestep de 100ms
        self.max_velocity = max_velocity         # Velocidade máxima do robô
        
        # Cria ambiente de simulação (mundo 4x4m com 20 obstáculos)
        self.world = World(width=4000, height=4000, num_obstacles=20)
        
        # Cria visualizador para acompanhar treinamento em tempo real
        self.visualizer = Visualizer(self.world, scale=0.35)
        
        # Cria agente DQN (cérebro do robô)
        self.agent = DQNAgent()
        
        # Métricas de treinamento
        self.episode_rewards = []  # Recompensa acumulada por episódio
        self.episode_lengths = []  # Número de passos por episódio
        
    def run_episode(self, episode: int) -> tuple:
        """Executa um episódio completo de simulação.
        
        Loop de interação agente-ambiente:
        1. Observa estado (sensores sonar)
        2. Agente escolhe ação
        3. Executa ação no ambiente
        4. Recebe recompensa e novo estado
        5. Armazena experiência
        6. Treina rede neural
        """
        # Cria robô no centro do mundo com orientação aleatória
        robot = Robot(x=2000, y=2000, theta=np.random.uniform(0, 360), 
                     max_velocity=self.max_velocity)
        
        episode_reward = 0  # Recompensa acumulada
        sim_time = 0.0      # Tempo de simulação
        steps = 0           # Contador de passos
        
        # Estado inicial: leituras dos 8 sensores sonar normalizadas
        robot.update_sonar(self.world)
        state = robot.get_normalized_sonar()
        
        # Loop principal do episódio
        while sim_time < self.episode_duration and robot.alive:
            # 1. AGENTE DECIDE AÇÃO (epsilon-greedy)
            action = self.agent.act(state)
            
            # 2. EXECUTA AÇÃO NO AMBIENTE
            robot.update(action, self.dt, self.world)
            robot.update_sonar(self.world)
            
            # 3. OBSERVA NOVO ESTADO E RECOMPENSA
            next_state = robot.get_normalized_sonar()
            reward = robot.calculate_reward(self.dt)
            
            # Penalidade extra por colisão (terminal state)
            if not robot.alive:
                reward = -50.0
            
            episode_reward += reward
            done = not robot.alive
            
            # 4. ARMAZENA EXPERIÊNCIA (s, a, r, s', done)
            self.agent.remember(state, action, reward, next_state, done)
            
            # 5. ATUALIZA ESTADO
            state = next_state
            steps += 1
            
            # Atualiza visualização (60 FPS)
            running = self.visualizer.update([robot], episode, sim_time)
            if not running:
                raise KeyboardInterrupt("User closed window")
            
            sim_time += self.dt
        
        # 6. TREINA REDE NEURAL com batch de experiências
        if len(self.agent.memory) > 32:
            self.agent.replay(32)
        
        return episode_reward, steps, sim_time
    
    def train(self):
        """Loop principal de treinamento.
        
        Para cada episódio:
        1. Executa simulação (robô interage com ambiente)
        2. Coleta métricas (recompensa, passos, tempo)
        3. Atualiza target network periodicamente
        4. Salva modelo a cada 50 episódios
        """
        print("Starting Neural RL Training with DQN...")
        print(f"Episodes: {self.num_episodes}")
        print(f"Episode duration: {self.episode_duration}s")
        print()
        
        try:
            for episode in range(self.num_episodes):
                # Regenera obstáculos a cada 20 episódios (variabilidade)
                if episode > 0 and episode % 20 == 0:
                    self.world.generate_obstacles(20)
                
                # Executa um episódio completo
                reward, steps, time_survived = self.run_episode(episode)
                self.episode_rewards.append(reward)
                self.episode_lengths.append(steps)
                
                # Atualiza target network a cada 10 episódios (estabilidade)
                if episode % 10 == 0:
                    self.agent.update_target_network()
                
                # Imprime progresso (média móvel de 10 episódios)
                recent_avg = np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else reward
                print(f"Episode {episode}: Reward: {reward:.1f}, Steps: {steps}, "
                      f"Time: {time_survived:.1f}s, Epsilon: {self.agent.epsilon:.3f}, "
                      f"Avg10: {recent_avg:.1f}")
                
                # Salva checkpoint do modelo a cada 50 episódios
                if (episode + 1) % 50 == 0:
                    torch.save(self.agent.q_network.state_dict(), f'dqn_model_episode_{episode+1}.pth')
                    print(f"Model saved at episode {episode+1}")
        
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
        
        finally:
            # Salva modelo final e estatísticas
            torch.save(self.agent.q_network.state_dict(), 'dqn_model_final.pth')
            print(f"\nTraining completed!")
            print(f"Average reward: {np.mean(self.episode_rewards):.2f}")
            print(f"Average steps: {np.mean(self.episode_lengths):.1f}")
            print(f"Final epsilon: {self.agent.epsilon:.3f}")
            
            self.visualizer.close()


# ============================================================================
# FUNÇÃO PRINCIPAL
# ============================================================================
def main():
    """Inicializa e executa o treinamento do robô."""
    trainer = NeuralRLTrainer(
        num_episodes=200,        # 200 episódios de treinamento
        episode_duration=30.0,   # Máximo 30 segundos por episódio
        max_velocity=300.0       # Velocidade máxima 300 mm/s
    )
    trainer.train()


if __name__ == "__main__":
    main()
