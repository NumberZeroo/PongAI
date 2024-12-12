import numpy as np
import random
import os
from pong import PongEnv  # Usa il tuo file pong.py

# Configurazione del Q-Learning
ALPHA = 0.3  # Tasso di apprendimento
GAMMA = 0.99  # Fattore di sconto
EPSILON = 1.0  # Probabilità iniziale di esplorazione
EPSILON_DECAY = 0.999
EPSILON_MIN = 0.05
EPISODES = 7000
DISCRETE_BINS = 10  # Numero di intervalli per discretizzare lo stato

# Funzione per discretizzare gli stati continui
def discretize_state(state, bins):
    """Trasforma lo stato continuo in uno stato discretizzato"""
    player_y, ball_x, ball_y, ball_dx, ball_dy = state
    discrete_state = (
        np.clip(np.digitize(player_y, bins[0]), 0, len(bins[0]) - 1),
        np.clip(np.digitize(ball_x, bins[1]), 0, len(bins[1]) - 1),
        np.clip(np.digitize(ball_y, bins[2]), 0, len(bins[2]) - 1),
        np.clip(np.digitize(ball_dx, bins[3]), 0, len(bins[3]) - 1),
        np.clip(np.digitize(ball_dy, bins[4]), 0, len(bins[4]) - 1),
    )
    return discrete_state

# Preparazione dell'ambiente
env = PongEnv()

# Creazione dei limiti dei bin per discretizzare lo spazio
bins = [
    np.linspace(0, env.SCREEN_HEIGHT, DISCRETE_BINS),  # Posizione paddle
    np.linspace(0, env.SCREEN_WIDTH, DISCRETE_BINS),  # Posizione x pallina
    np.linspace(0, env.SCREEN_HEIGHT, DISCRETE_BINS),  # Posizione y pallina
    np.linspace(-env.BALL_SPEED, env.BALL_SPEED, DISCRETE_BINS),  # Velocità palla (orizzontale)
    np.linspace(-env.BALL_SPEED, env.BALL_SPEED, DISCRETE_BINS),  # Velocità palla (verticale)
]

# Inizializza o carica le Q-Table
state_space_size = tuple([DISCRETE_BINS] * 5)
if os.path.exists("q_table_player1.npy"):
    q_table_player1 = np.load("q_table_player1.npy")
    print("[INFO] Q-Table Player 1 caricata da 'q_table_player1.npy'.")
else:
    q_table_player1 = np.random.uniform(low=-1, high=1, size=(state_space_size + (env.action_space.n,)))
    print("[INFO] Q-Table Player 1 inizializzata da zero.")

if os.path.exists("q_table_player2.npy"):
    q_table_player2 = np.load("q_table_player2.npy")
    print("[INFO] Q-Table Player 2 caricata da 'q_table_player2.npy'.")
else:
    q_table_player2 = np.random.uniform(low=-1, high=1, size=(state_space_size + (env.action_space.n,)))
    print("[INFO] Q-Table Player 2 inizializzata da zero.")

# Funzione per estrarre lo stato utile per ciascun giocatore
def process_observation_player1(obs):
    """Stato rilevante per il paddle sinistro (Player 1)"""
    return env.player1_y, env.ball_x, env.ball_y, env.ball_dx, env.ball_dy

def process_observation_player2(obs):
    """Stato rilevante per il paddle destro (Player 2)"""
    return env.player2_y, env.ball_x, env.ball_y, -env.ball_dx, env.ball_dy

# Ciclo di addestramento
for episode in range(EPISODES):
    obs = env.reset()  # Resetta l'ambiente e tutti gli stati
    state1 = process_observation_player1(obs)
    state2 = process_observation_player2(obs)
    discrete_state1 = discretize_state(state1, bins)
    discrete_state2 = discretize_state(state2, bins)

    done = False
    total_reward1 = 0
    total_reward2 = 0

    while not done:
        # Scegli le azioni con epsilon-greedy per entrambi i giocatori
        if random.uniform(0, 1) < EPSILON:
            action1 = env.action_space.sample()  # Esplorazione per Player 1
        else:
            action1 = np.argmax(q_table_player1[discrete_state1])  # Sfruttamento

        if random.uniform(0, 1) < EPSILON:
            action2 = env.action_space.sample()  # Esplorazione per Player 2
        else:
            action2 = np.argmax(q_table_player2[discrete_state2])  # Sfruttamento

        # Esegui le azioni
        env.player2_y += env.PADDLE_SPEED * (action2 - 1)  # Simula il paddle destro
        next_obs, rewards, done, _ = env.step(action1)
        reward_player1, reward_player2 = rewards

        total_reward1 += reward_player1
        total_reward2 += reward_player2

        # Aggiorna gli stati
        next_state1 = process_observation_player1(next_obs)
        next_state2 = process_observation_player2(next_obs)
        discrete_next_state1 = discretize_state(next_state1, bins)
        discrete_next_state2 = discretize_state(next_state2, bins)

        # Aggiorna la Q-Table per Player 1
        max_future_q1 = np.max(q_table_player1[discrete_next_state1])
        current_q1 = q_table_player1[discrete_state1 + (action1,)]
        new_q1 = current_q1 + ALPHA * (reward_player1 + GAMMA * max_future_q1 - current_q1)
        q_table_player1[discrete_state1 + (action1,)] = new_q1

        # Aggiorna la Q-Table per Player 2
        max_future_q2 = np.max(q_table_player2[discrete_next_state2])
        current_q2 = q_table_player2[discrete_state2 + (action2,)]
        new_q2 = current_q2 + ALPHA * (reward_player2 + GAMMA * max_future_q2 - current_q2)
        q_table_player2[discrete_state2 + (action2,)] = new_q2

        # Passa ai prossimi stati
        discrete_state1 = discrete_next_state1
        discrete_state2 = discrete_next_state2

        # Visualizzazione del gioco
        env.render()

    # Riduci EPSILON per diminuire l'esplorazione
    if EPSILON > EPSILON_MIN:
        EPSILON *= EPSILON_DECAY

    print(f"Episode {episode + 1}/{EPISODES}, Total Reward Player 1: {total_reward1}, Player 2: {total_reward2}")

    # Salva la Q-Table ogni 100 episodi
    if episode % 100 == 0:
        np.save("q_table_player1.npy", q_table_player1)
        np.save("q_table_player2.npy", q_table_player2)
        print("[INFO] Q-Table salvata al termine dell'episodio {episode}.")

# Chiudi l'ambiente
env.close()

# Salva le Q-Table alla fine
np.save("q_table_player1.npy", q_table_player1)
np.save("q_table_player2.npy", q_table_player2)
print("[INFO] Q-Table salvata al termine dell'addestramento.")
