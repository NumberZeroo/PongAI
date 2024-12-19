# pongAI.py
import numpy as np
import random
import os
from pong import PongEnv  # Assicurati che pong.py sia nella stessa directory o nel path

# Configurazione del Q-Learning
ALPHA = 0.1  # Tasso di apprendimento (Learning Rate)
GAMMA = 0.99  # Fattore di sconto per le future ricompense (Discount Factor)
EPSILON = 0.0  # Probabilità iniziale di esplorazione (1 = 100%)

EPSILON_DECAY = 0.999  # Fattore di decadimento per l'esplorazione
EPSILON_MIN = 0.05  # Probabilità minima di esplorazione

EPISODES = 100000  # Numero di episodi di addestramento
DISCRETE_BINS = 10  # Numero di intervalli per discretizzare lo stato

TRAINING = False  # True per addestrare l'agente, False per testare l'agente


# Funzione per discretizzare gli stati continui
def discretize_state(state, bins):
    """
    Trasforma lo stato continuo in uno stato discretizzato.

    Args:
        state (tuple): Stato continuo (player1_y, ball_x, ball_y, ball_dx, ball_dy).
        bins (list): Lista di array di bin per ogni componente dello stato.

    Returns:
        tuple: Stato discretizzato.
    """
    player_y, ball_x, ball_y, ball_dx, ball_dy = state
    discrete_state = (
        np.digitize(player_y, bins[0]) - 1,
        np.digitize(ball_x, bins[1]) - 1,
        np.digitize(ball_y, bins[2]) - 1,
        np.digitize(ball_dx, bins[3]) - 1,
        np.digitize(ball_dy, bins[4]) - 1,
    )
    # Assicurati che gli indici siano entro i limiti
    discrete_state = tuple(np.clip(discrete_state, 0, DISCRETE_BINS - 1))
    return discrete_state


# Preparazione dell'ambiente
env = PongEnv(render_mode= not TRAINING)

# Creazione dei limiti dei bin per discretizzare lo spazio
bins = [
    np.linspace(0, env.SCREEN_HEIGHT, DISCRETE_BINS),  # Posizione paddle
    np.linspace(0, env.SCREEN_WIDTH, DISCRETE_BINS),  # Posizione x palla
    np.linspace(0, env.SCREEN_HEIGHT, DISCRETE_BINS),  # Posizione y palla
    np.linspace(-env.BALL_SPEED, env.BALL_SPEED, DISCRETE_BINS),  # Velocità palla (orizzontale)
    np.linspace(-env.BALL_SPEED, env.BALL_SPEED, DISCRETE_BINS),  # Velocità palla (verticale)
]

# Inizializza o carica la Q-Table per Player 1 (Agente IA)
state_space_size = tuple([DISCRETE_BINS] * 5)
if os.path.exists("q_table_player1.npy"):
    q_table_player1 = np.load("q_table_player1.npy")
    print("[INFO] Q-Table Player 1 caricata da 'q_table_player1.npy'.")
else:
    q_table_player1 = np.random.uniform(low=-1, high=1, size=(state_space_size + (env.action_space.n,)))
    print("[INFO] Q-Table Player 1 inizializzata casualmente.")

# Inizializza o carica la Q-Table per Player 2 (Agente IA)
if os.path.exists("q_table_player2.npy"):
    q_table_player2 = np.load("q_table_player2.npy")
    print("[INFO] Q-Table Player 2 caricata da 'q_table_player2.npy'.")
else:
    q_table_player2 = np.random.uniform(low=-1, high=1, size=(state_space_size + (env.action_space.n,)))
    print("[INFO] Q-Table Player 2 inizializzata casualmente.")


# Funzione per estrarre lo stato utile per ciascun giocatore
def process_observation_player1(obs):
    """
    Stato rilevante per il paddle sinistro (Player 1).

    Args:
        obs (tuple): Osservazione dall'ambiente.

    Returns:
        tuple: Stato per Player 1.
    """
    return obs  # In questo caso, l'intera osservazione è rilevante


def process_observation_player2(obs):
    """
    Stato rilevante per il paddle destro (Player 2).

    Args:
        obs (tuple): Osservazione dall'ambiente.

    Returns:
        tuple: Stato per Player 2.
    """
    return obs  # In questo caso, l'intera osservazione è rilevante


# Ciclo di addestramento
for episode in range(EPISODES):
    obs = env.reset()  # Resetta l'ambiente e tutti gli stati

    # Prepara gli stati iniziali
    state1 = process_observation_player1(obs)
    state2 = process_observation_player2(obs)

    discrete_state1 = discretize_state(state1, bins)
    discrete_state2 = discretize_state(state2, bins)

    done = False  # Flag per indicare la fine dell'episodio

    # Inizializza il totale delle ricompense per ciascun giocatore
    total_reward1 = 0
    total_reward2 = 0

    # Loop per l'intero episodio
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
        next_obs, rewards, done, _ = env.step(action1, action2)
        reward_player1, reward_player2 = rewards

        total_reward1 += reward_player1
        total_reward2 += reward_player2

        # Aggiorna gli stati
        next_state1 = process_observation_player1(next_obs)
        next_state2 = process_observation_player2(next_obs)
        discrete_next_state1 = discretize_state(next_state1, bins)
        discrete_next_state2 = discretize_state(next_state2, bins)

        if TRAINING:
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
    if (episode + 1) % 100 == 0:
        np.save("q_table_player1.npy", q_table_player1)
        np.save("q_table_player2.npy", q_table_player2)
        print(f"[INFO] Q-Table salvata al termine dell'episodio {episode + 1}.")

# Chiudi l'ambiente
env.close()

# Salva le Q-Table alla fine dell'addestramento
np.save("q_table_player1.npy", q_table_player1)
np.save("q_table_player2.npy", q_table_player2)
print("[INFO] Q-Table salvata al termine dell'addestramento.")
