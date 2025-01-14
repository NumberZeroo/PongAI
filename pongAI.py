import numpy as np
import random
import os

from grafici_utils import *
from pong import PongEnv

# Configurazione del Q-Learning
ALPHA = 0.1
ALPHA_DECAY = 0
ALPHA_MIN = 0.01

GAMMA = 0.99

#Dati per salvataggio table
al = ALPHA
ad = ALPHA_DECAY
g = GAMMA

EPSILON_MIN = 0.05

EPISODES = 100000
DISCRETE_BINS = 10

TRAINING = True  # True per addestrare l'agente, False per testare l'agente
ALPHA_DECAY_STATUS = False # True per abilitare il decay di alpha

if TRAINING:
    print("[INFO] Addestramento in corso...")
    print("-----------------------------------------------")
    EPSILON = 1.0
else:
    print("[INFO] Test in corso...")
    print("-----------------------------------------------")
    EPSILON = 0.0


env = PongEnv(render_mode = not TRAINING)

# (player1_y, player2_y, ball_x, ball_y, ball_dx, ball_dy)
bins = [
    np.linspace(0, env.SCREEN_HEIGHT, DISCRETE_BINS),   # player1_y
    np.linspace(0, env.SCREEN_HEIGHT, DISCRETE_BINS),   # player2_y
    np.linspace(0, env.SCREEN_WIDTH, DISCRETE_BINS),    # ball_x
    np.linspace(0, env.SCREEN_HEIGHT, DISCRETE_BINS),   # ball_y
    np.linspace(-env.BALL_SPEED, env.BALL_SPEED, DISCRETE_BINS), # ball_dx
    np.linspace(-env.BALL_SPEED, env.BALL_SPEED, DISCRETE_BINS), # ball_dy
]

def discretize_state(state, bins):
    """
    Funzione per discretizzare lo stato dell'ambiente.
    :param state: Stato dell'ambiente
    :param bins: Bins per la discretizzazione
    :return: Stato discretizzato (tuple)
    """

    player1_y, player2_y, ball_x, ball_y, ball_dx, ball_dy = state
    discrete_state = (
        np.digitize(player1_y, bins[0]) - 1,
        np.digitize(player2_y, bins[1]) - 1,
        np.digitize(ball_x, bins[2]) - 1,
        np.digitize(ball_y, bins[3]) - 1,
        np.digitize(ball_dx, bins[4]) - 1,
        np.digitize(ball_dy, bins[5]) - 1,
    )
    discrete_state = tuple(np.clip(discrete_state, 0, DISCRETE_BINS - 1))
    return discrete_state

state_space_size = tuple([DISCRETE_BINS] * 6)

if os.path.exists("q_table_player1.npy"):
    q_table_player1 = np.load("q_table_player1.npy")
    print("[INFO] Q-Table Player 1 caricata da 'q_table_player1.npy'.")
else:
    q_table_player1 = np.random.uniform(low=-1, high=1, size=(state_space_size + (env.action_space.n,)))
    print("[INFO] Q-Table Player 1 inizializzata casualmente.")

if os.path.exists("q_table_player2.npy"):
    q_table_player2 = np.load("q_table_player2.npy")
    print("[INFO] Q-Table Player 2 caricata da 'q_table_player2.npy'.")
else:
    q_table_player2 = np.random.uniform(low=-1, high=1, size=(state_space_size + (env.action_space.n,)))
    print("[INFO] Q-Table Player 2 inizializzata casualmente.")

print("-----------------------------------------------")

def process_observation_player1(obs):
    """
    Funzione per ottenere l'osservazione per il Player 1.
    :param obs: Osservazione dell'ambiente
    :return: Osservazione così com'è (per il Player 1)
    """

    return obs

def process_observation_player2(obs):
    """
    Funzione per ottenere l'osservazione per il Player 2.
    :param obs: Osservazione dell'ambiente
    :return: Osservazione ribaltata rispetto all'asse x (per il Player 2)
    """

    player1_y, player2_y, ball_x, ball_y, ball_dx, ball_dy = obs
    # Ribaltiamo la prospettiva: consideriamo player2_y come se fosse player1_y (il suo paddle principale)
    # e player1_y come se fosse player2_y da questa prospettiva
    new_player1_y = player2_y
    new_player2_y = player1_y

    new_ball_x = env.SCREEN_WIDTH - ball_x
    new_ball_dx = -ball_dx
    new_ball_y = ball_y
    new_ball_dy = ball_dy

    #Restituiamo 6 valori, mantenendo la struttura (player1_y, player2_y, ball_x, ball_y, ball_dx, ball_dy)
    return new_player1_y, new_player2_y, new_ball_x, new_ball_y, new_ball_dx, new_ball_dy

# Inizializzazione delle variabili per i grafici
rewards_player1_total = []
rewards_player2_total = []

touches_total = []

epsilon_history = []

wins_p1 = 0
wins_p2 = 0

EPSILON_DECAY = (EPSILON_MIN / EPSILON) ** (1 / EPISODES)

try: # Gestione interruzione con CTRL+C durante l'addestramento o il test
    for episode in range(EPISODES):
        obs = env.reset()

        # Applichiamo le trasformazioni
        state1 = process_observation_player1(obs)
        state2 = process_observation_player2(obs)

        discrete_state1 = discretize_state(state1, bins)
        discrete_state2 = discretize_state(state2, bins)

        done = False

        total_reward1 = 0
        total_reward2 = 0

        # Loop per un singolo episodio
        while not done:
            # Epsilon-greedy per Player 1
            if random.uniform(0, 1) < EPSILON:
                action1 = env.action_space.sample()
            else:
                action1 = np.argmax(q_table_player1[discrete_state1])

            # Epsilon-greedy per Player 2
            if random.uniform(0, 1) < EPSILON:
                action2 = env.action_space.sample()
            else:
                action2 = np.argmax(q_table_player2[discrete_state2])

            next_obs, rewards, done, _ = env.step(action1, action2)
            reward_player1, reward_player2 = rewards

            total_reward1 += reward_player1
            total_reward2 += reward_player2

            next_state1 = process_observation_player1(next_obs)
            next_state2 = process_observation_player2(next_obs)

            discrete_next_state1 = discretize_state(next_state1, bins)
            discrete_next_state2 = discretize_state(next_state2, bins)

            if TRAINING:
                # Aggiorna Player 1
                max_future_q1 = np.max(q_table_player1[discrete_next_state1])
                current_q1 = q_table_player1[discrete_state1 + (action1,)]
                new_q1 = current_q1 + ALPHA * (reward_player1 + GAMMA * max_future_q1 - current_q1)
                q_table_player1[discrete_state1 + (action1,)] = new_q1

                # Aggiorna Player 2
                max_future_q2 = np.max(q_table_player2[discrete_next_state2])
                current_q2 = q_table_player2[discrete_state2 + (action2,)]
                new_q2 = current_q2 + ALPHA * (reward_player2 + GAMMA * max_future_q2 - current_q2)
                q_table_player2[discrete_state2 + (action2,)] = new_q2

            # Aggiornamento degli stati
            discrete_state1 = discrete_next_state1
            discrete_state2 = discrete_next_state2

            env.render()

        if EPSILON > EPSILON_MIN:
            EPSILON *= EPSILON_DECAY

        if ALPHA_DECAY_STATUS:
            if ALPHA > ALPHA_MIN:
               ALPHA *= ALPHA_DECAY

        # Salvataggio delle variabili per i grafici
        rewards_player1_total.append(total_reward1)
        rewards_player2_total.append(total_reward2)

        epsilon_history.append(EPSILON)

        touches_total.append(env.touches)

        if total_reward1 > total_reward2:
            wins_p1 += 1
        elif total_reward2 > total_reward1:
            wins_p2 += 1

        # Stampa dei risultati ogni 500 episodi durante l'addestramento
        if (episode + 1) % 500 == 0:
            sum_r1 = sum(rewards_player1_total[-500:])
            sum_r2 = sum(rewards_player2_total[-500:])
            print(f"Episodio {episode + 1}/{EPISODES}, Total Reward Player 1: {sum_r1}, Player 2: {sum_r2}")

except KeyboardInterrupt:
    print("-----------------------------------------------")
    print("[INFO] Addestramento interrotto.")

finally:
    env.close()

if TRAINING:
    # Salvataggio finale delle Q-Tables
    q_table_filename_p1 = f"p1_{EPISODES // 1000}k_alpha{al:.3f}({ad:.3f})_gamma{g:.3f}.npy"
    q_table_filename_p2 = f"p2_{EPISODES // 1000}k_alpha{al:.3f}({ad:.3f})_gamma{g:.3f}.npy"

    np.save(f"qtable/p1/{q_table_filename_p1}", q_table_player1)
    np.save(f"qTable/p2/{q_table_filename_p2}", q_table_player2)
    print(f"[INFO] Q-Tables salvate al termine dell'addestramento come {q_table_filename_p1} e {q_table_filename_p2}")

    # Grafico dell'andamento delle ricompense medie
    avg_rewards(rewards_player1_total, rewards_player2_total, EPISODES, al, ad, g)

    # Grafico della percentuale di vittorie
    win_percentage(wins_p1, wins_p2, EPISODES, al, ad, g)

    # Grafico dei tocchi totali
    plot_touches(touches_total, EPISODES, al, ad, g)

    # Grafico dell'andamento di epsilon
    plot_epsilon_decay(epsilon_history, EPISODES, al, ad, g)

    print("-----------------------------------------------")
else:
    print("[INFO] Test completato.")
    print("-----------------------------------------------")