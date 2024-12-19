import numpy as np
import random
import os
import matplotlib.pyplot as plt
from pong import PongEnv  # Assicurati che pong.py sia nella stessa directory o nel path

# Configurazione del Q-Learning
ALPHA = 0.1
GAMMA = 0.99
EPSILON = 1.0

EPSILON_DECAY = 0.999
EPSILON_MIN = 0.05

EPISODES = 100000
DISCRETE_BINS = 10

TRAINING = True  # True per addestrare l'agente, False per testare l'agente

env = PongEnv(render_mode = not TRAINING)

# Nuova definizione dei bin per stato a 6 dimensioni:
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


def process_observation_player1(obs):
    # Player 1 usa l’osservazione così com’è
    return obs

def process_observation_player2(obs):
    player1_y, player2_y, ball_x, ball_y, ball_dx, ball_dy = obs
    # Ribaltiamo la prospettiva: consideriamo player2_y come se fosse player1_y (il suo paddle principale)
    # e player1_y come se fosse player2_y da questa prospettiva
    new_player1_y = player2_y
    new_player2_y = player1_y

    new_ball_x = env.SCREEN_WIDTH - ball_x
    new_ball_dx = -ball_dx
    new_ball_y = ball_y
    new_ball_dy = ball_dy

    # Ora restituiamo 6 valori, mantenendo la struttura (player1_y, player2_y, ball_x, ball_y, ball_dx, ball_dy)
    return (new_player1_y, new_player2_y, new_ball_x, new_ball_y, new_ball_dx, new_ball_dy)

rewards_player1_total = []
rewards_player2_total = []

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

        discrete_state1 = discrete_next_state1
        discrete_state2 = discrete_next_state2

        env.render()

    if EPSILON > EPSILON_MIN:
        EPSILON *= EPSILON_DECAY

    print(f"Episode {episode + 1}/{EPISODES}, Total Reward Player 1: {total_reward1}, Player 2: {total_reward2}")
    rewards_player1_total.append(total_reward1)
    rewards_player2_total.append(total_reward2)

    if (episode + 1) % 100 == 0:
        np.save("q_table_player1.npy", q_table_player1)
        np.save("q_table_player2.npy", q_table_player2)
        print(f"[INFO] Q-Table salvata al termine dell'episodio {episode + 1}.")

env.close()
np.save("q_table_player1.npy", q_table_player1)
np.save("q_table_player2.npy", q_table_player2)
print("[INFO] Q-Table salvata al termine dell'addestramento.")

def moving_average(data, window_size):
    return [np.mean(data[i:i+window_size]) for i in range(len(data)-window_size+1)]

avg_rewards_p1 = moving_average(rewards_player1_total, 100)
avg_rewards_p2 = moving_average(rewards_player2_total, 100)

episodes_axis = range(1, len(avg_rewards_p1) + 1)
plt.figure(figsize=(10,5))
plt.plot(episodes_axis, avg_rewards_p1, label='Player 1 (media mobile)')
plt.plot(episodes_axis, avg_rewards_p2, label='Player 2 (media mobile)')
plt.xlabel('Blocchi di 100 episodi')
plt.ylabel('Ricompensa Media')
plt.title('Andamento dei Reward Medi nel Tempo')
plt.legend()
plt.show()
