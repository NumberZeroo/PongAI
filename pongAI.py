import numpy as np
import random
import os
import tqdm

from grafici_utils import *
from pong import PongEnv
from tqdm import tqdm

def modello(episodes, training, alpha, gamma, q_table_p1, q_table_p2, demo_status, decay):
    """
    Funzione per l'addestramento e il testing dell'agente.
    :param episodes: Numero di episodi
    :param training: Se True allora addestramento, altrimenti testing
    :param alpha: Valore di alpha, learning rate
    :param gamma: Valore di gamma, fattore di sconto
    :param q_table_p1: Nome del file per la Q-Table del Player 1 (se esiste)
    :param q_table_p2: Nome del file per la Q-Table del Player 2 (se esiste)
    :param demo_status: Se True allora demo, altrimenti no
    :param decay: Se True allora epsilon decay basato su episodi, altrimenti no
    :return: Nomi dei file delle Q-Table salvate (se in fase di addestramento)
    """
    epsilon_min = 0.05
    discrete_bins = 10

    #Dati per salvataggio table
    al = alpha
    g = gamma
    
    if training:
        print("[INFO] Addestramento in corso...")
        print("-----------------------------------------------")
        epsilon = 1.0

        if decay:
            epsilon_decay = (epsilon_min / epsilon) ** (1 / episodes)
        else:
            epsilon_decay = 0.999
    else:
        print("[INFO] Test in corso...")
        print("-----------------------------------------------")
        epsilon = 0.0
        epsilon_decay = 0.0

    env = PongEnv(render_mode = demo_status)

    # (player1_y, player2_y, ball_x, ball_y, ball_dx, ball_dy)
    bins = [
        np.linspace(0, env.SCREEN_HEIGHT, discrete_bins),   # player1_y
        np.linspace(0, env.SCREEN_HEIGHT, discrete_bins),   # player2_y
        np.linspace(0, env.SCREEN_WIDTH, discrete_bins),    # ball_x
        np.linspace(0, env.SCREEN_HEIGHT, discrete_bins),   # ball_y
        np.linspace(-env.BALL_SPEED, env.BALL_SPEED, discrete_bins), # ball_dx
        np.linspace(-env.BALL_SPEED, env.BALL_SPEED, discrete_bins), # ball_dy
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
        discrete_state = tuple(np.clip(discrete_state, 0, discrete_bins - 1))
        return discrete_state
    
    state_space_size = tuple([discrete_bins] * 6)
    
    if os.path.exists(f"qTable/p1/{q_table_p1}"):
        q_table_player1 = np.load(f"qTable/p1/{q_table_p1}")
        print("[INFO] Q-Table Player 1 caricata.")
    else:
        q_table_player1 = np.random.uniform(low=-1, high=1, size=(state_space_size + (env.action_space.n,)))
        print("[INFO] Q-Table Player 1 inizializzata casualmente.")
    
    if os.path.exists(f"qTable/p2/{q_table_p2}"):
        q_table_player2 = np.load(f"qTable/p2/{q_table_p2}")
        print("[INFO] Q-Table Player 2 caricata.")
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

    try: # Gestione interruzione con CTRL+C durante l'addestramento o il test

        # Loop per tutti gli episodi di addestramento o test (episodes)
        bar = tqdm(range(episodes), desc="[INFO] Episodi in corso", unit="episodi")
        for episode in bar:
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
                # epsilon-greedy per Player 1
                if random.uniform(0, 1) < epsilon:
                    action1 = env.action_space.sample()
                else:
                    action1 = np.argmax(q_table_player1[discrete_state1])
    
                # epsilon-greedy per Player 2
                if random.uniform(0, 1) < epsilon:
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
    
                if training:
                    # Aggiorna Player 1
                    max_future_q1 = np.max(q_table_player1[discrete_next_state1])
                    current_q1 = q_table_player1[discrete_state1 + (action1,)]
                    new_q1 = current_q1 + alpha * (reward_player1 + gamma * max_future_q1 - current_q1)
                    q_table_player1[discrete_state1 + (action1,)] = new_q1
    
                    # Aggiorna Player 2
                    max_future_q2 = np.max(q_table_player2[discrete_next_state2])
                    current_q2 = q_table_player2[discrete_state2 + (action2,)]
                    new_q2 = current_q2 + alpha * (reward_player2 + gamma * max_future_q2 - current_q2)
                    q_table_player2[discrete_state2 + (action2,)] = new_q2
    
                # Aggiornamento degli stati
                discrete_state1 = discrete_next_state1
                discrete_state2 = discrete_next_state2
    
                env.render()
    
            if epsilon > epsilon_min:
                epsilon *= epsilon_decay
    
            # Salvataggio delle variabili per i grafici
            rewards_player1_total.append(total_reward1)
            rewards_player2_total.append(total_reward2)
    
            epsilon_history.append(epsilon)
    
            touches_total.append(env.touches)
    
            if total_reward1 > total_reward2:
                wins_p1 += 1
            else:
                wins_p2 += 1

            bar.set_postfix({"Reward P1": sum(rewards_player1_total[-50:]), "Reward P2": sum(rewards_player2_total[-50:])})
    
    except KeyboardInterrupt:
        print("-----------------------------------------------")
        print("[INFO] Addestramento interrotto.")
    
    finally:
        env.close()
    
    if training:
        # Salvataggio finale delle Q-Tables
        q_table_filename_p1 = f"p1_{episodes // 1000}k_alpha{al:.3f}_gamma{g:.3f}.npy"
        q_table_filename_p2 = f"p2_{episodes // 1000}k_alpha{al:.3f}_gamma{g:.3f}.npy"
    
        np.save(f"qtable/p1/{q_table_filename_p1}", q_table_player1)
        np.save(f"qTable/p2/{q_table_filename_p2}", q_table_player2)
        print(f"[INFO] Q-Tables salvate al termine dell'addestramento come {q_table_filename_p1} e {q_table_filename_p2}")
    
        # Grafico dell'andamento delle ricompense medie
        avg_rewards(rewards_player1_total, rewards_player2_total, episodes, al, g)

        # Grafico della percentuale di vittorie
        win_percentage(wins_p1, wins_p2, episodes, al, g)
    
        # Grafico dei tocchi totali
        plot_touches(touches_total, episodes, al, g)
    
        # Grafico dell'andamento di epsilon
        plot_epsilon_decay(epsilon_history, episodes, al, g)
        print("-----------------------------------------------")

        return q_table_filename_p1, q_table_filename_p2
    else:
        # Grafico dell'andamento delle ricompense medie (testing)
        avg_rewards_testing(rewards_player1_total, rewards_player2_total, episodes, al, g)
        print("[INFO] Test completato.")
        print("-----------------------------------------------")