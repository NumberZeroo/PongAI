import numpy as np
from matplotlib import pyplot as plt

# Funzione per calcolare la media mobile di una lista di dati
def moving_average(data, window_size):
    return [np.mean(data[i:i + window_size]) for i in range(len(data) - window_size + 1)]


# Funzione per visualizzare l'andamento delle ricompense medie nel tempo
def avg_rewards(rewards_player1_total, rewards_player2_total, EPISODES, al, g):
    """
    Crea un grafico dell'andamento delle ricompense medie nel tempo.
    :param rewards_player1_total: Ricompense totali del Player 1
    :param rewards_player2_total: Ricompense totali del Player 2
    :param EPISODES: Numero totale di episodi
    :param al: valore di alpha
    :param g: valore di gamma
    :return: None
    """
    winsize = 500
    avg_rewards_p1 = moving_average(rewards_player1_total, winsize)
    avg_rewards_p2 = moving_average(rewards_player2_total, winsize)

    episodes_axis = range(1, len(avg_rewards_p1) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(episodes_axis, avg_rewards_p1, label='Player 1 (media mobile)')
    plt.plot(episodes_axis, avg_rewards_p2, label='Player 2 (media mobile)')
    plt.xlabel(f'Blocchi di {winsize} episodi')
    plt.ylabel('Ricompensa Media')
    plt.title('Andamento dei Reward Medi nel Tempo (Training)')
    plt.legend()

    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # plt.savefig(f"grafici/andamento_reward_{timestamp}.png")
    plt.savefig(f"grafici/avg_rewards/avgRew_{EPISODES // 1000}k_alpha{al:.3f}_gamma{g:.3f}.png")

    plt.show()

# Funzione per visualizzare l'andamento delle ricompense medie nel tempo
def avg_rewards_testing(rewards_player1_total, rewards_player2_total, EPISODES, al, g, winsize):
    """
    Crea un grafico dell'andamento delle ricompense medie nel tempo.
    :param rewards_player1_total: Ricompense totali del Player 1
    :param rewards_player2_total: Ricompense totali del Player 2
    :param EPISODES: Numero totale di episodi
    :param al: valore di alpha
    :param g: valore di gamma
    :return: None
    """

    avg_rewards_p1 = moving_average(rewards_player1_total, winsize)
    avg_rewards_p2 = moving_average(rewards_player2_total, winsize)

    episodes_axis = range(1, len(avg_rewards_p1) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(episodes_axis, avg_rewards_p1, label='Player 1 (media mobile)')
    plt.plot(episodes_axis, avg_rewards_p2, label='Player 2 (media mobile)')
    plt.xlabel(f'Blocchi di {winsize} episodi')
    plt.ylabel('Ricompensa Media')
    plt.title('Andamento dei Reward Medi nel Tempo (Testing)')
    plt.legend()

    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # plt.savefig(f"grafici/andamento_reward_{timestamp}.png")
    plt.savefig(f"grafici/avg_rewards_testing/testing_{EPISODES}k_alpha{al:.3f}_gamma{g:.3f} ({winsize}).png")

    plt.show()

# Funzione per analizzare la percentuale di vittorie di un giocatore
def win_percentage(wins_player1, wins_player2, EPISODES, al, g, testing):
    """
    Crea un grafico a torta per visualizzare la percentuale di vittorie tra i due giocatori.
    :param wins_player1: Totale delle vittorie del Player 1
    :param wins_player2: Totale delle vittorie del Player 2
    :param EPISODES: Numero totale di episodi
    :param al: Valore di alpha
    :param g: Valore di gamma
    :return: None
    """

    # Calcola la percentuale di vittorie
    total_episodes = wins_player1 + wins_player2
    win_percentage_p1 = (wins_player1 / total_episodes) * 100
    win_percentage_p2 = (wins_player2 / total_episodes) * 100

    print(f"Player 1: {wins_player1} vittorie ({win_percentage_p1:.2f}%)")
    print(f"Player 2: {wins_player2} vittorie ({win_percentage_p2:.2f}%)")

    # Dati per il grafico
    labels = [
        f'Player 1\n{wins_player1} vittorie\n({win_percentage_p1:.1f}%)',
        f'Player 2\n{wins_player2} vittorie\n({win_percentage_p2:.1f}%)'
    ]
    sizes = [win_percentage_p1, win_percentage_p2]
    colors = ['#1f77b4', '#ff7f0e']  # Colori più armoniosi
    explode = (0.1, 0)

    # Grafico a torta
    plt.figure(figsize=(8, 8))
    wedges, texts, autotexts = plt.pie(
        sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, explode=explode,
        textprops=dict(color="black"), pctdistance=0.85
    )

    # Migliora il design delle etichette
    for text in texts:
        text.set_fontsize(12)  # Dimensione del testo delle etichette
    for autotext in autotexts:
        autotext.set_fontsize(10)  # Dimensione del testo delle percentuali
        autotext.set_color("white")  # Percentuale bianca per leggibilità

    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    plt.gca().add_artist(centre_circle)

    plt.title('Percentuale di Vittorie tra i Player', fontsize=16, weight='bold')
    plt.tight_layout()

    # Salvataggio del grafico
    if testing:
        plt.savefig(f"grafici/win_percentage/TESTING_winPerc_{EPISODES // 1000}k_alpha{al:.3f}_gamma{g:.3f}.png")
    else:
        plt.savefig(f"grafici/win_percentage/TRAINING_winPerc_{EPISODES // 1000}k_alpha{al:.3f}_gamma{g:.3f}.png")

    plt.show()

def plot_touches(touches_total, EPISODES, al, g, window_size=300):
    """
    Crea un grafico dell'andamento degli scambi medi nel tempo.
    :param touches_total: Numero totale di tocchi per episodio
    :param EPISODES: Numero totale di episodi
    :param al: Valore di alpha
    :param g: Valore di gamma
    :param window_size: Dimensione della finestra per la media mobile
    :return: None
    """

    avg_touches = moving_average(touches_total, window_size)  # Calcola la media mobile
    episodes_axis = range(1, len(avg_touches) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(episodes_axis, avg_touches, label=f'Media scambi (finestra={window_size})', color='green')
    plt.xlabel('Blocchi di episodi')
    plt.ylabel('Scambi Medi')
    plt.title('Andamento degli Scambi Medi nel Tempo')
    plt.legend()

    # Salva il grafico
    plt.savefig(f"grafici/avg_touches/avgTouch_{EPISODES // 1000}k_alpha{al:.3f}_gamma{g:.3f}.png")
    plt.show()

def plot_epsilon_decay(epsilon_history, EPISODES, al, g):
    """
    Crea un grafico dell'andamento del decadimento di epsilon.
    :param epsilon_history: Storico dei valori di epsilon
    :param EPISODES: Numero totale di episodi
    :param al: Valore di alpha
    :param g: Valore di gamma
    :return: None
    """

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(epsilon_history) + 1), epsilon_history, label='Valore di Epsilon', color='red')
    plt.xlabel('Episodi')
    plt.ylabel('Epsilon')
    plt.title('Andamento del Decadimento di Epsilon')
    plt.legend()
    plt.savefig(f"grafici/epsilon_decay/epsDec_{EPISODES // 1000}k_alpha{al:.3f}_gamma{g:.3f}.png")
    plt.show()
