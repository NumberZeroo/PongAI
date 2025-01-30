from pongAI import modello

def main():
    """
    Funzione principale per l'addestramento e il testing dell'agente.
    """

    episodes_train = 100000
    episodes_test = 10000
    alpha = 0.1
    gamma = 0.99
    decay_ep = True

    print("[INFO] Inizio training...")
    name1, name2 = modello(episodes_train, True, alpha, gamma, "nulla", "nulla", False, decay_ep)

    print("[INFO] Training completato. Inizio testing...")
    modello(episodes_test, False, alpha, gamma, name1, name2, False, decay_ep)

    #Demo
    #modello(10, False, 0,0, "", "", True, False)

    print("[INFO] Processo completato.")

if __name__ == "__main__":
    main()
