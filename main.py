from pongAI import modello

def main():
    """
    Funzione principale per l'addestramento e il testing dell'agente.
    """
    #print("[INFO] Inizio training...")
    #name1, name2 = modello(100000, True, 0.1, 0.99, "nulla", "nulla", False, False)

    print("[INFO] Training completato. Inizio testing...")
    modello(10000, False, 0,0, "p1_100k_alpha0.100_gamma0.990 (1).npy", "p2_100k_alpha0.100_gamma0.990 (1).npy", False, False)

    #Demo
    #modello(1000, False, 0,0, name1, name2, True)

    print("[INFO] Processo completato.")

if __name__ == "__main__":
    main()
