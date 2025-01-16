from pongAI import modello  # Importa le funzioni da pongAI.py

def main():
    """
    Funzione principale per l'addestramento e il testing dell'agente.
    """
    print("[INFO] Inizio training...")
    name1, name2 = modello(100000, True, 0.1, 0.99, "nulla", "nulla", False, True)

    print("[INFO] Training completato. Inizio testing...")
    modello(1000, False, 0,0, name1, name2, False, False)

    #Demo
    #modello(1000, False, 0,0, name1, name2, True)

    print("[INFO] Processo completato.")

if __name__ == "__main__":
    main()
