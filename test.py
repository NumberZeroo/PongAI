import numpy as np

# Carica le Q-Tables
q_table_p1 = np.load("qTable/p1/p1_100k_alpha0.100_gamma0.990.npy")
q_table_p2 = np.load("qTable/p2/p2_100k_alpha0.100_gamma0.990.npy")

# Statistiche delle Q-Tables
print("Player 1 Q-Table Statistics:")
print("Mean:", np.mean(q_table_p1))
print("Std Dev:", np.std(q_table_p1))
print("Min:", np.min(q_table_p1))
print("Max:", np.max(q_table_p1))

print("\nPlayer 2 Q-Table Statistics:")
print("Mean:", np.mean(q_table_p2))
print("Std Dev:", np.std(q_table_p2))
print("Min:", np.min(q_table_p2))
print("Max:", np.max(q_table_p2))
