import pygame
import numpy as np
import gym
from gym import spaces
import random


class PongEnv(gym.Env):
    def __init__(self):
        super(PongEnv, self).__init__()

        # Dimensioni della finestra
        self.SCREEN_WIDTH = 800
        self.SCREEN_HEIGHT = 600
        self.PADDLE_WIDTH = 10
        self.PADDLE_HEIGHT = 100
        self.BALL_SIZE = 20
        self.PADDLE_SPEED = 5
        self.BALL_SPEED = 5  # Velocità palla in entrambe le direzioni

        # Variabili per il punteggio
        self.score_player1 = 0
        self.score_player2 = 0

        # Definire lo spazio di osservazione e di azione
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3),
                                            dtype=np.uint8)
        self.action_space = spaces.Discrete(3)  # 0: Stay, 1: Up, 2: Down

        # Inizializzazione di PyGame
        pygame.init()
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        pygame.display.set_caption("PongAI")

        self.clock = pygame.time.Clock()
        self.reset()

    # Funzione per resettare l'ambiente
    def reset(self):
        self.player1_y = (self.SCREEN_HEIGHT - self.PADDLE_HEIGHT) // 2
        self.player2_y = (self.SCREEN_HEIGHT - self.PADDLE_HEIGHT) // 2
        self.ball_x = self.SCREEN_WIDTH // 2
        self.ball_y = self.SCREEN_HEIGHT // 2

        # Direzioni casuali per la palla
        self.ball_dx = self.BALL_SPEED * random.choice([1, -1])
        self.ball_dy = self.BALL_SPEED * random.choice([1, -1])

        # Non resettiamo i punteggi
        self.done = False
        return self._get_obs()

    def step(self, action):
        # Gestione degli eventi (input dell'utente)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # Controllo paddle sinistro (Player 1)
        if action == 1 and self.player1_y > 0:
            self.player1_y -= self.PADDLE_SPEED
        if action == 2 and self.player1_y < self.SCREEN_HEIGHT - self.PADDLE_HEIGHT:
            self.player1_y += self.PADDLE_SPEED

        # Movimento della pallina
        self.ball_x += self.ball_dx
        self.ball_y += self.ball_dy

        # Collisioni con la parte superiore e inferiore dello schermo
        if self.ball_y <= 0 or self.ball_y >= self.SCREEN_HEIGHT - self.BALL_SIZE:
            self.ball_dy *= -1  # Inverti la direzione verticale

        reward = 0  # Inizializziamo la ricompensa

        # Collisioni con i paddle (aggiunta ricompensa per colpire la palla)
        if (
                self.ball_x <= self.PADDLE_WIDTH
                and self.player1_y < self.ball_y < self.player1_y + self.PADDLE_HEIGHT
        ):
            self.ball_dx *= -1  # Inverti la direzione orizzontale
            reward += 1  # Ricompensa positiva per colpire la palla

            # Bonus per un rimbalzo perfetto (se la palla colpisce il centro del paddle)
            if self.player1_y + self.PADDLE_HEIGHT / 4 <= self.ball_y <= self.player1_y + 3 * self.PADDLE_HEIGHT / 4:
                reward += 1  # Aggiungi ricompensa per un rimbalzo perfetto

        elif (
                self.ball_x >= self.SCREEN_WIDTH - self.PADDLE_WIDTH - self.BALL_SIZE
                and self.player2_y < self.ball_y < self.player2_y + self.PADDLE_HEIGHT
        ):
            self.ball_dx *= -1  # Inverti la direzione orizzontale
            reward += 1  # Ricompensa positiva per colpire la palla

            # Bonus per un rimbalzo perfetto (se la palla colpisce il centro del paddle)
            if self.player2_y + self.PADDLE_HEIGHT / 4 <= self.ball_y <= self.player2_y + 3 * self.PADDLE_HEIGHT / 4:
                reward += 1  # Aggiungi ricompensa per un rimbalzo perfetto

        # Se la palla non è colpita, punizione per non colpire la palla
        if self.ball_x < 0 or self.ball_x > self.SCREEN_WIDTH:
            reward -= 1  # Punizione per non colpire la palla

        # Se il paddle va fuori dallo schermo (punizione)
        if self.player1_y < 0 or self.player1_y > self.SCREEN_HEIGHT - self.PADDLE_HEIGHT:
            reward -= 2  # Punizione maggiore per aver sbagliato la posizione del paddle
        if self.player2_y < 0 or self.player2_y > self.SCREEN_HEIGHT - self.PADDLE_HEIGHT:
            reward -= 2  # Punizione maggiore per aver sbagliato la posizione del paddle

        # Punteggio (se la pallina supera i limiti laterali)
        if self.ball_x < 0:  # Player 1 ha segnato
            self.score_player1 += 1
            self.done = True
        if self.ball_x > self.SCREEN_WIDTH:  # Player 2 ha segnato
            self.score_player2 += 1
            self.done = True

        return self._get_obs(), reward, self.done, {}

    #Funzione per renderizzare l'ambiente
    def render(self, mode='human'):
        self.screen.fill((0, 0, 0))  # Pulisce lo schermo
        pygame.draw.rect(self.screen, (255, 255, 255),
                         (0, self.player1_y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT))  # Paddle sinistro
        pygame.draw.rect(self.screen, (255, 255, 255), (
        self.SCREEN_WIDTH - self.PADDLE_WIDTH, self.player2_y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT))  # Paddle destro
        pygame.draw.ellipse(self.screen, (255, 255, 255),
                            (self.ball_x, self.ball_y, self.BALL_SIZE, self.BALL_SIZE))  # Pallina

        # Visualizza il punteggio
        font = pygame.font.SysFont("Arial", 30)
        score_text = font.render(f"Player 1: {self.score_player1}   Player 2: {self.score_player2}", True,
                                 (255, 255, 255))
        self.screen.blit(score_text, (self.SCREEN_WIDTH // 2 - score_text.get_width() // 2, 20))

        pygame.display.flip()
        self.clock.tick(2000)

    # Funzione per ottenere l'osservazione
    def _get_obs(self):
        obs = pygame.surfarray.array3d(pygame.display.get_surface())
        return obs

    # Funzione per chiudere l'ambiente
    def close(self):
        pygame.quit()