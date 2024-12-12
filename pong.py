import pygame
import random
import numpy as np
from gym import Env, spaces

class PongEnv(Env):
    def __init__(self):
        super(PongEnv, self).__init__()

        # Dimensioni della finestra
        self.SCREEN_WIDTH = 800
        self.SCREEN_HEIGHT = 600
        self.PADDLE_WIDTH = 10
        self.PADDLE_HEIGHT = 100
        self.BALL_SIZE = 20
        self.PADDLE_SPEED = 5
        self.BALL_SPEED = 5

        # Punteggio
        self.score_player1 = 0
        self.score_player2 = 0

        # Numero di tocchi per il progresso della velocità
        self.touches = 0

        # Flag per verificare se i paddle hanno toccato la palla
        self.paddle1_touched = False
        self.paddle2_touched = False

        # Spazio osservazione e azione
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.SCREEN_HEIGHT, self.SCREEN_WIDTH, 3), dtype=np.uint8)
        self.action_space = spaces.Discrete(3)  # 0: Stay, 1: Up, 2: Down

        # PyGame
        pygame.init()
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        pygame.display.set_caption("PongAI")

        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.player1_y = (self.SCREEN_HEIGHT - self.PADDLE_HEIGHT) // 2
        self.player2_y = (self.SCREEN_HEIGHT - self.PADDLE_HEIGHT) // 2
        self.ball_x = self.SCREEN_WIDTH // 2
        self.ball_y = self.SCREEN_HEIGHT // 2

        # Direzione casuale per la palla
        self.ball_dx = random.choice([-1, 1]) * random.randint(3, 7)
        self.ball_dy = random.choice([-1, 1]) * random.randint(2, 5)

        self.touches = 0
        self.paddle1_touched = False
        self.paddle2_touched = False
        self.done = False
        return self._get_obs()

    def step(self, action):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # Paddle Player 1
        if action == 1 and self.player1_y > 0:
            self.player1_y -= self.PADDLE_SPEED
        if action == 2 and self.player1_y < self.SCREEN_HEIGHT - self.PADDLE_HEIGHT:
            self.player1_y += self.PADDLE_SPEED

        # Movimento palla
        self.ball_x += self.ball_dx
        self.ball_y += self.ball_dy

        # Collisione con bordi superiore/inferiore
        if self.ball_y <= 0 or self.ball_y >= self.SCREEN_HEIGHT - self.BALL_SIZE:
            self.ball_dy *= -1

        reward_player1 = 0
        reward_player2 = 0

        # Rappresentazione paddle e palla con Rect di PyGame
        paddle1_rect = pygame.Rect(0, self.player1_y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        paddle2_rect = pygame.Rect(self.SCREEN_WIDTH - self.PADDLE_WIDTH, self.player2_y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        ball_rect = pygame.Rect(self.ball_x, self.ball_y, self.BALL_SIZE, self.BALL_SIZE)

        # Collisione paddle sinistro
        if ball_rect.colliderect(paddle1_rect):
            self.ball_dx *= -1
            self.ball_x = self.PADDLE_WIDTH  # Evita sovrapposizioni
            self.touches += 1
            self.paddle1_touched = True

            # Incremento velocità ogni 3 tocchi
            if self.touches % 3 == 0:
                self.ball_dx += 1 if self.ball_dx > 0 else -1
                self.ball_dy += 1 if self.ball_dy > 0 else -1

            # Adjust trajectory based on hit point
            impact_point = (self.ball_y - self.player1_y) / self.PADDLE_HEIGHT
            self.ball_dy = (impact_point - 0.5) * 2 * abs(self.ball_dx)
            reward_player1 += 1

        # Collisione paddle destro
        if ball_rect.colliderect(paddle2_rect):
            self.ball_dx *= -1
            self.ball_x = self.SCREEN_WIDTH - self.PADDLE_WIDTH - self.BALL_SIZE  # Evita sovrapposizioni
            self.touches += 1
            self.paddle2_touched = True

            # Incremento velocità ogni 3 tocchi
            if self.touches % 3 == 0:
                self.ball_dx += 1 if self.ball_dx > 0 else -1
                self.ball_dy += 1 if self.ball_dy > 0 else -1

            # Adjust trajectory based on hit point
            impact_point = (self.ball_y - self.player2_y) / self.PADDLE_HEIGHT
            self.ball_dy = (impact_point - 0.5) * 2 * abs(self.ball_dx)
            reward_player2 += 1

        # Penalità e Reward quando la palla supera i paddle
        if self.ball_x < 0:  # Punto per Player 2
            self.score_player2 += 1
            if self.paddle2_touched:  # Paddle 2 ha toccato, penalità per Player 1
                reward_player1 -= 5
                reward_player2 += 1
            else:  # Paddle 2 non ha toccato, solo penalità per Player 1
                reward_player1 -= 5
            self.done = True

        elif self.ball_x > self.SCREEN_WIDTH:  # Punto per Player 1
            self.score_player1 += 1
            if self.paddle1_touched:  # Paddle 1 ha toccato, penalità per Player 2
                reward_player2 -= 5
                reward_player1 += 1
            else:  # Paddle 1 non ha toccato, solo penalità per Player 2
                reward_player2 -= 5
            self.done = True

        return self._get_obs(), (reward_player1, reward_player2), self.done, {}

    def render(self, mode='human'):
        self.screen.fill((0, 0, 0))
        pygame.draw.rect(self.screen, (255, 255, 255),
                         (0, self.player1_y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT))
        pygame.draw.rect(self.screen, (255, 255, 255),
                         (self.SCREEN_WIDTH - self.PADDLE_WIDTH, self.player2_y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT))
        pygame.draw.ellipse(self.screen, (255, 255, 255),
                            (self.ball_x, self.ball_y, self.BALL_SIZE, self.BALL_SIZE))

        font = pygame.font.SysFont("Arial", 30)
        score_text = font.render(f"Player 1: {self.score_player1}   Player 2: {self.score_player2}", True, (255, 255, 255))
        self.screen.blit(score_text, (self.SCREEN_WIDTH // 2 - score_text.get_width() // 2, 20))

        # Mostra il numero di tocchi
        touches_text = font.render(f"Touches: {self.touches}", True, (255, 255, 255))
        self.screen.blit(touches_text, (self.SCREEN_WIDTH // 2 - touches_text.get_width() // 2, 50))

        pygame.display.flip()
        self.clock.tick(2000)

    def _get_obs(self):
        obs = pygame.surfarray.array3d(pygame.display.get_surface())
        return obs

    def close(self):
        pygame.quit()
