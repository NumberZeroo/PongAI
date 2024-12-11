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
        self.BALL_SPEED = 5

        # Punteggio
        self.score_player1 = 0
        self.score_player2 = 0

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

        self.ball_dx = self.BALL_SPEED * random.choice([1, -1])
        self.ball_dy = self.BALL_SPEED * random.choice([1, -1])

        # Flag per determinare se i paddle hanno toccato la palla
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

        # Collisione paddle sinistro
        if (self.ball_x <= self.PADDLE_WIDTH and
            self.player1_y <= self.ball_y + self.BALL_SIZE and
            self.player1_y + self.PADDLE_HEIGHT >= self.ball_y):
            self.ball_x = self.PADDLE_WIDTH
            self.ball_dx *= -1
            self.paddle1_touched = True  # Flag: paddle 1 ha toccato la palla

            # Adjust trajectory based on hit point
            impact_point = (self.ball_y - self.player1_y) / self.PADDLE_HEIGHT
            self.ball_dy = (impact_point - 0.5) * 2 * self.BALL_SPEED
            reward_player1 += 2 if 0.4 <= impact_point <= 0.6 else 1

        # Collisione paddle destro
        if (self.ball_x >= self.SCREEN_WIDTH - self.PADDLE_WIDTH - self.BALL_SIZE and
            self.player2_y <= self.ball_y + self.BALL_SIZE and
            self.player2_y + self.PADDLE_HEIGHT >= self.ball_y):
            self.ball_x = self.SCREEN_WIDTH - self.PADDLE_WIDTH - self.BALL_SIZE
            self.ball_dx *= -1
            self.paddle2_touched = True  # Flag: paddle 2 ha toccato la palla

            # Adjust trajectory based on hit point
            impact_point = (self.ball_y - self.player2_y) / self.PADDLE_HEIGHT
            self.ball_dy = (impact_point - 0.5) * 2 * self.BALL_SPEED
            reward_player2 += 2 if 0.4 <= impact_point <= 0.6 else 1

        # Penalità e Reward quando la palla supera i paddle
        if self.ball_x < 0:  # Punto per Player 2
            self.score_player2 += 1
            if self.paddle2_touched:  # Paddle 2 ha toccato, penalità per Player 1 e reward a Player 2
                reward_player1 -= 5  # Penalità a Player 1
                reward_player2 += 1  # Reward a Player 2
            else:  # Paddle 2 non ha toccato, solo penalità a Player 1
                reward_player1 -= 5
            self.done = True

        elif self.ball_x > self.SCREEN_WIDTH:  # Punto per Player 1
            self.score_player1 += 1
            if self.paddle1_touched:  # Paddle 1 ha toccato, penalità per Player 2 e reward a Player 1
                reward_player1 += 1  # Reward a Player 1
                reward_player2 -= 5  # Penalità a Player 2
            else:  # Paddle 1 non ha toccato, solo penalità a Player 2
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

        pygame.display.flip()
        self.clock.tick(2000)

    def _get_obs(self):
        obs = pygame.surfarray.array3d(pygame.display.get_surface())
        return obs

    def close(self):
        pygame.quit()
