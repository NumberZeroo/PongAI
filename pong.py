# pong.py
import pygame
import random
import numpy as np
from gym import Env, spaces


class PongEnv(Env):
    """
    Ambiente Pong personalizzato che estende gym.Env.
    Utilizza Pygame per la simulazione grafica del gioco.
    """
    def __init__(self, render_mode=True):
        super(PongEnv, self).__init__()

        # Dimensioni della finestra
        self.SCREEN_WIDTH = 800
        self.SCREEN_HEIGHT = 600
        self.PADDLE_WIDTH = 10
        self.PADDLE_HEIGHT = 100
        self.BALL_SIZE = 25
        self.PADDLE_SPEED = 5
        self.BALL_SPEED = 7
        self.MAX_BALL_SPEED = 15  # Limite massimo della velocità della palla

        # Punteggio
        self.score_player1 = 0
        self.score_player2 = 0

        # Numero di tocchi per il progresso della velocità
        self.touches = 0

        # Flag per verificare se i paddle hanno toccato la palla
        self.paddle1_touched = False
        self.paddle2_touched = False

        # Spazio osservazione e azione
        # Definizione dello spazio di osservazione come valori continui
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, -self.BALL_SPEED, -self.BALL_SPEED], dtype=np.float32),
            high=np.array([self.SCREEN_HEIGHT, self.SCREEN_WIDTH, self.SCREEN_HEIGHT, self.BALL_SPEED, self.BALL_SPEED],
                          dtype=np.float32),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)  # 0: Stay, 1: Up, 2: Down

        # PyGame inizializzazione
        self.render_mode = render_mode
        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
            pygame.display.set_caption("PongAI")
            self.clock = pygame.time.Clock()

            # Carica il suono per la collisione
            self.collision_sound = pygame.mixer.Sound("envSound/paddleTouch.wav")

            # Carica il suono per il punto
            self.point_sound = pygame.mixer.Sound("envSound/pointScored.wav")

            #Carica la ost di sottofondo
            pygame.mixer.music.load("envSound/PongAI_ost.wav")

            #Riproduci la musica in loop
            pygame.mixer.music.play(-1)

        else:
            self.screen = None
            self.clock = None
            self.collision_sound = None

        self.reset()

    def reset(self):
        """
        Resetta lo stato dell'ambiente all'inizio di un episodio.
        """
        self.player1_y = (self.SCREEN_HEIGHT - self.PADDLE_HEIGHT) // 2
        self.player2_y = (self.SCREEN_HEIGHT - self.PADDLE_HEIGHT) // 2
        self.ball_x = self.SCREEN_WIDTH // 2
        self.ball_y = self.SCREEN_HEIGHT // 2

        # Direzione casuale per la palla
        self.ball_dx = random.choice([-1, 1]) * random.randint(2, 3)
        self.ball_dy = random.choice([-1, 1]) * random.randint(2, 3)

        self.touches = 0
        self.paddle1_touched = False
        self.paddle2_touched = False
        self.done = False
        return self._get_obs()

    def step(self, action1, action2):
        """
        Esegue un passo nell'ambiente basato sulle azioni fornite.

        Args:
            action1 (int): Azione del Player 1.
            action2 (int): Azione del Player 2.

        Returns:
            tuple: (osservazione, ricompense, done, info)
        """
        if self.render_mode:
            self._handle_events()

        # Aggiorna le posizioni dei paddle
        self._update_paddle_position(action1, action2)

        # Muovi la palla
        self._move_ball()

        # Gestisci le collisioni e aggiorna le ricompense
        rewards = self._handle_collisions()

        return self._get_obs(), rewards, self.done, {}

    def render(self, mode='human'):
        """
        Visualizza lo stato corrente del gioco utilizzando Pygame.
        """
        if not self.render_mode:
            return

        self.screen.fill((30, 30, 30))  # Colore di sfondo nero

        # Disegna i paddle
        pygame.draw.rect(self.screen, (255, 0, 0),
                         (0, self.player1_y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT))
        pygame.draw.rect(self.screen, (0, 200, 255),
                         (self.SCREEN_WIDTH - self.PADDLE_WIDTH, self.player2_y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT))

        # Disegna la palla
        pygame.draw.ellipse(self.screen, (160, 32, 240),
                            (self.ball_x, self.ball_y, self.BALL_SIZE, self.BALL_SIZE))

        # Disegna la linea centrale della metà campo
        for y in range(0, self.SCREEN_HEIGHT, 40):
            pygame.draw.line(self.screen, (100, 100, 100), (self.SCREEN_WIDTH // 2, y),
                             (self.SCREEN_WIDTH // 2, y + 20), 2)

        # Mostra il punteggio
        font = pygame.font.SysFont("Arial", 30)
        score_text = font.render(f"Player 1: {self.score_player1}   Player 2: {self.score_player2}", True,
                                 (255, 255, 255))
        self.screen.blit(score_text, (self.SCREEN_WIDTH // 2 - score_text.get_width() // 2, 20))

        # Mostra il numero di tocchi
        touches_text = font.render(f"Touches: {self.touches}", True, (255, 255, 255))
        self.screen.blit(touches_text, (self.SCREEN_WIDTH // 2 - touches_text.get_width() // 2, 50))

        pygame.display.flip()
        self.clock.tick(120)  # Limite di FPS a 60

    def _get_obs(self):
        # Ora ritorniamo anche player2_y
        return (self.player1_y, self.player2_y, self.ball_x, self.ball_y, self.ball_dx, self.ball_dy)

    def _handle_events(self):
        """
        Gestisce gli eventi di Pygame, come la chiusura della finestra.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

    def _update_paddle_position(self, action1, action2):
        """
        Aggiorna la posizione dei paddle in base alle azioni fornite.

        Args:
            action1 (int): Azione del Player 1.
            action2 (int): Azione del Player 2.
        """
        # Paddle Player 1
        if action1 == 1 and self.player1_y > 0:
            self.player1_y -= self.PADDLE_SPEED
        if action1 == 2 and self.player1_y < self.SCREEN_HEIGHT - self.PADDLE_HEIGHT:
            self.player1_y += self.PADDLE_SPEED

        # Paddle Player 2
        if action2 == 1 and self.player2_y > 0:
            self.player2_y -= self.PADDLE_SPEED
        if action2 == 2 and self.player2_y < self.SCREEN_HEIGHT - self.PADDLE_HEIGHT:
            self.player2_y += self.PADDLE_SPEED

    def _move_ball(self):
        """
        Aggiorna la posizione della palla.
        """
        self.ball_x += self.ball_dx
        self.ball_y += self.ball_dy

        # Collisione con bordi superiore/inferiore
        if self.ball_y <= 0 or self.ball_y >= self.SCREEN_HEIGHT - self.BALL_SIZE:
            self.ball_dy *= -1

    def _handle_collisions(self):
        """
        Gestisce le collisioni tra la palla e i paddle e aggiorna le ricompense.

        Returns:
            tuple: Ricompense per Player 1 e Player 2.
        """
        reward_player1 = 0
        reward_player2 = 0

        # Rappresentazione paddle e palla con Rect di Pygame
        paddle1_rect = pygame.Rect(0, self.player1_y, self.PADDLE_WIDTH, self.PADDLE_HEIGHT)
        paddle2_rect = pygame.Rect(self.SCREEN_WIDTH - self.PADDLE_WIDTH, self.player2_y, self.PADDLE_WIDTH,
                                   self.PADDLE_HEIGHT)
        ball_rect = pygame.Rect(self.ball_x, self.ball_y, self.BALL_SIZE, self.BALL_SIZE)

        # Collisione paddle sinistro
        if ball_rect.colliderect(paddle1_rect):
            self.ball_dx *= -1
            self.ball_x = self.PADDLE_WIDTH  # Evita sovrapposizioni
            self.touches += 1
            self.paddle1_touched = True
            if self.render_mode:
                self.collision_sound.play()

            # Incremento velocità ogni 3 tocchi
            if self.touches % 3 == 0:
                self.ball_dx = np.clip(self.ball_dx + (1 if self.ball_dx > 0 else -1), -self.MAX_BALL_SPEED,
                                       self.MAX_BALL_SPEED)
                self.ball_dy = np.clip(self.ball_dy + (1 if self.ball_dy > 0 else -1), -self.MAX_BALL_SPEED,
                                       self.MAX_BALL_SPEED)

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
            if self.render_mode:
                self.collision_sound.play()

            # Incremento velocità ogni 3 tocchi
            if self.touches % 3 == 0:
                self.ball_dx = np.clip(self.ball_dx + (1 if self.ball_dx > 0 else -1), -self.MAX_BALL_SPEED,
                                       self.MAX_BALL_SPEED)
                self.ball_dy = np.clip(self.ball_dy + (1 if self.ball_dy > 0 else -1), -self.MAX_BALL_SPEED,
                                       self.MAX_BALL_SPEED)

            # Adjust trajectory based on hit point
            impact_point = (self.ball_y - self.player2_y) / self.PADDLE_HEIGHT
            self.ball_dy = (impact_point - 0.5) * 2 * abs(self.ball_dx)
            reward_player2 += 1

        # Penalità e Reward quando la palla supera i paddle
        if self.ball_x < 0:  # Punto per Player 2
            if self.render_mode:
                self.point_sound.play()

            if self.paddle2_touched:  # Paddle 2 ha toccato, penalità per Player 1
                self.score_player2 += 1
                reward_player1 -= 5
                reward_player2 += 1
            else:  # Paddle 2 non ha toccato, solo penalità per Player 1
                reward_player1 -= 5
            self.done = True

        elif self.ball_x > self.SCREEN_WIDTH:  # Punto per Player 1
            if self.render_mode:
                self.point_sound.play()

            if self.paddle1_touched:  # Paddle 1 ha toccato, penalità per Player 2
                self.score_player1 += 1
                reward_player2 -= 5
                reward_player1 += 1
            else:  # Paddle 1 non ha toccato, solo penalità per Player 2
                reward_player2 -= 5
            self.done = True

        return reward_player1, reward_player2

    def close(self):
        """
        Chiude l'ambiente e Pygame correttamente.
        """
        if self.render_mode:
            pygame.quit()
