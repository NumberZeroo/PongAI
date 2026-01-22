# Pong AI - Pong Game with Q-Learning

This repository contains a **Pong** game where two agents compete against each other using **Q-Learning**, a Reinforcement Learning algorithm.  
The agents learn autonomously, improving their gameplay strategies over time by aiming to hit the ball and score points.

## Features
- 🕹️ **Pong game** with real-time visualization using `pygame`.
- 🤖 **Q-Learning-based AI**: Two agents (Player 1 and Player 2) train themselves to play the game.
- 📊 **Real-time score display**: The current score is displayed at the top of the screen during gameplay.
- 🎯 **Custom reward system**: Rewards are provided for different agent behaviors, such as collisions, perfect bounces, and errors.

## Installation

1. Clone the repository:
2. 
   ```bash
   git clone https://github.com/NumberZeroo/PongAI.git
   cd PongAI
   ```
3. Create a virtual environment (optional but recommended):
4. 
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

## Usage

To start the training and run the game:
```bash
python main.py
```

## Customization
Modify parameters in `main.py` to tweak training settings, game speed, and AI behavior.

## License
This project is licensed under the MIT [LICENCE](License).

