import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import os
from reversi import reversi

BOARD_SIZE = 8
EPISODES = 2000
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.995
LR = 0.001
MODEL_PATH = "models/reversi_model.pth"

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(BOARD_SIZE * BOARD_SIZE, 128),
            nn.ReLU(),
            nn.Linear(128, BOARD_SIZE * BOARD_SIZE)
        )
    def forward(self, x):
        return self.model(x)

def get_valid_moves(game, player):
    return [(x, y) for x in range(8) for y in range(8) if game.step(x, y, player, False) > 0]

def apply_move(board, player, move):
    game = reversi()
    game.board = np.copy(board)
    game.step(move[0], move[1], player, True)
    return game.board

def choose_action(model, board, valid_moves, epsilon, device):
    if random.random() < epsilon:
        return random.choice(valid_moves)
    board_tensor = torch.tensor(board, dtype=torch.float32, device=device).unsqueeze(0)
    q_values = model(board_tensor).squeeze(0)
    best_move = max(valid_moves, key=lambda m: q_values[m[0] * 8 + m[1]].item())
    return best_move

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DQN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    epsilon = EPSILON_START
    os.makedirs("models", exist_ok=True)

    for episode in range(EPISODES):
        game = reversi()
        player = 1
        done = False
        while not done:
            board = game.board
            valid_moves = get_valid_moves(game, player)
            if not valid_moves:
                player *= -1
                if not get_valid_moves(game, player):
                    done = True
                    break
                continue
            move = choose_action(model, board, valid_moves, epsilon, device)
            new_board = apply_move(board, player, move)

            reward = np.sum(new_board == player) - np.sum(board == player)
            target = reward

            board_tensor = torch.tensor(board, dtype=torch.float32, device=device).unsqueeze(0)
            q_values = model(board_tensor)
            target_q_values = q_values.clone().detach()

            idx = move[0] * 8 + move[1]
            target_q_values[0][idx] = target

            loss = loss_fn(q_values, target_q_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            game.board = new_board
            player *= -1

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        if episode % 100 == 0:
            print(f"Episode {episode}, Epsilon: {epsilon:.3f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print("Model trained and saved.")

if __name__ == "__main__":
    main()
