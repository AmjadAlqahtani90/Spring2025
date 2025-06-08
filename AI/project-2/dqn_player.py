import numpy as np
import torch
import torch.nn as nn
import socket, pickle, os
from reversi import reversi

# ====== Config ======
BOARD_SIZE = 8
MODEL_PATH = "models/reversi_model.pth"

# ====== DQN Model ======
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(BOARD_SIZE * BOARD_SIZE, 128),
            nn.ReLU(),
            nn.Linear(128, BOARD_SIZE * BOARD_SIZE)  # Q-values for all 64 positions
        )

    def forward(self, x):
        return self.model(x)  # Output shape: (batch_size, 64)

# ====== Move Selection ======
def get_valid_moves(game, player):
    return [(x, y) for x in range(BOARD_SIZE) for y in range(BOARD_SIZE) if game.step(x, y, player, False) > 0]

def select_action(model, board, valid_moves, device):
    board_tensor = torch.tensor(board, dtype=torch.float32, device=device).unsqueeze(0)  # shape: [1, 8, 8]
    q_values = model(board_tensor).squeeze(0)  # shape: [64]

    move_scores = {}
    for x, y in valid_moves:
        idx = x * BOARD_SIZE + y
        move_scores[(x, y)] = q_values[idx].item()

    return max(move_scores, key=move_scores.get) if move_scores else (-1, -1)

# ====== Board Display (Optional) ======
def print_board_readable(board):
    symbol = {1: 'W', -1: 'B', 0: '.'}
    print("\n  " + " ".join(str(i) for i in range(BOARD_SIZE)))
    for i, row in enumerate(board):
        print(f"{i} " + " ".join(symbol[x] for x in row))

# ====== Main Game Loop ======
def main():
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model file not found at: {MODEL_PATH}")
        return

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DQN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # Connect to game server
    game_socket = socket.socket()
    game_socket.connect(('127.0.0.1', 33333))
    game = reversi()

    print("[INFO] Connected to Reversi server.")

    while True:
        data = game_socket.recv(4096)
        turn, board = pickle.loads(data)

        if turn == 0:
            print("[INFO] Game over.")
            game_socket.close()
            return

        game.board = board
        print(f"\nTurn: {'White (1)' if turn == 1 else 'Black (-1)'}")
        print_board_readable(board)

        valid_moves = get_valid_moves(game, turn)
        move = select_action(model, board, valid_moves, device) if valid_moves else (-1, -1)

        print(f"[AI MOVE] Selected: {move}")
        game_socket.send(pickle.dumps([move[0], move[1]]))

if __name__ == "__main__":
    main()
