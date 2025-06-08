import numpy as np
import socket, pickle, time
from reversi import reversi

class ReversiAI:
    def __init__(self, max_depth=3, time_limit=4.8):
        self.max_depth = max_depth
        self.time_limit = time_limit
        self.start_time = None

    def get_valid_moves(self, game, player):
        """Returns a list of valid moves for the given player."""
        return [(x, y) for x in range(8) for y in range(8) if game.step(x, y, player, False) > 0]

    def make_move(self, game, x, y, player):
        """Applies a move on a new board and returns the updated game state."""
        new_game = reversi()
        new_game.board = np.copy(game.board)
        new_game.step(x, y, player, True)
        return new_game

    def evaluate(self, game, player, player_moves, opponent_moves):
        """Improved heuristic evaluation function with weighted features."""
        piece_weight = 1
        corner_weight = 100
        edge_weight = 10
        x_square_penalty = -50
        mobility_weight = 5

        score = 0
        corners = {(0, 0), (0, 7), (7, 0), (7, 7)}
        edges = {(0, i) for i in range(8)} | {(7, i) for i in range(8)} | \
                {(i, 0) for i in range(8)} | {(i, 7) for i in range(8)}
        x_squares = {(1, 1), (1, 6), (6, 1), (6, 6)}

        for x in range(8):
            for y in range(8):
                if game.board[x, y] == player:
                    score += piece_weight
                    if (x, y) in corners:
                        score += corner_weight
                    elif (x, y) in edges:
                        score += edge_weight
                    elif (x, y) in x_squares:
                        score += x_square_penalty
                elif game.board[x, y] == -player:
                    score -= piece_weight

        score += mobility_weight * (player_moves - opponent_moves)
        return score

    def minimax_alpha_beta(self, game, depth, alpha, beta, is_maximizing, player):
        """Minimax algorithm with Alpha-Beta Pruning."""
        if depth == 0 or time.time() - self.start_time > self.time_limit:
            player_moves = len(self.get_valid_moves(game, player))
            opponent_moves = len(self.get_valid_moves(game, -player))
            return self.evaluate(game, player, player_moves, opponent_moves)

        moves = self.get_valid_moves(game, player if is_maximizing else -player)

        if not moves:
            player_moves = len(self.get_valid_moves(game, player))
            opponent_moves = len(self.get_valid_moves(game, -player))
            return self.evaluate(game, player, player_moves, opponent_moves)

        if is_maximizing:
            max_eval = -float('inf')
            for move in moves:
                new_game = self.make_move(game, move[0], move[1], player)
                eval = self.minimax_alpha_beta(new_game, depth - 1, alpha, beta, False, player)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in moves:
                new_game = self.make_move(game, move[0], move[1], -player)
                eval = self.minimax_alpha_beta(new_game, depth - 1, alpha, beta, True, player)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval

    def best_move(self, game, player):
        """Finds the best move using Minimax with Alpha-Beta Pruning."""
        self.start_time = time.time()
        moves = self.get_valid_moves(game, player)
        if not moves:
            return None

        best_score = -float('inf')
        best_move = None

        for move in moves:
            new_game = self.make_move(game, move[0], move[1], player)
            score = self.minimax_alpha_beta(new_game, self.max_depth, -float('inf'), float('inf'), False, player)

            if score > best_score:
                best_score = score
                best_move = move

        return best_move


# ============================
# Main function to interact with the Reversi game server
# ============================

def main():
    game_socket = socket.socket()
    game_socket.connect(('127.0.0.1', 33333))

    game = reversi()
    ai = ReversiAI()

    while True:
        data = game_socket.recv(4096)
        turn, board = pickle.loads(data)

        if turn == 0:
            game_socket.close()
            return

        # Debugging information: Print the current turn and board state
        print(turn)
        print(board)

        game.board = board  # Update game board
        
        # Get the best move using Minimax AI
        best_move = ai.best_move(game, turn)

        
        if best_move is None:
            best_move = (-1, -1)

        x, y = best_move  # Unpack the best move coordinates

        # Debugging: Print AI's chosen move
        print(f"AI chooses move: {x}, {y}")

        # Send AI's move to the server
        game_socket.send(pickle.dumps([x, y]))

if __name__ == '__main__':
    main()  # Start the program
