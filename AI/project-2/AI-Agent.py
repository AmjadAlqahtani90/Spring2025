import numpy as np
import socket, pickle, time
from reversi import reversi

class ReversiAI:
    def __init__(self, max_depth=3, time_limit=4.8):
        #Initialize the AI with a maximum depth for Minimax and a time limit per move.
        self.max_depth = max_depth
        self.time_limit = time_limit
        self.start_time = None

    def get_valid_moves(self, game, player):
        # Returns a list of valid moves for the given player.
        moves = []
        for x in range(8):
            for y in range(8):
                if game.step(x, y, player, False) > 0:
                    moves.append((x, y))
        return moves

    def make_move(self, game, x, y, player):
        # Applies a move on a new board and returns the updated game state.
        new_game = reversi()
        new_game.board = np.copy(game.board)
        new_game.step(x, y, player, True)
        return new_game

    def evaluate(self, game, player):
        # Improved heuristic evaluation function with weighted features.
        # Weights for different board positions
        piece_weight = 1
        corner_weight = 100
        edge_weight = 10 
        x_square_penalty = -50 
        mobility_weight = 5

        score = 0 
        corners = [(0, 0), (0, 7), (7, 0), (7, 7)]
        edges = [(0, i) for i in range(8)] + [(7, i) for i in range(8)] + \
                [(i, 0) for i in range(8)] + [(i, 7) for i in range(8)]
        x_squares = [(1, 1), (1, 6), (6, 1), (6, 6)]

        # Count mobility (number of valid moves)
        player_moves = len(self.get_valid_moves(game, player))
        opponent_moves = len(self.get_valid_moves(game, -player))

        # Evaluate board state
        for x in range(8):
            for y in range(8):
                if game.board[x, y] == player:
                    score += piece_weight

                    # Positional advantages
                    if (x, y) in corners:
                        score += corner_weight  # Strongest position
                    elif (x, y) in edges:
                        score += edge_weight  # Good position
                    elif (x, y) in x_squares:
                        score += x_square_penalty  # Dangerous position
                elif game.board[x, y] == -player:
                    score -= piece_weight  # Penalize opponent pieces

        # Add mobility factor
        score += mobility_weight * (player_moves - opponent_moves)

        return score

        # Bonus for reducing opponent's mobility (fewer available moves for the opponent)
        opponent_moves = len(self.get_valid_moves(game, -player))
        player_moves = len(self.get_valid_moves(game, player))
        score += (10 * (player_moves - opponent_moves))  # Higher difference = better position

        return score

    def minimax_alpha_beta(self, game, depth, alpha, beta, is_maximizing, player):
        # Minimax algorithm with Alpha-Beta Pruning to find the best move
        
        if depth == 0 or time.time() - self.start_time > self.time_limit:
            return self.evaluate(game, player)

        # Get all valid moves for the current player
        moves = self.get_valid_moves(game, player if is_maximizing else -player)

        # If no moves available, return evaluation score
        if not moves:
            return self.evaluate(game, player)

        
        if is_maximizing:
            max_eval = -float('inf')  # Worst case scenario for maximizing
            for move in moves:
                new_game = self.make_move(game, move[0], move[1], player)  # Apply move
                eval = self.minimax_alpha_beta(new_game, depth - 1, alpha, beta, False, player)  # Recur for opponent
                max_eval = max(max_eval, eval)  # Store the best maximum score
                alpha = max(alpha, eval)  # Update alpha
                if beta <= alpha:  
                    break
            return max_eval  

        
        else:
            min_eval = float('inf')  # Worst case scenario for minimizing
            for move in moves:
                new_game = self.make_move(game, move[0], move[1], -player)  # Apply opponent's move
                eval = self.minimax_alpha_beta(new_game, depth - 1, alpha, beta, True, player)  # Recur for AI
                min_eval = min(min_eval, eval)  # Store the best minimum score
                beta = min(beta, eval)  # Update beta
                if beta <= alpha:  # Alpha-beta pruning condition
                    break 
            return min_eval  

    def best_move(self, game, player):
        # Finds the best possible move using Minimax with Alpha-Beta Pruning.
        self.start_time = time.time()
        moves = self.get_valid_moves(game, player)  # all possible moves
        #print(moves)
        if not moves:
            return None

        best_score = -float('inf')
        best_move = None

        for move in moves:
            new_game = self.make_move(game, move[0], move[1], player)  # Apply move
            score = self.minimax_alpha_beta(new_game, self.max_depth, -float('inf'), float('inf'), False, player)  
            
            if score > best_score:
                best_score = score  # Update best score
                best_move = move  # Update best move

        return best_move

## AI game start 
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

        game.board = board
        
        # Get the best move using Minimax AI
        best_move = ai.best_move(game, turn)

        
        if best_move is None:
            best_move = (-1, -1)

        x, y = best_move

        # Debugging: Print AI's chosen move
        print(f"AI chooses move: {x}, {y}")

        # Send AI's move to the server
        game_socket.send(pickle.dumps([x, y]))

if __name__ == '__main__':
    main()
