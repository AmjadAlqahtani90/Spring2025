from random import randint
from random import uniform
import math

def initial_state():
  return [randint(0, 7) for _ in range(8)]

def random_successor(state):
  new_state = state[:]
  col = randint(0, 8 - 1)
  new_row = randint(0, 8 - 1)
  new_state[col] = new_row
  return new_state

def conflicts(state):
  conflicts = 0
  for i in range(8):
      for j in range(i + 1, 8):
          if state[i] == state[j] or abs(state[i] - state[j]) == abs(i - j):
              conflicts += 1
  return -conflicts

def exponential_schedule(t, t0=100, cooling_factor=0.99, min_temp=0.001):
  return max(min_temp, t0 * (cooling_factor ** t))

def simulated_annealing(schedule, max_iterations=10000):
  current_state = initial_state()
  t = 1 # iteration counter

  for _ in range(max_iterations):
    T = schedule(t)
    if T <= 0:
        break

    next_state = random_successor(current_state)
    delta_E = conflicts(next_state) - conflicts(current_state)

    if delta_E > 0 or uniform(0, 1) < math.exp(delta_E / T):
        current_state = next_state

    if conflicts(current_state) == 0:
        return current_state

    t += 1

  return "No result found"

def visualize_board(state):
  size = len(state)
  board = [["." for _ in range(size)] for _ in range(size)]

  for col, row in enumerate(state):
      board[row][col] = "Q"

  print("\nSolution Board:")
  for row in board:
      print(" ".join(row))
  print()

if __name__ == "__main__":
    solution = simulated_annealing(lambda t: exponential_schedule(t, cooling_factor=0.99))
    print("\nFinal Solution (Queen positions per column):\n", solution)
    if isinstance(solution, list):
        visualize_board(solution)
