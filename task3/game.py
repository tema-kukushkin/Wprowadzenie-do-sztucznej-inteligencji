import copy
from player import Player
from state import State

class Game:
    def __init__(self, max_depth=9):
        self.state = State()
        self.players = [Player('X'), Player('O')]
        self.max_depth = max_depth
    
    def evaluate(self, state):
        if state.is_winner('X'):
            return 1
        elif state.is_winner('O'):
            return -1
        return 0
    
    def minimax(self, state, depth, is_maximizing, alpha=-float('inf'), beta=float('inf')):
        if depth == 0 or state.is_terminal():
            return self.evaluate(state)
        
        if is_maximizing:
            max_eval = -float('inf')
            for move in state.available_moves():
                new_state = copy.deepcopy(state)
                new_state.board[move[0]][move[1]] = 'X'
                eval = self.minimax(new_state, depth - 1, False, alpha, beta)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in state.available_moves():
                new_state = copy.deepcopy(state)
                new_state.board[move[0]][move[1]] = 'O'
                eval = self.minimax(new_state, depth - 1, True, alpha, beta)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval
    
    def find_best_move(self, state, player):
        best_move = None
        best_value = -float('inf') if player.symbol == 'X' else float('inf')
        
        for move in state.available_moves():
            new_state = copy.deepcopy(state)
            new_state.board[move[0]][move[1]] = player.symbol
            value = self.minimax(new_state, self.max_depth - 1, 
                               player.symbol == 'O')
            if player.symbol == 'X':
                if value > best_value:
                    best_value = value
                    best_move = move
            else:
                if value < best_value:
                    best_value = value
                    best_move = move
        return best_move
    
    def get_human_move(self):
        while True:
            try:
                row = int(input("Podaj wiersz (0-2): "))
                col = int(input("Podaj kolumnę (0-2): "))
                if (row, col) in self.state.available_moves():
                    return (row, col)
                else:
                    print("Nieprawidłowy ruch! Spróbuj ponownie.")
            except ValueError:
                print("Podaj liczby od 0 do 2!")
    
    def play(self, mode="computer_vs_computer", human_symbol='X'):
        current_player = 0
        while not self.state.is_terminal():
            self.state.print_board()
            
            if mode == "human_vs_computer" and self.players[current_player].symbol == human_symbol:
                print(f"Twój ruch ({human_symbol})!")
                move = self.get_human_move()
            else:
                print(f"Ruch komputera ({self.players[current_player].symbol})...")
                move = self.find_best_move(self.state, self.players[current_player])
            
            self.state.board[move[0]][move[1]] = self.players[current_player].symbol
            current_player = 1 - current_player
        
        self.state.print_board()
        if self.state.is_winner('X'):
            print("Wygrał X!")
        elif self.state.is_winner('O'):
            print("Wygrał O!")
        else:
            print("Remis!")