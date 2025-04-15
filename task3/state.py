

class State:
    def __init__(self):
        self.board = [[' ' for _ in range(3)] for _ in range(3)]
    
    def print_board(self):
        for row in self.board:
            print('|'.join(row))
            print('-' * 5)
        print()
    
    def is_winner(self, symbol):
        for i in range(3):
            if all(self.board[i][j] == symbol for j in range(3)) or \
               all(self.board[j][i] == symbol for j in range(3)):
                return True
        if all(self.board[i][i] == symbol for i in range(3)) or \
           all(self.board[i][2-i] == symbol for i in range(3)):
            return True
        return False
    
    def is_full(self):
        return all(self.board[i][j] != ' ' for i in range(3) for j in range(3))
    
    def is_terminal(self):
        return self.is_winner('X') or self.is_winner('O') or self.is_full()
    
    def available_moves(self):
        return [(i, j) for i in range(3) for j in range(3) if self.board[i][j] == ' ']