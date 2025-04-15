from game import Game
import time

def main():
    print("Witaj w grze Kółko i Krzyżyk!")
    print("Wybierz tryb gry:")
    print("1. Komputer vs Komputer")
    print("2. Człowiek vs Komputer")
    
    while True:
        choice = input("Podaj numer trybu (1 lub 2): ")
        if choice in ['1', '2']:
            break
        print("Nieprawidłowy wybór! Wybierz 1 lub 2.")
    
    if choice == '2':
        while True:
            symbol = input("Wybierz symbol (X lub O): ").upper()
            if symbol in ['X', 'O']:
                break
            print("Nieprawidłowy symbol! Wybierz X lub O.")
        
        depth = int(input("Podaj głębokość przeszukiwania dla komputera (1-9): "))
        print(f"\nGra człowiek ({symbol}) vs komputer z głębokością: {depth}")
        game = Game(max_depth=depth)
        game.play(mode="human_vs_computer", human_symbol=symbol)
    
    else:
        for depth in [1, 3, 5, 9]:
            print(f"\nGra komputer vs komputer z głębokością przeszukiwania: {depth}")
            game = Game(max_depth=depth)
            game.play(mode="computer_vs_computer")
            time.sleep(1) 

if __name__ == "__main__":
    main()