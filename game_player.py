from minesweep_env import MinesweeperEnv

def play_game():
    env = MinesweeperEnv()
    done = False

    print("Welcome to Minesweeper!")
    
    # Gameplay loop
    while not done:
        state = env.get_state()
        for row in state:
            print(' '.join(str(cell) if cell != -2 else '?' for cell in row))

        valid_input = False
        while not valid_input:
            try:
                row = int(input("Enter row (0 to {}): ".format(env.size - 1)))
                column = int(input("Enter column (0 to {}):".format(env.size -1)))    
            
                if 0 <= row < env.size and 0 <= column < env.size:
                    if state[row, column] != -2: #Check if cell hasnt been revealed
                        print("This cell has already been revealed. Chose another one")
                    else:
                        valid_input = True
                else:
                    print(f"Invalid input! Please enter row and column between 0 and {env.size - 1}.")
            except ValueError:
                print("Invalid input! Please enter integers for row and column")
            
        action = (row, column)
        next_state, reward, done, _ = env.step(action)

        # Output the result of the move
        print("Reward: ", reward)
        print("Game over: ", done)

        # Check if the game is over, and display the final state
        if done:
            if reward == - 10:
                print("You hit the mine! Game over")
            elif reward > 0:
                print("You won the game")
            print("Final state:")
            final_state = env.get_state()
            for row in final_state:
                print(' '.join(str(cell) for cell in row))  # Show final state

# Run the game
if __name__ == "__main__":
    play_game()