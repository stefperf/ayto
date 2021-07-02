from game_oracle import GameOracle
from game_solver import GameSolver


n_couples = 10  # number of perfect matches alpha-beta to be identified = number of days available
n_games = 100  # number of games to be played in order to collect stats


def play_game(show_process):
    """
    play the information theory game underlying "Are You the One?", 1st season
    :param show_process: if True, illustrate the game and the solver's reasoning, too
    :return: (was the game won?: boolean, nr. of days elapsed)
    """
    oracle = GameOracle(n_couples)
    solver = GameSolver(n_couples, show_optimization=show_process)
    if show_process:
        print()
        print('=== START OF THE GAME ===')
        print('The oracle has randomly chosen this solution:')
        print(list(oracle.betas))
        print(f'The solver has {n_couples} days to guess it.')
        solver.print_admissible_permutations()
        solver.print_couple_stats()
    
    game_won = False
    for day in range(n_couples):
    
        if show_process:
            print()
            print(f'=== On the morning of day {day + 1}:')
        (alpha, beta) = solver.choose_m_test()
        if show_process:
            print(f'The solver chooses the M-test ({alpha}, {beta}).')
        m_test_result = oracle.m_test(alpha, beta)
        if show_process:
            print(f'The oracle answers {m_test_result}.')
        solver.assess_m_test(alpha, beta, m_test_result)
    
        if show_process:
            print()
            print(f'--- On the night of day {day + 1}:')
        betas = solver.choose_n_test()
        if show_process:
            print(f'The solver chooses the N-test {list(betas)}.')
        n_test_result = oracle.n_test(betas)
        if show_process:
            print(f'The oracle answers {n_test_result}.')
    
        if n_test_result == n_couples:
            game_won = True
            break
    
        solver.assess_n_test(betas, n_test_result)

    if show_process:
        if game_won:
            print('The solver has won the game!')
        else:
            if show_process: print('The solver has lost the game.')
        print('\n=== END OF THE GAME ===\n')
        print()

    return game_won, day + 1


print()
print('------------------------------------------------------------------------------')
print(f"Collecting stats over {n_games} games, of which the first is fully shown:")
print('------------------------------------------------------------------------------')
n_games_won = 0
n_days_distrib = {d: 0 for d in range(1, n_couples + 1)}
for game_nr in range(1, n_games + 1):
    game_won, n_days = play_game(show_process=True if game_nr == 1 else False)
    if game_won:
        print(f'The solver won game # {game_nr} in {n_days} days.')
        n_games_won += 1
        n_days_distrib[n_days] += 1
    else:
        print(f'The solver lost game # {game_nr}.')
    if not (game_nr % 10):
        print()
        print(f'The solver won {n_games_won} out of {game_nr} test games, or '
              f'{n_games_won / game_nr * 100:.2f}% of them. '
              f'On average, a victory took '
              f'{sum([n_days * freq for n_days, freq in n_days_distrib.items()]) / n_games_won:.2f} days.')
        print()
