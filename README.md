# AYTO (Are You the One?)

SUMMARY
-------
- Solving the mathematical problem underlying the reality show ["Are You the One?"](https://en.wikipedia.org/wiki/Are_You_the_One%3F), 1st season's rules.
- Mathematical problem: 20 "contestants", 10 male and 10 female, have 10 days to pair up in 10 "perfect" mixed-sex couples, as chosen secretly by the game oracle at the start. Every morning, they can test the correctness of one single couple, with the oracle replying truthfully. Every evening, they can test the correctness of 10 mixed-sex couples at the same time, with the oracle telling them truthfully how many perfect matches they got right. What algorithm is best for deducing the solution 1. with the highest winning probability, 2. with the lowest number of guesses, 3. computing fastest?
- Developed with Python 3.9.1.

PERFORMANCE STATS
-----------------
- The solver won 100 out of 100 test games, i.e. 100.00% of them. 
- On average, a victory took 7.93 in-game days.
- On my MacBook Air with processor 2.2 GHz Dual-Core Intel Core i7, it takes about 1 minute on average to solve a game.

ALGORITHM
---------
The solver:
1. tracks all the still admissible permutations as its search space
2. exactly prunes the search space based on every test result
3. tries to optimally choose one test at a time, in a greedy way (which is probably optimal anyway)
4. chooses one of the M-tests with maximum [entropy](https://en.wikipedia.org/wiki/Entropy_(information_theory)) among all possible M-tests
    1. the M-test result probabilities (coinciding with the couple probabilities) needed for these entropy calculations are re-computed whenever needed from the still admissible permutations
5. chooses one of the N-tests with maximum entropy among an adaptive number of randomly chosen possible N-tests; the number is increased as the search space and consequently the computational cost grow smaller
    1. the N-test result probabilities needed for these entropy calculations are re-computed whenever needed from the still admissible permutations.

Using just a random sample in point 5. for speed-up works pretty well because, at the start of the game, all possible N-tests have more or less the same entropy, due to the symmetrical nature of the problem; the possible N-tests start having significantly different entropies only later, right when the search space grows smaller and consequently computing their entropies becomes faster.

However, point 5. could be parallelized in order to assess all possible N-tests.

ACKNOWLEDGEMENTS
----------------
Many thanks to Daniel Ronde for acquainting me with this problem.
