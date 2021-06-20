# AYTO (Are You the One?)

SUMMARY
-------
- Solving the mathematical problem underlying the reality show ["Are You the One?"](https://en.wikipedia.org/wiki/Are_You_the_One%3F), 1st season's rules. 
- Developed with Python 3.9.1.

PERFORMANCE STATS
-----------------
- The solver won 50 out of 50 test games, or 100.00% of them. 
- On average, a victory took 7.88 days.
- On my MacBook Air with processor 2.2 GHz Dual-Core Intel Core i7, it takes about 1 minute on average to solve a game.

ALGORITHM
---------
The solver:
1. uses the (still) admissible permutations / couplings as the search space
2. exactly prunes the search space based on every test result
3. tries to optimally choose one test at a time, in a greedy way (which may or may not be optimal)
4. chooses one of the M-test with maximum information entropy among all possible M-tests
5. chooses one of the N-test with maximum information entropy among an adaptive number of randomly chosen possible N-tests; the number is increased as the search space and consequently the computational cost grow smaller.

The speed-up trick used in 5. works pretty well because, at the start of the game, all possible N-tests have more or less the same entropy, due to the symmetrical nature of the problem; the possible N-tests start having significantly different entropies only later, as the search space grows smaller and consequently computing their entropies becomes faster.

However, 5. could be easily parallelized in order to assess all possible N-tests.

ACKNOWLEDGEMENTS
----------------
Many thanks to Daniel Ronde for identifying and formulating the problem.
