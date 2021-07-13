# AYTO (Are You the One?)

SUMMARY
-------
- Solving the mathematical problem underlying the reality show ["Are You the One?"](https://en.wikipedia.org/wiki/Are_You_the_One%3F), 1st season's rules.
- Mathematical problem: 20 "contestants", 10 male and 10 female, have 10 days to pair up in 10 "perfect" mixed-sex couples, as chosen secretly by the game oracle at the start. Every morning, they can test the correctness of one single couple, with the oracle replying truthfully. Every evening, they can test the correctness of 10 mixed-sex couples at the same time, with the oracle telling them truthfully how many perfect matches they got right. What algorithm is best for deducing the solution 1. with the highest winning probability, 2. with the lowest number of guesses, 3. computing fastest?
- Developed with Python 3.9.1.

ALGORITHM (AND VARIANTS)
------------------------
The solver:
1. tracks all the still admissible permutations as its search space
2. exactly prunes the search space based on every test result
3. tries to optimally choose one test at a time, in a greedy way (which is probably optimal anyway)
4. chooses one of the best ranking M-tests among all possible M-tests
    1. the M-test result probabilities (coinciding with the couple probabilities) needed for these entropy calculations are re-computed whenever needed from the still admissible permutations
5. chooses one of the best ranking N-tests among an adaptive number of randomly chosen possible N-tests; the number is increased as the search space and consequently the computational cost grow smaller
    1. the N-test result probabilities needed for these entropy calculations are re-computed whenever needed from the still admissible permutations.
6. variants: the tests can be ranked in either of two ways:
    1. by the vector of their outcome frequencies in decreasing order, so as to maximize the worst-case information discovery on each test; this variant was expected to minimize the probability of losing some games out of bad lucktest_10_couples_100_games.txt
    2. by their [entropy](https://en.wikipedia.org/wiki/Entropy_(information_theory)), so as to maximize the average information discovery on each test; this variant was expected to minimize the number of in-game days needed, to the cost risking to lose more games out of bad luck

Using just a random sample in point 5. for speed-up works pretty well because, at the start of the game, all possible N-tests have more or less the same entropy, due to the symmetrical nature of the problem; the possible N-tests start having significantly different entropies only later, right when the search space grows smaller and consequently computing their entropies becomes faster.

However, point 5. could be parallelized in order to assess all possible N-tests.

After testing each variant with 100 games each, the frequency vector-ranking variant seemed slightly superior under both criteria of average winning speed and leaving a "buffer" of in-game days, contrary to expectations, but this could be an artefact of testing with only 100 games and/or of not optimizing over all possible tests.

PERFORMANCE STATS
-----------------
Both implemented variants of the solver performed more or less equally:
- On my MacBook Air with processor 2.2 GHz Dual-Core Intel Core i7, either solver takes about 1 minute on average to solve a game.
- Over 100 test games, the frequency vector-ranking solver worked slightly better than the entropy-ranking solver:
    - the frequency vector-ranking solver won 99 games taking 7.93 days on average
    - the entropy-ranking solver won 98 games taking 8.11 days on average

For more details, please see the test outputs.

ACKNOWLEDGEMENTS
----------------
Many thanks to Daniel Ronde for acquainting me with this problem.
