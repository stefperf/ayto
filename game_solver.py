import itertools
import math
import numpy as np
import random


class HeuristicTestScore(object):
    """
    HeuristicTestScore is an abstract base class for the way to rank possible N-tests
    """
    @classmethod
    def rank(cls, freqs, n_total=None):
        """
        Heuristic for scoring possible M- or N- tests, such that the optimal test has the lowest score.
        :param freqs: np.array containing either the frequencies or the probabilities of all test results
        :param n_total: sum of freqs (to be provided, optionally, to save calculations)
        :return: heuristic score, as float
        """
        pass


class ExpectedEntropyReduction(HeuristicTestScore):
    """
    The most straightforward heuristic, coming from information theory.
    This choice is actually the optimal criterion for comparing different possible tests when choosing one at a time
    with the goal of minimizing the time needed to guess the solution.
    """
    @classmethod
    def rank(cls, freqs, n_total=None):
        """
        Heuristic for scoring possible M- or N- tests, such that the optimal test has the lowest score.
        :param freqs: np.array containing either the frequencies or the probabilities of all test results
        :param n_total: sum of freqs (to be provided, optionally, to save calculations)
        :return: heuristic score, as float
        """
        if n_total is None:
            n_total = sum(freqs)
        freqs = freqs / n_total
        return sum([freq * math.log2(freq) for freq in freqs if freq > 0.0])


class EffortAllocator(object):
    """
    EffortAllocator is an abstract base class for deciding how many permutations should be explored for each N-test
    """
    def decide_permutations_nr_to_consider(self):
        pass


class ConstantEstimatedTimeEffortAllocator(EffortAllocator):
    """
    EffortAllocator that always allocates the same estimated computation time for each N-test
    """
    def __init__(self, optimization_effort=1e7, min_optimize=1):
        """
        init
        :param optimization_effort: indicative number of comparisons between permutations for calculating N-test ranks
        :param min_optimize: min. number of admissible permutations to be considered as possible N-tests
        """
        self.optimization_effort = optimization_effort
        self.min_optimize = min_optimize

    def decide_permutations_nr_to_consider(self, game_solver):
        max_optimize = len(game_solver.admissible_permutations) if game_solver._day > 0 else 1
        return min(
            max(
                self.min_optimize,
                int(self.optimization_effort // (game_solver.n * len(game_solver.admissible_permutations)))
            ),
            max_optimize
        )


class AlwaysDirectlyChoosingOnePermutation(EffortAllocator):
    """
    EffortAllocator that always directly chooses one permutation, effectively randomly
    """
    def decide_permutations_nr_to_consider(self, game_solver):
        return 1


class GameSolver:
    """
    Try to solve the information theory game underlying "Are You the One?", 1st season's rules.
    See project's README.md for more info on the algorithm used.
    """
    def __init__(self,
                 n,
                 heuristicTestRank=None, effortAllocator=None,
                 show_optimization=False, max_permutations_show=10
                 ):
        """
        init
        :param n: nr. of couples
        :param heuristicTestRank: a Heuristic subclass instance, defaulting to ExpectedEntropyReduction
        :param effortAllocator: an EffortAllocator subclass instance, defaulting to ConstantEstimatedTimeEffortAllocator
        :param show_optimization: True if the internal "reasoning" must be shown, else False
        :param max_permutations_show: max. number of permutations being shown when showing the search space
        """
        self.n = n
        self.heuristic = heuristicTestRank or ExpectedEntropyReduction()
        self.effortAllocator = effortAllocator or ConstantEstimatedTimeEffortAllocator()
        self.alphas = list(range(n))
        self.betas = self.alphas
        self.admissible_permutations = set(itertools.permutations(self.betas))
        self.max_admissible_permutations = int(math.factorial(self.n))
        self.max_permutations_show = max_permutations_show
        self.show_optimization = show_optimization
        self._couple_probabilities = np.ones((n, n)) / n
        self._day = 0
        self._time = 0  # 0 = morning, 1 = night
        self._moment = 0  # 0 = before testing, 1 = after testing

    def time(self):
        """
        return "morning X" or "night X" depending on the day nr. and time of day
        """
        return f'{"morning" if self._time == 0 else "night"} {self._day + 1}'

    def compute_couple_probabilities(self):
        """
        return all couple probabilities computed from the still admissible permutations
        """
        couple_probabilities = np.zeros((self.n, self.n), dtype=float)
        incr = 1. / len(self.admissible_permutations)
        for ap in self.admissible_permutations:
            for alpha, beta in enumerate(ap):
                couple_probabilities[alpha, beta] += incr
        return couple_probabilities

    def choose_m_test(self):
        """
        return the optimally chosen couple (alpha, beta) to be tested in the morning (M-test), with its rank
        """
        assert self._day < self.n and self._time == 0 and self._moment == 0
        self._moment = 1
        min_rank, max_rank = float('inf'), float('-inf')
        best_tests = []
        for r in range(self.n):
            for c in range(self.n):
                prob = self._couple_probabilities[r, c]
                rank = self.heuristic.rank(np.array([prob, 1. - prob]), 1.)
                if rank < min_rank:
                    best_tests = [(r, c)]
                    min_rank = rank
                elif rank == min_rank:
                    best_tests.append((r, c))
                if rank > max_rank:
                    max_rank = rank
        if self.show_optimization:
            rank_range = max_rank - min_rank
            print(f'The solver analyzes all the {self.n ** 2} possible couples, '
                  f'the chosen best one(s) have rank {min_rank:.3f}, '
                  f'{rank_range:.3f} better than the worst one(s).')
        return random.choice(best_tests)

    def assess_m_test(self, alpha, beta, result):
        """
        prune the search space upon learning the result of the M-test on the couple (alpha, beta)
        """
        assert self._day < self.n and self._time == 0 and self._moment == 1
        self._time = 1
        self._moment = 0
        for permutation in list(self.admissible_permutations):
            if (1 if permutation[alpha] == beta else 0) != result:
                self.admissible_permutations.remove(permutation)
        self._couple_probabilities = self.compute_couple_probabilities()  # update the couple probabilities

    def choose_n_test(self):
        """
        return the optimally chosen permutation (a.k.a. sequence "betas") to be tested in the night (N-test)
        """
        assert self._day < self.n and self._time == 1 and self._moment == 0
        self._moment = 1
        # on day 0, just pick one permutation randomly, as it is not worth optimizing
        nr_analyzed_permutations = self.effortAllocator.decide_permutations_nr_to_consider(self)
        pct_analyzed_permutations = nr_analyzed_permutations / len(self.admissible_permutations) * 100
        if self.show_optimization:
            print(f'On this {self.time()}, the solver decides to consider '
                  f'{nr_analyzed_permutations}/{len(self.admissible_permutations)}, '
                  f'i.e. {pct_analyzed_permutations:.5f}% '
                  f'{"randomly chosen " if nr_analyzed_permutations < len(self.admissible_permutations) else ""}'
                  f'admissible permutation(s).')
        if nr_analyzed_permutations == 1:
            return random.choice(list(self.admissible_permutations))
        elif len(self.admissible_permutations) > nr_analyzed_permutations:
            candidate_permutations = random.sample(tuple(self.admissible_permutations), nr_analyzed_permutations)
        else:
            candidate_permutations = self.admissible_permutations
        candidate_permutations = {cp: -1 for cp in candidate_permutations}
        min_rank, max_rank = float('inf'), float('-inf')
        best_tests = []
        initial_frequencies = np.zeros((self.n + 1), dtype=int)
        for candidate_permutation in candidate_permutations:
            test_frequencies = initial_frequencies.copy()
            for ap in self.admissible_permutations:
                nr_overlaps = 0
                for c, a in zip(ap, candidate_permutation):
                    if c == a:
                        nr_overlaps += 1
                test_frequencies[nr_overlaps] += 1
            rank = self.heuristic.rank(test_frequencies, len(self.admissible_permutations))
            candidate_permutations[candidate_permutation] = rank
            if rank < min_rank:
                best_tests = [candidate_permutation]
                min_rank = rank
            elif rank == min_rank:
                best_tests.append(candidate_permutation)
            if rank > max_rank:
                max_rank = rank
        if self.show_optimization:
            rank_range = max_rank - min_rank
            print(f'On this {self.time()}, the chosen best permutation(s) have rank {min_rank:.3f}, '
                  f'{rank_range:.3f} better than the worst one(s).')
            if len(candidate_permutations) <= self.max_permutations_show:
                print('Here is the ranking of the analyzed permutations:')
                for permutation, rank in sorted(candidate_permutations.items(), key=lambda item: item[1]):
                    print(f'rank({permutation}) = {rank:.3f}')
        return random.choice(best_tests)

    def assess_n_test(self, betas, result):
        """
        prune the search space upon learning the result of the N-test on the permutation "betas"
        """
        assert self._day < self.n and self._time == 1 and self._moment == 1
        self._day += 1
        self._time = 0
        self._moment = 0
        for permutation in list(self.admissible_permutations):
            if sum([1 if b == p else 0 for b, p in zip(betas, permutation)]) != result:
                self.admissible_permutations.remove(permutation)
        self._couple_probabilities = self.compute_couple_probabilities()  # update the couple probabilities

    def print_admissible_permutations_stats(self):
        """
        print the number of still admissible permutations and the corresponding information entropy
        """
        entropy = math.log2(len(self.admissible_permutations))
        print(f'The solver knows there are {len(self.admissible_permutations)} admissible permutation(s) left now, '
              f'corresponding to {entropy:.3f} bits of information.')


    def print_couple_stats(self):
        """
        print all couple probabilities and entropies, if available
        """
        np.set_printoptions(formatter={'float_kind': '{:.3f}'.format})  # formatting options for floats
        if self._couple_probabilities is not None:
            print('Based on the still admissible permutations, the solver knows these couple probabilities:')
            print(self._couple_probabilities)
            variance = np.var(self._couple_probabilities)
            stdev = variance ** 0.5
            print(f'On this {self.time()}, '
                  f'the couple probabilities have variance = {variance:.3f} and stdev = {stdev:.3f}.')
