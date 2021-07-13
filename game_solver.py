import itertools
import math
import numpy as np
import random
from pprint import pprint


# global formatting options for floats
FF = '{:.3f}'.format
np.set_printoptions(formatter={'float_kind': FF})


def entropy_delta(freqs, n_total=None):
    """
    entropy of a test with switched sign
    :param freqs: np.array containing either the frequencies or the probabilities of all test results
    :param n_total: sum of freqs (to be provided, optionally, to save calculations)
    :return: - entropy, float
    """
    if n_total is None:
        n_total = sum(freqs)
    freqs = freqs / n_total
    return sum([freq * math.log2(freq) for freq in freqs if freq > 0.0])


class GameSolver:
    """
    Try to solve the information theory game underlying "Are You the One?", 1st season's rules.
    See project's README.md for more info on the algorithm used.
    """
    def __init__(self, n, freqs2rank=None, max_permutations_show=24, max_optimize=1e7, show_optimization=False):
        """
        init
        :param n: nr. of couples
        :param freqs2rank: function with signature freqs: np.array, optional n_total: int -> rank: float
            where freqs are the test outcome frequencies and n_total is their sum;
            the solver tries to find the lowest ranking possible test.
            The default ranking criterion is by test outcome frequencies in descending order.
        :param max_permutations_show: max. number of permutations being shown when showing the search space
        :param max_optimize: max. number of comparisons between permutations for calculating N-test entropies
        :param show_optimization: True if the internal "reasoning" must be shown, else False
        """
        self.n = n
        self.freqs2rank = freqs2rank if freqs2rank else lambda freqs, n_total: sorted(freqs, reverse=True)
        self.alphas = list(range(n))
        self.betas = self.alphas
        self.admissible_permutations = set(itertools.permutations(self.betas))
        self.n_admissible_permutations = len(self.admissible_permutations)
        self.max_admissible_permutations = int(math.factorial(self.n))
        self.max_permutations_show = max_permutations_show
        self.max_optimize = max_optimize
        self.show_optimization = show_optimization
        self._couple_probabilities = np.ones((n, n)) / n
        self._day = 0
        self._time = 0  # 0 = morning, 1 = night
        self._moment = 0  # 0 = before testing, 1 = after testing
        # calculating upper bounds
        self.m_test_rank_ubound = self.freqs2rank(
            np.array([self.max_admissible_permutations, 0]),
            self.max_admissible_permutations
        )
        self.n_test_rank_ubound = self.freqs2rank(
            np.array([self.max_admissible_permutations] + [0] * (n)),
            self.max_admissible_permutations
        )

    def get_couple_probabilities(self):
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
        return the optimally chosen couple (alpha, beta) to be tested in the morning (M-test)
        """
        assert self._day < self.n and self._time == 0 and self._moment == 0
        self._moment = 1
        if self.n_admissible_permutations == self.max_admissible_permutations:
            return random.randrange(self.n), random.randrange(self.n)
        self._couple_probabilities = self.get_couple_probabilities()  # recalculate just in time as they become needed
        min_rank = self.m_test_rank_ubound
        candidate_tests = []
        for r in range(self.n):
            for c in range(self.n):
                prob = self._couple_probabilities[r, c]
                rank = self.freqs2rank(np.array([prob, 1. - prob]), 1.)
                if rank < min_rank:
                    candidate_tests = [(r, c)]
                    min_rank = rank
                elif rank == min_rank:
                    candidate_tests.append((r, c))
        if self.show_optimization:
            self.print_couple_stats()
        return random.choice(candidate_tests)

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
        self.n_admissible_permutations = len(self.admissible_permutations)
        self._couple_probabilities = None  # delete the no longer valid couple probabilities
        if self.show_optimization:
            self.print_admissible_permutations()

    def choose_n_test(self):
        """
        return the optimally chosen permutation (a.k.a. sequence "betas") to be tested in the night (N-test)
        """
        assert self._day < self.n and self._time == 1 and self._moment == 0
        self._moment = 1
        nr_analyzed_permutations = min(
            max(
                1,
                int(self.max_optimize // (self.n * self.n_admissible_permutations))
            ),
            self.n_admissible_permutations
        )
        if nr_analyzed_permutations == 1:
            return random.choice(list(self.admissible_permutations))
        elif self.n_admissible_permutations > nr_analyzed_permutations:
            candidate_permutations = random.sample(tuple(self.admissible_permutations), nr_analyzed_permutations)
        else:
            candidate_permutations = self.admissible_permutations
        candidate_permutations = {cp: None for cp in candidate_permutations}
        min_rank = self.n_test_rank_ubound
        candidate_tests = []
        initial_frequencies = np.zeros((self.n + 1), dtype=int)
        for candidate_permutation in candidate_permutations:
            test_frequencies = initial_frequencies.copy()
            for ap in self.admissible_permutations:
                nr_overlaps = 0
                for c, a in zip(ap, candidate_permutation):
                    if c == a:
                        nr_overlaps += 1
                test_frequencies[nr_overlaps] += 1
            rank = self.freqs2rank(test_frequencies, self.n_admissible_permutations)
            candidate_permutations[candidate_permutation] = rank
            if rank < min_rank:
                candidate_tests = [candidate_permutation]
                min_rank = rank
            elif rank == min_rank:
                candidate_tests.append(candidate_permutation)
        if self.show_optimization:
            print(f'The solver randomly chooses and analyzes {len(candidate_permutations)} '
                  f'admissible permutation{"s" if len(candidate_permutations) > 1 else ""}:')
            sorted_candidates = sorted(candidate_permutations.items(), key=lambda x: x[1])
            print(f'The N-test using the admissible permutation {list(sorted_candidates[-1][0])} '
                      f'has rank {sorted_candidates[-1][1]}.')
            if len(candidate_permutations) > 1:
                print('...')
                print(f'The N-test using the admissible permutation {list(sorted_candidates[0][0])} '
                      f'has rank {sorted_candidates[0][1]}.')
        return random.choice(candidate_tests)

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
        self.n_admissible_permutations = len(self.admissible_permutations)
        if self.show_optimization:
            self.print_admissible_permutations()

    def print_admissible_permutations(self):
        """
        print all still admissible permutations, or just their number if they are more than self.max_permutations_show
        """
        message = f'The solver knows that there {"are" if self.n_admissible_permutations > 1 else "is"} ' + \
                  f'{self.n_admissible_permutations} ' + \
                  f'admissible permutation{"s" if self.n_admissible_permutations > 1 else ""} now'
        if self.n_admissible_permutations > self.max_permutations_show:
            print(message + '.')
        else:
            print(message + ':')
            pprint(sorted(self.admissible_permutations))

    def print_couple_stats(self):
        """
        print all couple probabilities and entropies, if available
        """
        if self._couple_probabilities is not None:
            print('Based on the admissible permutations, the solver knows that these are the couple probabilities:')
            print(self._couple_probabilities)

