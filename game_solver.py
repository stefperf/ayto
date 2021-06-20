import itertools
import math
import numpy as np
import random
from pprint import pprint


# global formatting options for floats
FF = '{:.3f}'.format
np.set_printoptions(formatter={'float_kind': FF})


class GameSolver:
    """
    Try to solve the information theory game underlying "Are You the One?", 1st season with an approach that:
    1. uses the (still) admissible permutations / couplings as the search space
    2. exactly prunes the search space based on every test result
    3. tries to optimally choose one test at a time, in a greedy way (which may or may not be optimal)
    4. chooses one of the M-test with maximum information entropy among all possible M-tests
    5. chooses one of the N-test with maximum information entropy among an adaptive number of randomly chosen possible
      N-tests; the number is increased as the search space and consequently the computational cost grow smaller.

     The speed-up trick used in 5. works pretty well because, at the start of the game, all possible N-tests have more
     or less the same entropy, due to the symmetrical nature of the problem; the possible N-tests start having
     significantly different entropies only later, as the search space grows smaller and consequently computing their
     entropies becomes faster.
     However, 5. could be easily parallelized in order to assess all possible N-tests.

    """
    def __init__(self, n, max_show=24, max_optimize=1e7, show_optimization=False):
        self.n = n
        self.alphas = list(range(n))
        self.betas = self.alphas
        self.admissible_permutations = set(itertools.permutations(self.betas))
        self.nr_admissible_permutations = len(self.admissible_permutations)
        self.max_admissible_permutations = int(math.factorial(self.n))
        self.max_show = max_show
        self.max_optimize = max_optimize
        self.show_optimization = show_optimization
        self._couple_probabilities = np.ones((n, n)) / n
        self._day = 0
        self._time = 0  # 0 = morning, 1 = night
        self._moment = 0  # 0 = before testing, 1 = after testing

    def get_couple_probabilities(self):
        couple_probabilities = np.zeros((self.n, self.n), dtype=float)
        incr = 1. / len(self.admissible_permutations)
        for ap in self.admissible_permutations:
            for alpha, beta in enumerate(ap):
                couple_probabilities[alpha, beta] += incr
        return couple_probabilities

    def choose_m_test(self):
        assert self._day < self.n and self._time == 0 and self._moment == 0
        self._moment = 1
        if self.nr_admissible_permutations == self.max_admissible_permutations:
            return random.randrange(self.n), random.randrange(self.n)
        self._couple_probabilities = self.get_couple_probabilities()  # recalculate just in time as they become needed
        distances_from_half = np.abs(self._couple_probabilities - 0.5)
        min_dist = 2.
        candidate_coords = []
        for r in range(self.n):
            for c in range(self.n):
                if math.isclose(distances_from_half[r, c], min_dist):
                    candidate_coords.append((r, c))
                elif distances_from_half[r, c] < min_dist:
                    min_dist = distances_from_half[r, c]
                    candidate_coords = [(r, c)]
        prob = self._couple_probabilities[candidate_coords[0]]
        test_entropy = sum([-p * math.log2(p) for p in [prob, 1. - prob] if p > 0.])
        if self.show_optimization:
            self.print_couple_probabilities()
        return random.choice(candidate_coords)

    def assess_m_test(self, alpha, beta, result):
        assert self._day < self.n and self._time == 0 and self._moment == 1
        self._time = 1
        self._moment = 0
        for permutation in list(self.admissible_permutations):
            if (1 if permutation[alpha] == beta else 0) != result:
                self.admissible_permutations.remove(permutation)
        self.nr_admissible_permutations = len(self.admissible_permutations)
        self._couple_probabilities = None  # delete the no longer valid couple probabilities
        if self.show_optimization:
            self.print_admissible_permutations()

    def choose_n_test(self):
        assert self._day < self.n and self._time == 1 and self._moment == 0
        self._moment = 1
        nr_analyzed_permutations = min(
            max(
                1,
                int(self.max_optimize // (self.n * self.nr_admissible_permutations))
            ),
            self.nr_admissible_permutations
        )
        if nr_analyzed_permutations == 1:
            return random.choice(list(self.admissible_permutations))
        elif self.nr_admissible_permutations > nr_analyzed_permutations:
            candidate_permutations = random.sample(tuple(self.admissible_permutations), nr_analyzed_permutations)
        else:
            candidate_permutations = self.admissible_permutations
        candidate_permutations = {cp: None for cp in candidate_permutations}
        highest_entropy = 0.
        highest_entropy_candidate = None
        initial_probabilities = np.zeros((self.n + 1))
        for cp in candidate_permutations:
            probabilities = initial_probabilities.copy()
            for ap in self.admissible_permutations:
                nr_overlaps = 0
                for c, a in zip(ap, cp):
                    if c == a:
                        nr_overlaps += 1
                probabilities[nr_overlaps] += 1
            probabilities /= self.nr_admissible_permutations
            test_entropy = 0.
            for p in probabilities:
                if p > 0.:
                    test_entropy -= p * math.log2(p)
            candidate_permutations[cp] = test_entropy
            if test_entropy > highest_entropy:
                highest_entropy = test_entropy
                highest_entropy_candidate = cp
        if self.show_optimization:
            print(f'The solver randomly chooses and analyzes these {len(candidate_permutations)} '
                  f'admissible permutations:')
            for (cp, test_entropy) in sorted(candidate_permutations.items(), key=lambda x: x[1]):
                print(f'An N-test using the admissible permutation {list(cp)} '
                      f'would be worth {FF(test_entropy)} bits of additional information.')
        return highest_entropy_candidate

    def assess_n_test(self, betas, result):
        assert self._day < self.n and self._time == 1 and self._moment == 1
        self._day += 1
        self._time = 0
        self._moment = 0
        for permutation in list(self.admissible_permutations):
            if sum([1 if b == p else 0 for b, p in zip(betas, permutation)]) != result:
                self.admissible_permutations.remove(permutation)
        self.nr_admissible_permutations = len(self.admissible_permutations)
        if self.show_optimization:
            self.print_admissible_permutations()

    def print_admissible_permutations(self):
        message = f'The solver knows that there {"are" if self.nr_admissible_permutations > 1 else "is"} ' + \
                  f'{self.nr_admissible_permutations} ' + \
                  f'admissible permutation{"s" if self.nr_admissible_permutations > 1 else ""} now'
        if self.nr_admissible_permutations > self.max_show:
            print(message + '.')
        else:
            print(message + ':')
            pprint(sorted(self.admissible_permutations))

    def print_couple_probabilities(self):
        if self._couple_probabilities is not None:
            print('Based on the admissible permutations, the solver knows that these are the couple probabilities:')
            print(self._couple_probabilities)
            test_entropies = np.zeros(self._couple_probabilities.shape)
            for prob_matrix in (self._couple_probabilities, 1. - self._couple_probabilities):
                for row in range(self.n):
                    for col in range(self.n):
                        prob = prob_matrix[row, col]
                        if prob > 0.:
                            test_entropies[row, col] -= prob * math.log2(prob)
            print('Therefore, using each couple for an M-test would be worth as many bits of additional information:')
            print(test_entropies)
