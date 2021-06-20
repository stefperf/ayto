import numpy as np


class GameOracle:
    """
    Randomly choose the secret perfect matches to be guessed, then compute the M- and N-test results
    """
    def __init__(self, n, betas=None):
        """
        set n secret perfect matches randomly or, optionally, explicitly choosing them
        """
        assert isinstance(n, int) and n > 0
        self.n = n
        self.alphas = list(range(n))
        if betas:
            self._validate_betas(betas)
            self.betas = betas
        else:
            self.betas = np.random.permutation(list(range(n)))
       
    def _validate_betas(self, betas):
        assert sorted(betas) == self.alphas
            
    def m_test(self, alpha, beta):
        """
        morning test (M-test): return 1 if (alpha, beta) is a perfect match, else 0
        """
        assert alpha in self.alphas
        assert beta in self.betas
        return 1 if self.betas[alpha] == beta else 0
    
    def n_test(self, betas):
        """
        night test (N-test): return the nr. of perfect matches when pairing the sorted alphas with the betas so ordered
        """
        self._validate_betas(betas)
        return sum([1 if beta == true_beta else 0 for beta, true_beta in zip(betas, self.betas)])
