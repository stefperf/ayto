import numpy as np


class GameOracle:
    def __init__(self, n, betas=None):
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
        assert alpha in self.alphas
        assert beta in self.betas
        return 1 if self.betas[alpha] == beta else 0
    
    def n_test(self, betas):
        self._validate_betas(betas)
        return sum([1 if beta == true_beta else 0 for beta, true_beta in zip(betas, self.betas)])
