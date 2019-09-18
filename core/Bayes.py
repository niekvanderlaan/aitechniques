class Bayes(object):
    def __init__(self, hs, ps, os, ls):
        # List of hypothesis
        self.hs = hs.copy()

        # List of priors
        self.ps = ps.copy()

        # List of observations
        self.os = os.copy()

        # Likelihood array
        # ls is formatted as a double array,
        # storing the hypotheses as first and second dimensions respectively.
        # i.e. ls[0][1] corresponds to the likelihood of hs[0] and os[1]
        self.ls = ls.copy()

    # Likelihood P(O|H)
    def likelihood(self, o, h):
        return self.ls[self.hs.index(h)][self.os.index(o)]

    # Normalizing constant P(O)
    # P(O) = SUMi{ P(Hi) * P(O|Hi) }
    def norm_constant(self, o):
        norm = 0
        #Store index at which observation O is stored for reference
        o_index = self.os.index(o)
        for i in range(0, len(self.hs)):
            norm += self.ps[i]*self.ls[i][o_index]
        return norm

    # Posterior Probability P(H|O)
    # P(H|O) = ( P(H) * P(O|H) ) / P(O)
    def single_posterior_update(self, o, ps):
        posteriors = []
        for i, h in enumerate(self.hs):
            # Calculate P(h|o) for each hypothesis h given observation o
            posteriors.append((ps[i]*self.likelihood(o, h))/self.norm_constant(o))

        # update priors
        self.ps = posteriors.copy()

        return posteriors

    # Computes all posterior probabilities given a sequence of observations
    def compute_posterior(self, os):
        posteriors = []
        for i, o in enumerate(os):
            posteriors = self.single_posterior_update(o, self.ps)

        return posteriors

if __name__ == '__main__':
    print("Bayes rule")



