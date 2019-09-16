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
        # update priors
        self.ps = ps
        posteriors = []
        for i, h in enumerate(self.hs):
            # Calculate P(h|o) for each hypothesis h given observation o
            posteriors.append((self.ps[i]*self.likelihood(o, h))/self.norm_constant(o))
        return posteriors

    # Computes all posterior probabilities for each observation given
    # e.g. if we have 2 hypothesis h1 and h2 in self.hs,
    # and os consists of 2 observations o1 and o2,
    # We return [ [P(h1|o1), P[h2|o1], [P[h1|o2], P[h2|o2] ]
    def compute_posterior(self, os):
        posteriors = []
        for i, o in enumerate(os):
            posteriors.append(self.single_posterior_update(o, self.ps))

        return posteriors

if __name__ == '__main__':

    print("===================")
    print("THE COOKIE PROBLEM")
    print("===================")
    hypos = ["Bowl1", "Bowl2"]
    priors = [0.5, 0.5]
    obs = ["chocolate", "vanilla"]
    # e.g. likelihood[0][1] corresponds to the likelihood of Bowl1 and vanilla, or 35/50
    likelihood = [[15/50, 35/50], [30/50, 20/50]]

    b = Bayes(hypos, priors, obs, likelihood)

    l = b.likelihood("chocolate", "Bowl1")
    print("likelihood(chocolate, Bowl1) = %s " % l)

    n_c = b.norm_constant("vanilla")
    print("normalizing constant for vanilla: %s" % n_c)

    p_1 = b.single_posterior_update("vanilla", [0.5, 0.5])
    print("vanilla - posterior: %s" % p_1)

    p_2 = b.compute_posterior(['chocolate', 'vanilla'])
    print("chocolate, vanilla - posterior: %s" % p_2)

    print("===================")
    print("THE ARCHERY PROBLEM")
    print("===================")

    priors = [0.5]



