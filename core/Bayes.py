class Bayes:
    def __init__(self, hs, ps, os, ls):
        self.hs = hs
        self.ps = ps,
        self.os = os
        self.ls = ls

    def likelihood(self, o, h):
        return self.ls[self.hs.index(h)][self.os.index(o)]

    def norm_constant(self, o):
        norm = 0
        o_index = self.os.index(o)
        for i in range(0, len(self.hs)):
            norm += self.hs[i]*self.ls[i][o_index]
        return norm

    def single_posterior_update(self, o, ps):
        self.ps = ps
        posteriors = []
        for i, h in enumerate(self.hs):
            posteriors.append((self.ps[i]*self.likelihood(o, h))/self.norm_constant(o))
        return posteriors

if __name__ == '__main__':
    hypos = ["Bowl1", "Bowl2"]
    priors = [0.5, 0.5]
    obs = ["chocolate", "vanilla"]
    likelihood = [[15/50, 35/50], [30/50, 20/50]]

    b = Bayes(hypos, priors, obs, likelihood)

    l = b.likelihood("chocolate", "Bowl1")
    print("likelihood(chocolate, Bowl1) = %s " % l)


