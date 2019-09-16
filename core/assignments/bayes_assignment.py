from aitechniques.core.Bayes import Bayes


def the_cookie_problem():
    print("===================")
    print("THE COOKIE PROBLEM")
    print("===================")
    hypos = ["Bowl1", "Bowl2"]
    priors = [0.5, 0.5]
    obs = ["chocolate", "vanilla"]
    # e.g. likelihood[0][1] corresponds to the likelihood of Bowl1 and vanilla, or 35/50
    likelihood = [[15 / 50, 35 / 50], [30 / 50, 20 / 50]]

    b = Bayes(hypos, priors, obs, likelihood)

    l = b.likelihood("chocolate", "Bowl1")
    print("likelihood(chocolate, Bowl1) = %s " % l)

    n_c = b.norm_constant("vanilla")
    print("normalizing constant for vanilla: %s" % n_c)

    p_1 = b.single_posterior_update("vanilla", [0.5, 0.5])
    print("vanilla - posterior: %s" % p_1)

    p_2 = b.compute_posterior(['chocolate', 'vanilla'])
    print("chocolate, vanilla - posterior: %s" % p_2)


if __name__ == '__main__':
    result = the_cookie_problem()