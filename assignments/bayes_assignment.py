from core.Bayes import Bayes

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

    q1 = b.single_posterior_update("vanilla", [0.5, 0.5])
    print("Q1: Probability that vanilla came from bowl 1: %s" % q1[b.hs.index("Bowl1")])

    b = Bayes(hypos, priors, obs, likelihood)

    q2 = b.compute_posterior(['chocolate', 'vanilla'])
    print("Q2: probability that [chocolate, vanilla] came from Bowl2: %s" % q2[b.hs.index("Bowl2")])

def the_archery_problem():
    print("===================")
    print("THE ARCHERY PROBLEM")
    print("===================")

    hypos = ["beginner", "intermediate", "advanced", "expert"]
    priors = [0.25, 0.25, 0.25, 0.25]
    observations = ["yellow", "red", "blue", "black", "white"]
    likelihood = [[0.05, 0.1, 0.4, 0.25, 0.2],
                  [0.1, 0.2, 0.4, 0.2, 0.1],
                  [0.2, 0.4, 0.25, 0.1, 0.05],
                  [0.3, 0.5, 0.125, 0.05, 0.025]]

    b = Bayes(hypos, priors, observations, likelihood)

    result = b.compute_posterior(['yellow', 'white', 'blue', 'red', 'red', 'blue'])

    q3 = result[b.hs.index('intermediate')]
    q4 = b.hs[result.index(max(result))]
    print("yellow, white, blue, red, red, blue - posterior: %s" % result)
    print("Q3: Probability of the archer being at an intermediate level: %s" % q3)
    print("Q4: Most likely level of the archer: %s" % q4)

if __name__ == '__main__':
    the_cookie_problem()
    the_archery_problem()