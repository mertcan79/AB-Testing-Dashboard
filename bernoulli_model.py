import numpy as np
from scipy.special import loggamma
import emcee
from scipy.special import gammaln


class BernoulliModel:
    """

    The Bernoulli model class will help in the calculation of conversion rates and other topics.

    """

    def __init__(self):
        self.a = 0.0

    def log_beta_coefficient(self, a, b):
        """
        The log of the Beta coefficient

        B = Gamma(a)*Gamma(b)/Gamma(a+b)
        """
        return loggamma(a)+loggamma(b)-loggamma(a+b)

    def log_prior(self, theta):
        """
        The prior that may be used in the analysis.
        """

        x = theta

        if 0.0 <= x <= 1.0:
            return 0.0

        return -np.inf

    def log_likelihood(self, theta, n_total, n_success):
        """
        The log-likelihood of the beta distribution

        P(x,a,b) = x^(a-1) * (1-x)^(b-1)* Coef(a,b)

        """

        x = theta
        a = n_success+1
        b = n_total-n_success+1

        return (a-1)*np.log(x)+(b-1)*np.log(1.0-x)-self.log_beta_coefficient(a, b)

    def log_probability(self, theta, n_total, n_success):
        """
        log_probability = log_prior + log_likelihood
        """

        lp = self.log_prior(theta)

        if not np.isfinite(lp):
            return -np.inf

        return lp + self.log_likelihood(theta, n_total, n_success)

    def generate_posterior(self, n_total, n_success, nwalkers=30, eps=1e-5, n_iter=3000):
        """
        Compute the probability distribution of the conversion rates x
        """

        ndim = 1
        x0 = np.array([n_success / n_total])
        pos = x0 + eps * np.random.randn(nwalkers, ndim)
        nwalkers, ndim = pos.shape


        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, self.log_probability, args=(n_total, n_success)
        )

        sampler.run_mcmc(pos, n_iter, progress=False)

        flat_samples = sampler.get_chain(flat=True)
        samples = flat_samples[:, 0]

        return samples

    def generate_binomial_distribution():
        """
        Compute the probabilities for a binomial distribution model
        """

        p = 0.0


        return p

    def log_beta(self, x: float, y: float) -> float:
        """
        The natural logarithm of the beta function. This is used to avoid numerical overflow problems
        associated with factorials.

        Beta(x,y) = Gamma(x)Gamma(y)/Gamma(x+y)

        @param x:
        @param y:
        """

        ln_beta = gammaln(x)+gammaln(y)-gammaln(x+y)

        return ln_beta

    def hypothesis_testing_two_independent_samples(self, n_samples1: int, n1: int, n_samples2: int, n2: int) -> float:
        """
        Conduct a hypothesis test of two independent samples using bayesian analysis.

        Note: We assume the prior for both models is the same.

        P(x1=x2|D) = P(D| x1=x2 )/(P(D|x1=x2)+P(D|x1!=x2))
        P(x1 != x2 ) = 1 - P(x1=x2)

        r  = Bayesian odds ratio = P(x1 != x2)/P(x1 = x2)
        """
        log_r = self.log_beta(n1 + 1, n_samples1 - n1 + 1) + self.log_beta(n2 + 1, n_samples2 - n2 + 1)  # P(x1 != x2)
        log_r -= self.log_beta(n1 + n2 + 1, n_samples1 + n_samples2 - n1 - n2 + 1)  # # P(x1 = x2)
        r = np.exp(log_r)

        # prob that x1 != x2
        px1x2 = r/(1+r)

        # prob that x1=x2
        px12 = 1-px1x2


        results = {
        'P(A: x1 != x2)': px1x2,
        'P(B: x1=x2)': px12,
        'r(A/B)' : r
        }

        return results

    def hypothesis_testing_three_independent_samples(self, nsamples1, n1, nsamples2, n2, nsamples3, n3):
        """
        We compute the probability of five hypothesis

        A. P(D|p1 ! p2 ! p3) = x1 x2 x3
        B. P(D|p2=p3, !p1) = x23*x1
        C. P(D|p1=p3, !p2) = x13*x2
        D. P(D|p1=p2, !p3) = x12*x3
        E. P(D|p1 = p2 = p3) = x123
        """

        # compute the first ratio x123/(x1*x2*x3)
        log_r123 = self.log_beta(n1+n2+n3+1, nsamples1+nsamples2+nsamples3-n1-n2-n3+1)
        log_r123 += -1.0*self.log_beta(n1+1, nsamples1-n1+1)-self.log_beta(n2+1, nsamples2-n2+1)-self.log_beta(n3+1, nsamples3-n3+1)
        log_r12 = self.log_beta(n1+n2+1, nsamples1+nsamples2-n1-n2+1)-self.log_beta(n1+1, nsamples1-n1+1)-self.log_beta(n2+1, nsamples2-n2+1)
        log_r13 = self.log_beta(n1+n3+1, nsamples1+nsamples3-n1-n3+1)-self.log_beta(n1+1, nsamples1-n1+1)-self.log_beta(n3+1, nsamples2-n3+1)
        log_r23 = self.log_beta(n2+n3+1, nsamples2+nsamples3-n2-n3+1)-self.log_beta(n2+1, nsamples2-n2+1)-self.log_beta(n3+1, nsamples3-n3+1)

        r123 = np.exp(log_r123)
        r12 = np.exp(log_r12)
        r13 = np.exp(log_r13)
        r23 = np.exp(log_r23)
        denom = 1.0+r123+r12+r13+r23


        pA = 1.0/denom
        pB = r23/denom
        pC = r13/denom
        pD = r12/denom
        pE = r123/denom

        results = {
            'P(p1 != p2 !=p3|D)': pA,
            'P(p2 = p3, !p1|D)': pB,
            'P(p1 = p3, !p2|D)': pC,
            'P(p1 =p2, !p3 |D)': pD,
            'P(p1=p2=p3|D)': pE
        }

        return results


if __name__ == "__main__":

    model = BernoulliModel()

    # Corresponds to the test
    n_total1 = 10000
    n1 = 0.1*n_total1

    # Corresponds to the control
    n_total2 = 558
    n2 = 0.13*n_total2

    n_total3 = 183900
    n3 = 0.13*n_total3

    print(model.hypothesis_testing_three_independent_samples(n_total1, n1, n_total2, n2, n_total3, n3))


    print("program completed")
