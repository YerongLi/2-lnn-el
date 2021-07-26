import torch
import torch.nn as nn
import torch.nn.functional as func
from numpy import log
from lnn.src.meta_rule.cdd_interface import cdd_lnn


def gen_sigm(theta, partial):
    '''
    Computes the coefficients of sigmoid of the form 1 / (1 + exp(- a*x + b))

    Parameters:
    ----------
    theta: the value of the sigmoid at x=theta
    partial: the gradient of the sigmoid at x=theta

    Returns: coefficients a and b (in that order)

    ONLY MEANT FOR USE WITHIN lnn_operators.py
    '''
    return partial / theta / (1 - theta), log(theta / (1 - theta)) - partial / (1 - theta)


class and_lukasiewicz(nn.Module):
    def __init__(self, alpha, arity, with_slack):
        '''
        Initializes an LNN conjunction (weighted Lukasiewicz logic).

        The cdd member variable is where the double description 
        method related stuff happens.

        Parameters: 
        ---------- 

        alpha: hyperparameter that defines how farther from
        traditional logic we want to go. Note that, 0.5 < alpha < 1

        arity: the number of inputs to this conjunction. try to stick
        to 2

        with_slack: set to true if you want the version with
        slack. see Logical_Neural_Networks.pdf.
        '''

        super().__init__()

        self.alpha = alpha
        self.arity = arity
        self.with_slack = with_slack
        self.cdd = cdd_lnn(alpha, arity, with_slack)

        # differentiable clamping
        # need a function whose gradient is non-zero everywhere
        # generalized sigmoid (gen_sigm): 1 / (1 + exp(-(ax+b)))
        # < lower: gen_sigm(x; a_lower, b_lower)
        # > upper: gen_sigm(x; a_upper, b_upper)
        # given theta, partial: solve for a,b such that
        #  - gen_sigm(theta; a,b) = theta
        #  - diff.gen_sigm(theta; a,b) =  gen_sigm(theta; a,b) (1 -  gen_sigm(theta; a,b)) = partial
        # soln is given by gen_a, gen_b (see functions above)
        self.lower = 0.01
        self.upper = 0.99
        partial = 0.01
        self.lower_a, self.lower_b = gen_sigm(self.lower, partial)
        self.upper_a, self.upper_b = gen_sigm(self.upper, partial)
        self.param_arg_weights = nn.Parameter(torch.zeros(self.arity).uniform_(0, 1))
        self.param_beta = nn.Parameter(torch.zeros(1).uniform_(0, 1))

    def forward(self, x):
        '''
        Forward function does three things:

        - Requests and obtains beta and argument weights
        from cdd (uses rays and points from double description)

        - Computes the input to the clamping function

        - Depending on the above input, uses the appropriate
        switch to compute the output

        Parameters:
        ----------

        x: 2D tensor. Shape is (*, self.arity). Assumes each row
        contains separate input to the conjunction operator.

        Returns:
        -------

        column vector if non-slack version is used. also, returns a
        scalar (sum of slacks) if version with slacks is used.
        '''

        sum_slacks = None
        #if self.with_slack:
        #    beta, arg_weights, sum_slacks = self.cdd()
        #else:
        #    beta, arg_weights = self.cdd()
        arg_weights = func.relu(self.param_arg_weights)
        beta = func.relu(self.param_beta)


        #arg_weights = func.softplus(self.param_arg_weights)
        #beta = func.softplus(self.param_beta)
        # print('>>', self.param_arg_weights, self.param_beta)

        tmp = -torch.add(torch.unsqueeze(torch.mv(1 - x, torch.t(arg_weights)), 1), -beta)


        # ret = torch.clamp(tmp, 0, 1) #creates issues during learning. gradient vanishes if tmp is outside [0,1]
        ret = torch.where(tmp > self.upper, func.sigmoid(self.upper_a * tmp + self.upper_b), \
                          torch.where(tmp < self.lower, func.sigmoid(self.lower_a * tmp + self.lower_b), tmp))

        #print('lnn:and',ret)

        
        if self.training and self.with_slack:
            return ret, sum_slacks
        else:
            return ret


class or_lukasiewicz(nn.Module):
    def __init__(self, alpha, arity, with_slack):
        '''
        Initializes an LNN disjunction (weighted Lukasiewicz logic).

        Uses an LNN conjunction since:
                          or(x1,x2...) = 1 - and(1-x1,1-x2,...)
        The cdd member variable is where the double description
        method related stuff happens.

        Parameters:
        ----------

        alpha: hyperparameter that defines how farther from
        traditional logic we want to go. Note that, 0.5 < alpha < 1

        arity: the number of inputs to this disjunction. try to stick
        to 2

        with_slack: set to true if you want the version with
        slack. see Logical_Neural_Networks.pdf.
        '''

        super().__init__()
        self.AND = and_lukasiewicz(alpha, arity, with_slack)

    def forward(self, x):
        '''
        Forward function invokes LNN conjunction operator. Returns
        whatever the conjunction returns. Note that, depending on
        whether with_slack was set to True or False, may return 2
        things (the results of the disjunction and sum of slacks)
        or 1 (just the results of the disjunction)

        Parameters:
        ----------

        x: 2D tensor. Shape is (*, self.arity). Assumes each row
        contains separate input to the disjunction operator.

        Returns:
        -------

        column vector if non-slack version is used. also, returns a
        scalar (sum of slacks) if version with slacks is used.
        '''

        ret = 1 - self.AND(1 - x)
        #print('lnn:or',ret)

        return ret


class and_product(nn.Module):
    def __init__(self):
        '''
        Initializes a product t-norm conjunction. Is parameter-free.
        Use this if you don't know arity or if your application needs
        a conjunction operator that can take variable number of
        arguments.

        '''
        super().__init__()

    def forward(self, x):
        '''
        Forward function returns product of inputs.

        Parameters:
        ----------

        x: 2D tensor. Shape is (*, self.arity). Assumes each row
        contains separate input to the disjunction operator.

        Returns: column vector
        -------
        '''

        return torch.exp(torch.sum(torch.log(x + 1e-17), 1, keepdim=True))  # multiplies row-wise


class or_max(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.max(x, 1, keepdim=True)[0]


class or_sum(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sum(x, 1, keepdim=True)


class predicates(nn.Module):
    def __init__(self, num_predicates, body_len):
        '''
        Use these to express a choice amongst predicates. For use when
        learning rules.

        Parameters:
        ----------

        num_predicates: The domain size of predicates
        body_len: The number of predicates to choose
        '''
        super().__init__()
        self.log_weights = nn.Parameter(torch.zeros(body_len, num_predicates).uniform_(-.01, .01))

    def forward(self, x):
        '''
        Forward function computes the attention weights and returns the result of mixing predicates.

        Parameters:
        ----------

        x: a 2D tensor whose number of columns should equal self.num_predicates

        Returns: A 2D tensor with 1 column
        -------
        '''
        weights = self.get_params()
        return func.linear(x, weights)

    def get_params(self):
        return func.softmax(self.log_weights, dim=1)


def negation(x):
    '''
    Negation is just pass-through, parameter-less.
    '''
    return 1 - x
