import sys
sys.path.append('../../src/meta_rule/')

from lnn_operators import and_lukasiewicz, or_lukasiewicz, negation
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

def main():
    #this is a hyperparameter
    alpha = 0.8

    op_and1 = and_lukasiewicz(alpha, 2, False)
    op_and2 = and_lukasiewicz(alpha, 2, False)
    op_or = or_lukasiewicz(alpha, 2, False)

    #to train a xor we need its truth table
    x = torch.from_numpy(np.array([[0, 0], \
                                   [0, 1], \
                                   [1, 0], \
                                   [1, 1]])).float()

    #the target values for each row in the truth table (xor)
    y = torch.from_numpy(np.array([[0], \
                                   [1], \
                                   [1], \
                                   [0]])).float()

    loss_fn = nn.BCELoss()
    optimizer = optim.Adam([{'params': op_or.parameters()}, \
                            {'params': op_and1.parameters()}, \
                            {'params': op_and2.parameters()}], lr=0.1)

    for iter in range(100):
        op_or.train()
        op_and1.train()
        op_and2.train()
        optimizer.zero_grad()

        x0 = x[:,0].view(-1,1)
        x1 = x[:,1].view(-1,1)
        yhat = op_or(torch.cat((op_and1(torch.cat((x0, negation(x1)), 1)), \
                                op_and2(torch.cat((negation(x0), x1), 1))), 1))
        loss = loss_fn(yhat, y)

        print("Iteration " + str(iter) + ": " + str(loss.item()))
        loss.backward()
        optimizer.step()

    #check to see output of xor post-training
    x0 = x[:,0].view(-1,1)
    x1 = x[:,1].view(-1,1)
    yhat = op_or(torch.cat((op_and1(torch.cat((x0, negation(x1)), 1)), \
                            op_and2(torch.cat((negation(x0), x1), 1))), 1))
    check_values = torch.cat((yhat, y), 1)
    print("------- Checking outputs (left) vs ground truth (right): -----")
    print(check_values.detach())

    #LNN parameters: post-training (we have 3 sets of beta, argument weights)
    print("--------------- LNN Parameters (post-training) ---------------")
    beta_or, argument_wts_or = op_or.AND.cdd()
    beta_and1, argument_wts_and1 = op_and1.cdd()
    beta_and2, argument_wts_and2 = op_and2.cdd()

    np.set_printoptions(precision=3, suppress=True)
    print("OR (beta, argument weights): " \
          + str(np.around(beta_or.item(), decimals=3)) + " " \
          + str(argument_wts_or.detach().numpy()))
    print("AND1 (beta, argument weights): " \
          + str(np.around(beta_and1.item(), decimals=3)) + " " \
          + str(argument_wts_and1.detach().numpy()))
    print("AND2 (beta, argument weights): " \
          + str(np.around(beta_and2.item(), decimals=3)) + " " \
          + str(argument_wts_and2.detach().numpy()))

    
if __name__ == "__main__":
    main()

