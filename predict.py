#!/usr/bin/env python3

import pickle as pickle
import matplotlib.pyplot as plt
import numpy as np


def predict(X, w, y=None, normazlized = None):
    # X_new: Nsample x (d+1)
    # w: (d+1) x 1
    # y_new: Nsample

    y_hat = np.matmul(X,w) # predicted ( normalized)
    
    if normazlized== False:
        y = y* std_y + mean_y
        y_hat = y_hat * std_y + mean_y

    size_x = len(X)  # size_batch = 10, size_val = 100
    loss = 0         # 1/2M_batch * sum ( (y_hat(m)-y(m))^2 )
    risk = 0         # sum( |y_hat(m) - y(m| )

    for i in range(size_x):
        loss+=1/(2*size_x)* (y_hat[i]-y[i])**2+decay*np.square(np.linalg.norm(w))
        risk+=(1/size_x)* abs(y_hat[i]-y[i])
    
    # loss = np.sum(np.square(np.subtract(y,y_hat)))/(2*size_x)
    # risk = np.sum(np.absolute(np.subtract(y,y_hat)))/size_x

    return y_hat, loss, risk


def train(X_train, y_train, X_val, y_val):
    N_train = X_train.shape[0]
    N_val = X_val.shape[0]

    # initialization
    w = np.zeros([X_train.shape[1], 1])
    # w: (d+1)x1

    losses_train = []
    risks_val = []

    w_best = None
    risk_best = 10000
    epoch_best = 0

    for epoch in range(MaxIter):
        # Do could put cordinate descent here ( For lambda in lambda1,2,3,...k)
             # Do mini-batch gradient descent
        # have loss_this_epoach for each lambda [labmda1_lose, lambda2_lose]
        # compute trainging loss for each lambda [traing_lose_1, 2...]
        # have vaidation error for each labmda
        loss_this_epoch = 0
        for b in range(int(np.ceil(N_train/batch_size))):

            X_batch = X_train[b*batch_size: (b+1)*batch_size]
            y_batch = y_train[b*batch_size: (b+1)*batch_size]            

            y_hat_batch, loss_batch, _= predict(X_batch, w, y_batch,True)
            loss_this_epoch += loss_batch
            
            # TODO: Your code here
            # Mini-batch gradient descent 
            #   gradient(J) [ 27, 1]: 1/M_batch (X(T)Xw-X(T)y)
            #   w(t)  [27,1]      : w(t-1) - alpha* gradient(w(t-1), D_batch)
            gradient = (np.matmul(np.transpose(X_batch), (y_hat_batch-y_batch)))/len(X_batch)+2*decay*w
            w = w-alpha*(gradient+ decay*w)
            

        # TODO: Your code here
        # monitor model behavior after each epoch
        # 1. Compute the training loss by averaging loss_this_epoch 
        batch_number = int(np.ceil(N_train/batch_size))
        training_loss = loss_this_epoch[0]/ batch_number
        losses_train.append(training_loss)

        # 2. Perform validation on the validation set by the risk
        _,_, risk_this_epoch = predict(X_val,w,y_val, False)
        risks_val.append(risk_this_epoch[0])        

        # 3. Keep track of the best validation epoch, risk, and the weights
        if risk_this_epoch< risk_best:
            w_best = w
            risk_best = risk_this_epoch
            epoch_best = epoch

    # print("Best validation epoch: {}\nBest validation risk: {}".format(epoch_best,risk_best[0]))

    # Return some variables as needed
    return  w_best, risk_best, epoch_best ,losses_train, risks_val


############################
# Main code starts here
############################
# Load data
with open("housing.pkl", "rb") as f:
    (X, y) = pickle.load(f)

# X: sample x dimension
# y: sample x 1
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0) 
X_extend = np.square(X)
X_new = np.concatenate( (X, X_extend), axis=1)

# Augment feature
X_ = np.concatenate((np.ones([X_new.shape[0], 1]), X_new), axis=1) # len(X): # of samples ; len(X_[0]): number of features d+1 : 14
# X_: Nsample x (d+1)

# normalize features:
mean_y = np.mean(y)
std_y = np.std(y)

y = (y - np.mean(y)) / np.std(y)

# print(X.shape, y.shape) # It's always helpful to print the shape of a variable

# Randomly shuffle the data
np.random.seed(314)
np.random.shuffle(X_)
np.random.seed(314)
np.random.shuffle(y)

X_train = X_[:300] # normalized
y_train = y[:300]

X_val = X_[300:400]  # normalized
y_val = y[300:400]

X_test = X_[400:]  # normalzied
y_test = y[400:]

#####################
# setting

alpha = 0.001      # learning rate
batch_size = 10    # batch size
MaxIter = 100        # Maximum iteration
decay_set = [3, 1, 0.3, 0.1, 0.03, 0.01]
decay = None        # weight decay


# TODO: Tuning Hyperparameter
best_decay= None  # best hyperparameter with (least validation performance)
best_decay_epoch_best = None
best_decay_val_performance = float("inf")
w_test= None
losses_train = None

for decay in decay_set:
    # print("\nDecay {}: ".format(decay))
    w_best, risk_best, epoch_best, loss, risks_val = train(X_train, y_train, X_val, y_val)  # training and validation
    print("decay: {}, risk_validation: {}".format(decay, risk_best))
    if risk_best < best_decay_val_performance:
        best_decay_val_performance = risk_best
        best_decay_epoch_best = epoch_best
        best_decay = decay
        w_test = w_best
        losses_train = loss
        

print("\nThe best hyperparameter: ", best_decay)
print("The number of epoch that yields the best validation performance: ", best_decay_epoch_best)
print("The validation performance (risk) in that epoch :",best_decay_val_performance[0])

_, _, risk_test= predict(X_test,w_test, y_test, False)  # testing
print("The test performance (risk) in that epoch", risk_test[0])

# Report numbers and draw plots as required.
epochs = [i for i in range(MaxIter)]
plt.figure()
plt.xlabel('epoch')
plt.ylabel('training loss')
X = np.linspace(0,MaxIter,MaxIter,endpoint=True).reshape([MaxIter,1])
plt.plot(X, losses_train, color = "blue")
plt.tight_layout()
plt.savefig('Training loss_2b.jpg')

plt.figure()
plt.xlabel('epoch')
plt.ylabel('risks')
X = np.linspace(0,MaxIter,MaxIter,endpoint=True).reshape([MaxIter,1])
plt.plot(X, risks_val, color = "red")
plt.tight_layout()
plt.savefig('risks_2b.jpg')



