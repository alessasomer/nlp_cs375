from gensim.models.keyedvectors import KeyedVectors
import json
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from nltk.tokenize import TweetTokenizer
import pickle
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score

class DeepAveragingNetwork(nn.Module):
    """
    Pytorch implementation for Deep Averaging Network for classification 
    """
    def __init__(self, num_classes, #number of labels / y-values
                     pretrained_embedding_matrix, #we'll pass in embed_array_w_oov_pad
                     embedding_dim: int, #architecture/pre-processing decision 
                     hidden_dim1: int, #architecture decision
                     hidden_dim2: int, #architecture decision 
                     leaky_relu_negative_slope: float, #hyperparameter
                     dropout_probability: float #hyperparameter
                ):
        """
        Create the network architecture. 
        
        Hints: 
        - Make sure all your dimesions of various layers work out correctly 
        """
        super().__init__()
        self.num_classes = num_classes
        self.pretrained_embedding_matrix = pretrained_embedding_matrix
        self.embedding_dim = embedding_dim
        self.leaky_relu_negative_slope = leaky_relu_negative_slope
        self.dropout_probability = dropout_probability
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        #create a hidden linear layer (X_batch.shape[1] = number of embeddings)
        self.hidden1 = nn.Linear(embedding_dim, self.hidden_dim1)
        self.hidden2 = nn.Linear(self.hidden_dim1, self.hidden_dim2)
        self.theta = nn.Linear(self.hidden_dim2, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, X_batch: torch.Tensor) -> torch.Tensor:
        """
        Given X_batch, make the forward pass through the network. 
        
        The output should be the predicted *log probabilities*. 
        
        Returns: 
            - (torch.Tensor): the log probabilites after the forward pass 
                The shape of this tensor should be (X_batch.shape[0], 2)
                
        Hints: 
            - Look at Pytorch's implemenation of .mean()
            - There should be NO for-loops in this method 
        """ 
        vecs = torch.FloatTensor(self.pretrained_embedding_matrix)
        embed = nn.Embedding.from_pretrained(vecs, freeze =True)
        input_tensor = torch.LongTensor(X_batch)
        input_embedded = embed(input_tensor)
        #DROPOUT LAYER
        dropout = torch.nn.Dropout(self.dropout_probability, inplace=False)
        dropout_layer = dropout(input_embedded)
        #average across a row for x batch
        a = torch.mean(dropout_layer, 1)
        #pass "a" through hidden layer1
        hid1 = self.hidden1(a)
        
        #pass ouput of hidden layer through leakyrelu
        hid1 = nn.functional.leaky_relu(hid1, self.leaky_relu_negative_slope)
        
        #DO SECOND DROPOUT LAYER
        second_dropout_layer = dropout(hid1)
        hid2 = self.hidden2(second_dropout_layer)
        hid2 = nn.functional.leaky_relu(hid2, self.leaky_relu_negative_slope)
        #DO THIRD DROPOUT LAYER
        third_dropout_layer = dropout(hid2)
        out = self.theta(third_dropout_layer)
        log_probs = self.log_softmax(out)
        return log_probs
    
    def train_model(self, X_train, Y_train, X_dev, Y_dev, loss_fn, optimizer, num_iterations, batch_size = 500, check_every=10, verbose=False): 
        """
        Method to train the model. 
        
        No need to modify this method. 
        """
        self.train() # tells nn.Module its in training mode 
                      # (important when we get to things like dropout)
            
        loss_history = [] #We'll record the loss for inspection
        train_accuracy = []
        dev_accuracy = []
        precision = []
        recall = []
        f1 = []

        for t in range(num_iterations):
            if batch_size >= X_train.shape[0]: 
                X_batch = X_train
                Y_batch = Y_train
            else: #randomly choose batch_size number of examples 
                batch_indices = np.random.randint(X_train.shape[0], size=batch_size)
                X_batch = X_train[batch_indices]
                Y_batch = Y_train[batch_indices]

            #print("X_batch", X_batch.detach().numpy())
            # Forward pass 
            pred = self.forward(X_batch)
            loss = loss_fn(pred, Y_batch)

            #Backprop
            optimizer.zero_grad() # clears the gradients from the previous iteration
                                  # this step is important because otherwise Pytorch will 
                                  # *accumulate* gradients for all itereations (all backwards passes)
            loss.backward() # calculate gradients from forward step 
            optimizer.step() # gradient descent update equation 
            
            #Check the loss and train and dev accuracies every 
            if t % check_every == 0:
                loss_value = loss.item() # call .item() to detach from the tensor 
                loss_history.append(loss_value)
                
                #Check train accuracy (entire set, not just batch) 
                train_y_pred, _ = self.predict(X_train)
                train_acc = self.accuracy(train_y_pred, Y_train.detach().numpy()) 
                train_accuracy.append(train_acc)
                
                #Check dev accuracy (entire set, not just batch) 
                dev_y_pred, _ = self.predict(X_dev)
                dev_acc = self.accuracy(dev_y_pred, Y_dev.detach().numpy())
                dev_accuracy.append(dev_acc)
                dev_precision = self.precision(dev_y_pred, Y_dev.detach().numpy())

                precision.append(dev_precision)

                dev_recall = self.recall(dev_y_pred, Y_dev.detach().numpy())
                recall.append(dev_recall)

                dev_f1 = self.f1(dev_y_pred, Y_dev.detach().numpy())
                f1.append(dev_f1)
                if verbose: print(f"Iteration={t}, Loss={loss_value}")
                
        return loss_history, train_accuracy, dev_accuracy, precision, recall, f1
    
    def predict(self, X): 
        """
        Method to make predictions given a trained model. 
        
        No need to modify this method. 
        """
        self.eval() # tells nn.Module its NOT in training mode 
                 # (important when we get to things like dropout)
    
        pred_log_probs = self.forward(X)
        
        
        if self.num_classes == 2: 
            log_pred_pos_class = pred_log_probs[:,1].detach().numpy() #get only the positive class 
            pred_pos_class = np.exp(log_pred_pos_class) #exp to undo the log 
            # decision threshold
            y_pred = np.zeros(X.shape[0])
            y_pred[pred_pos_class>= 0.5] = 1
            return y_pred, pred_pos_class
        
        else: 
            return pred_log_probs
    
    @staticmethod
    def accuracy(y_pred: np.ndarray, y_true: np.ndarray) -> float: 
        """
        Calculates accuracy. No need to modify this method. 
        """
        return np.mean(y_pred == y_true)
    @staticmethod
    def precision(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Calculates accuracy. No need to modify this method.
        """
        return precision_score(y_true,y_pred)

    @staticmethod
    def recall(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Calculates accuracy. No need to modify this method.
        """
        return recall_score(y_true,y_pred)

    @staticmethod
    def f1(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Calculates accuracy. No need to modify this method.
        """
        return f1_score(y_true,y_pred)
    
def main():
    LEARNING_RATE = 1e-1
    HIDDEN_DIM1 = 200 
    HIDDEN_DIM2 = 100
    LEAKY_RELU_NEG_SLOPE = 0.01
    DROPOUT_PROB = 0.4 
    #get preprocessed data
    file = open('mypickle.pickle', 'rb')
    data = pickle.load(file)
    file.close()
    X_train = data[0]
    Y_train = torch.LongTensor(data[1])
    X_dev = data[2]
    Y_dev = torch.LongTensor(data[3])
    
    from preprocess import Twitter
    twitter = Twitter()
    embed_array = twitter.create_embeddings()
    #print(tweet_examples_test[0][0:5])
    from deepaverage import DeepAveragingNetwork
    #Grid Search

    LEARNING_RATE = 1e-1
    HIDDEN_DIM1 = 200
    HIDDEN_DIM2 = 100
    LEAKY_RELU_NEG_SLOPE = 0.01
    DROPOUT_PROB = 0.4

    opt_params=[]
    max_acc = 0.0

    for l_r in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
        for dropout_p in [0.0, 0.2, 0.4, 0.6, 0.8]:
            for slope in [0.001, 0.005, 0.01, 0.05, 0.1]:
                LEARNING_RATE = l_r
                HIDDEN_DIM1 = 200
                HIDDEN_DIM2 = 100
                LEAKY_RELU_NEG_SLOPE = slope
                DROPOUT_PROB = dropout_p
                model = DeepAveragingNetwork(2,embed_array,50, HIDDEN_DIM1, HIDDEN_DIM2, LEAKY_RELU_NEG_SLOPE, DROPOUT_PROB)
                loss_fn= nn.NLLLoss() #For binary logistic regression
                optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
                NUMBER_ITERATIONS = 300
                loss_history, train_accuracy, dev_accuracy, precision, recall, f1 = model.train_model(X_train, Y_train, X_dev, Y_dev, loss_fn, optimizer, NUMBER_ITERATIONS, batch_size = 200, check_every=50, verbose=False)
                if (dev_accuracy[1] > max_acc):
                    max_acc = dev_accuracy[1]
                    opt_params = [l_r, dropout_p, slope]

    print("Gridsearch max accuracy=", max_acc)
    print(opt_params)
    #OUT FOR GRID SEARCH model = DeepAveragingNetwork(2,embed_array,50, 20, 20, 0.01,0.1)
    #OUT FOR GRID SEARCH loss_fn= nn.NLLLoss() #For binary logistic regression 
    # OUT FOR GRID SEARCHoptimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    #Random inputs to check syntax
    #X_batch = torch.randint(low=0, high=len(vocab2indx), size=(100, 10))
    #print(X_batch.shape)
    #log_probs_out = model.forward(X_batch)
    #NUMBER_ITERATIONS = 300
    #loss_history, train_accuracy, dev_accuracy, precision, recall, f1 = model.train_model(X_train, Y_train, X_dev, Y_dev, 
    #                                                         loss_fn, optimizer, NUMBER_ITERATIONS, 
     #                                                        batch_size = 200,
     #                                                        check_every=50, verbose=False)
    print("loss history", loss_history)
    print("train accuracy", train_accuracy[-1])
    print("dev accuracy", dev_accuracy[-1])
    print("Final precision = ", precision)
    print("Final recall =", recall)
    print("Final f1 =", f1)
    #print("X_test before predict:", X_test)
    #test_predictions = model.predict(X_test)
    #predict_accuracy = model.accuracy(test_predictions, Y_test)
    #print("predict accuracy", predict_accuracy)


if __name__ == '__main__':
    main()