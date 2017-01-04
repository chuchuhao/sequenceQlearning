'''
A TensorFlow implementation of Theano LSTM sentiment analyzer tutorial,
this model is a variation on TensorFlow's ptb_word_lm.py seq2seq model
tutorial to accomplish the sentiment analysis task from the IMDB dataset.

'''

'''
from __future__ import print_function
import six.moves.cPickle as pickle

from collections import OrderedDict
import sys
import time

'''

from collections import deque 
import numpy as np
import tensorflow as tf
import pickle
import random

import jsfeed
from game import flappyScript
import os
import sys

FINAL_EPSILON = 0.0001 
INITIAL_EPSILON = 0.1 # starting value of epsilon

class SentimentModel(object):
    def __init__(self, is_training, config):
        # init statistical variable
        self.TP = self.TN = self.FP = self.FN = 0 
        self.TPL = self.TNL = self.FPL = self.FNL = 0 
        self.win = 0
        self.total_num_step_played = 0
        self.num_action = 0
        # init replay memory
        self.replayMemory = deque()
        # init some parameters 
        self.timeStep = 0
        # self.lr = 1e-6
        self.lr = tf.Variable(0.0, trainable=False)
        self.rlr = tf.Variable(0.0, trainable=False)
        self.config = config 
        # self.actions = 4
        self.actions = 2

        self.epsilon = INITIAL_EPSILON

        self.observe = config.observe
        self.explore = config.explore

        # init Q network
        tf.Graph().as_default()
        
        self.stateInput, self.stateMask, self.W_fc1, self.b_fc1, self.W_fc2, self.b_fc2, self.QValue, \
            self.W_embedding, self.W_softmax, self.b_softmax, self.logits, self.pred, \
            self.final_rnn_state, self.cell, self.pt_initial_state = self.createQNetwork(rnn_scope="qScope")
        
        # init Target Q network
        self.stateInputT, self.stateMaskT, self.W_fc1T, self.b_fc1T, self.W_fc2T, self.b_fc2T, self.QValueT, \
            self.W_embeddingT, self.W_softmaxT, self.b_softmaxT, self.logitsT, self.predT, \
            self.final_rnn_stateT,self.cellT, self.pt_initial_stateT = self.createQNetwork(rnn_scope="qTScope")

        self.createTrainingMethod()
        self.createPreTrainMethod()

        '''
        self.copyTargetQNetworkOperation = [
            self.W_embeddingT.assign(self.W_embedding),
            self.W_softmaxT.assign(self.W_softmax),
            self.b_softmaxT.assign(self.b_softmax),
            self.W_fc1T.assign(self.W_fc1),
            self.b_fc1T.assign(self.b_fc1),
            self.W_fc2T.assign(self.W_fc2),
            self.b_fc2T.assign(self.b_fc2)]
        '''
        # saving and loading networks
        self.saver = tf.train.Saver()
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())
        checkpoint = tf.train.get_checkpoint_state("saved_networks")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.session. checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")
    
    def assign_lr(self, lr_value):
        # self.lr = lr_value
        self.session.run(tf.assign(self.lr, lr_value))
    def assign_rlr(self, lr_value):
        # self.lr = lr_value
        self.session.run(tf.assign(self.rlr, lr_value))
    
    def createQNetwork(self, rnn_scope, dropoutLayer=False):
        with tf.variable_scope(rnn_scope):
            # network weights
            cell = tf.nn.rnn_cell.GRUCell(self.config.hidden_size)
            if dropoutLayer and self.config.dropout < 1:
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.config.dropout)
            if self.config.num_layers > 1:
                cell = tf.nn.rnn_cell.MultiRNNCell([cell] * self.config.num_layers)
            
            with tf.device("/cpu:0"):
                W_embedding = self.weight_variable([self.config.num_tokens,self.config.hidden_size],name="W_embedding")

            W_softmax = self.weight_variable([self.config.hidden_size, self.config.num_tokens],name="W_softmax")
            b_softmax = self.bias_variable([self.config.num_tokens],name="b_softmax")

            W_fc1 = self.weight_variable([self.config.hidden_size,256],name="W_fc1")
            b_fc1 = self.bias_variable([256],name="b_fc1")

            W_fc2 = self.weight_variable([256,self.actions],name="W_fc2")
            b_fc2 = self.bias_variable([self.actions],name="b_fc2")

            # input layer, should be [num_steps, batch_size]
            stateInput = tf.placeholder(tf.int64,   [self.config.num_steps, self.config.batch_size], name="stateInput")
            stateMask  = tf.placeholder(tf.float32, [self.config.num_steps, self.config.batch_size], name="stateMask")
            stateMaskD = tf.expand_dims(stateMask, -1)
            '''
            tensorflow.python.framework.errors_impl.InvalidArgumentError: You must feed a value for placeholder tensor 'stateMask_1' with dtype float and shape [87,32]
            [[Node: stateMask_1 = Placeholder[dtype=DT_FLOAT, shape=[87,32], _device="/job:localhost/replica:0/task:0/cpu:0"]()]]
            '''
            # hidden layers
            with tf.device("/cpu:0"):
                embedding_input = tf.nn.embedding_lookup(W_embedding, stateInput)
                if dropoutLayer and self.config.dropout < 1:
                    embedding_input = tf.nn.dropout(embedding_input, self.config.dropout)
                
            bptt = []
            initial_rnn_state = cell.zero_state(self.config.batch_size, tf.float32)
            rnn_state = initial_rnn_state
            with tf.variable_scope("RNN"):
                for time_step in range(self.config.num_steps):
                    if time_step > 0:
                        tf.get_variable_scope().reuse_variables()
                    (cell_output, rnn_state) = cell(embedding_input[time_step,:,:], rnn_state)
                    bptt.append(tf.expand_dims(cell_output, 0))
            bptt = tf.concat(0,bptt)*stateMaskD
            stateMask_sum = tf.reduce_sum(stateMaskD, 0)
            proj = tf.reduce_sum(bptt, 0)/stateMask_sum
            
            # Q Value layer 
            # print(proj) # Tensor("truediv:0", shape=(32<batch>, 100<hidden_size>), dtype=float32)
            h_fc1 = tf.nn.relu(tf.matmul(proj, W_fc1) + b_fc1)
            Qvalue= tf.matmul(h_fc1,W_fc2) + b_fc2

            # Predict layer
            logits = tf.matmul(proj,W_softmax) + b_softmax
            # print(logits) # Tensor("add_2:0", shape=(32<batch>, 126), dtype=float32)
            pred   = tf.nn.softmax(logits)
            final_rnn_state = rnn_state
            
            return (stateInput, stateMask,\
                    W_fc1,b_fc1,W_fc2,b_fc2,Qvalue,   \
                    W_embedding,W_softmax, b_softmax, \
                    logits, pred, final_rnn_state, cell, initial_rnn_state)

    def createTrainingMethod(self):
        self.actionInput = tf.placeholder(tf.float32, [None,self.actions])
        self.yInput      = tf.placeholder(tf.float32, [None])
        Q_Action   = tf.reduce_mean( tf.mul(self.QValue, self.actionInput), reduction_indices = 1)
        self.Qcost = tf.reduce_mean( tf.square(self.yInput - Q_Action) )
        # tvarsQ = tf.trainable_variables()
        # self.trainQStep = tf.train.AdamOptimizer(self.lr).minimize(self.Qcost)
        # tvarsQScope = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="qScope")
        # tvarsQTScope = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="qTScope")
        # print([n.name for n in tvarsQScope])
        tvarsQ = [self.W_embedding, self.W_softmax, self.b_softmax, 
                  self.W_fc1, self.b_fc1, self.W_fc2, self.b_fc2]
        # gradsQ, _ = tf.clip_by_global_norm(tf.gradients(self.Qcost, tvarsQ), self.config.max_grad_norm)
        gradsQ = tf.gradients(self.Qcost, tvarsQ)
        optimizerQ = tf.train.AdamOptimizer(self.lr)
        self.trainQStep = optimizerQ.apply_gradients(zip(gradsQ, tvarsQ))

    def createPreTrainMethod(self):
        self.label = tf.placeholder(tf.int64, [self.config.batch_size])
        # Calculate accuracy
        correct_prediction = tf.equal(tf.argmax(self.pred,1), self.label)
        predi_bool = tf.cast(tf.argmax(self.pred,1), tf.bool)
        label_bool = tf.cast(self.label, tf.bool)
        TN_and = tf.logical_and(predi_bool, label_bool)
        tmp_or = tf.logical_or(predi_bool, label_bool)
        TP_not = tf.logical_not(tmp_or)

        self.pt_TP = tf.reduce_sum(tf.cast(TP_not, tf.float32))
        self.pt_TN = tf.reduce_sum(tf.cast(TN_and, tf.float32))
        self.pt_accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

        # Train
        loss   = tf.nn.sparse_softmax_cross_entropy_with_logits(self.logits, self.label)
        self.Rcost    = tf.reduce_mean(loss) / self.config.batch_size
        
        # no Clip Train :
        # self.trainRNN = tf.train.GradientDescentOptimizer(self.rlr).minimize(self.Rcost)
        # Clip Train    : 
        # tvars0 = tf.trainable_variables()
        tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="qScope")

        grads, _ = tf.clip_by_global_norm(tf.gradients(self.Rcost, tvars),self.config.max_grad_norm)
        #optimizer = tf.train.GradientDescentOptimizer(self.rlr)
        optimizer = tf.train.AdagradOptimizer(self.rlr)
        self.trainRNN = optimizer.apply_gradients(zip(grads, tvars))


    def pt_get_minibatches_idx(self, n, batch_size, shuffle=False):
        """
        Used to shjuffle the dataset at each iteration.
        """

        idx_list = np.arange(n, dtype="int32")

        if shuffle:
            np.random.shuffle(idx_list)

        minibatches = []
        minibatch_start = 0
        for i in range(n // batch_size):
            minibatches.append(idx_list[minibatch_start:
                                        minibatch_start + batch_size])
            minibatch_start += batch_size

        if (minibatch_start != n):
            # Make a minibatch out of what is left
            minibatches.append(idx_list[minibatch_start:])

        return minibatches
        
    def preTrain_run_epoch(self, data, eval_op, verbose=False):
        print("batch size", self.config.batch_size)
        self.pt_initial_state = tf.convert_to_tensor(self.pt_initial_state)
        state = self.pt_initial_state.eval()
        n_samples = data[0].shape[1]
        print("Testing %d samples:"%(n_samples))

        minibatches = self.pt_get_minibatches_idx(n_samples, self.config.batch_size, shuffle=True)
        n_batches = len(minibatches) - 1
        b_ind = 0
        correct = 0.
        total = 0

        tp_a = 0.
        tn_a = 0.
        for inds in minibatches[:-1]:
            print("\rbatch %d / %d"%(b_ind, n_batches), end="")
            sys.stdout.flush()

            x = data[0][:,inds]
            mask = data[1][:,inds]
            y = data[2][inds]
            cost, state, count, _ , tp, tn= self.session.run(\
                [self.Rcost, self.final_rnn_state, self.pt_accuracy, eval_op, self.pt_TP, self.pt_TN],\
                { self.stateInput: x, 
                  self.stateMask : mask, 
                  self.label: y, 
                  self.pt_initial_state: state})
            # print("...", count, ", ", tp, ", ", tn, ", ", y)
            tp_a += tp
            tn_a += tn
            correct += count
            total += len(inds)
            b_ind += 1

        print("\ntotal,",total,",correct,",correct, ",TP,",tp_a,",TN,", tn_a)
        accuracy = correct/total
        return accuracy


    
    def pt_keep_learning(self, X, y):
        train = jsfeed.prepare_data(X, y)        
        print("Train rnn with new Data")
        for i in range(self.config.re_max_epoch):
            lr_decay = self.config.lr_decay ** max(i - self.config.re_max_epoch, 0.0)
            self.assign_rlr(self.config.learning_rate * lr_decay )
            train_acc = self.preTrain_run_epoch(train, self.trainRNN)

    def weight_variable(self,shape, name):
        initial = tf.truncated_normal(shape, stddev = 0.01)
        return tf.Variable(initial, name)

    def bias_variable(self,shape, name):
        initial = tf.constant(0.01, shape = shape)
        return tf.Variable(initial, name)

    def copyTargetQNetwork(self):
        self.final_rnn_stateT = self.final_rnn_state
        # self.cellT = self.cell
        tvarsQcollect = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="qScope")
        # print([n.name for n in tvarsQcollect])
        tvarsQTcollect = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="qTScope")
        copyOperation = []
        # print([n.name for n in tvarsQTcollect])
        for i in range(len(tvarsQcollect)):
            copyOperation.append(tvarsQTcollect[i].assign(tvarsQcollect[i]))
        self.session.run(copyOperation)
        # print("Copyed .. ")
        
        # self.session.run(self.copyTargetQNetworkOperation)

    def trainQNetwork(self):
        # Step 1: obtain random minibatch from replay Memory
        minibatch = random.sample(self.replayMemory, self.config.batch_size)
        ## Structure of replay Memory
        # [0] - s_t   - currentState
        # [1] - a_t   - action taked
        # [2] - r_t+1 - reward after action taked
        # [3] - s_t_1 - next state after action took
        # [4] - end?  - whether the game is end
        # [5] - mask  - mask for [0] use 

        state_batch     = [data[0] for data in minibatch]
        action_batch    = [data[1] for data in minibatch]
        reward_batch    = [data[2] for data in minibatch]
        nextState_batch = [data[3] for data in minibatch]
        mask_batch      = [data[5] for data in minibatch]

        # Step 2: calculate y (target for iteration i)
        y_batch = []
        QValue_batch = self.QValueT.eval(feed_dict={
            self.stateInputT: np.transpose(state_batch),
            self.stateMaskT : np.transpose(mask_batch)
            })

        for i in range(0,self.config.batch_size):
            terminal = minibatch[i][4]
            if terminal:
                y_batch.append(reward_batch[i])
            else:
                y_batch.append(reward_batch[i] + self.config.gamma * np.max(QValue_batch[i]))

        self.trainQStep.run(feed_dict={
            self.yInput      : y_batch,
            self.actionInput : action_batch,
            self.stateInput: np.transpose(state_batch),
            self.stateMask : np.transpose(mask_batch)
            })

        # save network every 100000 iteration
        if self.timeStep % 500 == 0:
            print("....500")
            print(",TP :", self.TP, ",TPL:", self.TPL)
            print(",TN :", self.TN, ",TNL:", self.TNL)
            print(",FP :", self.FP, ",FPL:", self.FPL)
            print(",FN :", self.FN, ",FNL:", self.FNL)
            print(",STEP:", self.total_num_step_played)
            print(",num_action :", self.num_action)
            self.TP  = self.TN  = self.FP  = self.FN = 0 
            self.TPL = self.TNL = self.FPL = self.FNL = 0 
            self.num_action = 0
            self.total_num_step_played = 0
            
            if (self.TP + self.TN > self.win):
                print(win)
                self.win = self.TP + self.TN
                save_path = self.saver.save(self.session, os.getcwd()+'\\savedModel\\network-dqn-win.pickle', global_step = self.timeStep)
                print("Model saved in ", save_path)
        
        if self.timeStep % 2000 == 0:
            save_path = self.saver.save(self.session, os.getcwd()+'\\savedModel\\network-dqn.pickle', global_step = self.timeStep)
            print("Model saved in ", save_path)
                
        if self.timeStep % self.config.update_time == 0:
            self.copyTargetQNetwork()


    def setPerception(self,nextObservation,action,reward,terminal,nextMask):
        ''' Procedure Control 
        ''' 
        # TODO: 這邊的 newstate要改成上次的資訊加上新資訊, 還沒改之前我先用只有新資訊
        ##- newState = np.append(self.currentState[:,:,1:], nextObservation,axis=2)
        newState = nextObservation
        newMask  = nextMask
        # Put Experience to replayMemory 
        self.replayMemory.append(
            (self.currentState, action, reward, newState, terminal, self.currentMask)
        )

        # Memory Full
        if len(self.replayMemory) > self.config.num_replay_memory:
            self.replayMemory.popleft()
        
        # After Enough TimeStep, we Train (Batch)
        if self.timeStep > self.observe:
            self.trainQNetwork()

        state = ""
        if self.timeStep <= self.observe:
            state = "observe"
        elif (self.timeStep > self.observe) and (self.timeStep <= (self.observe + self.explore)):
            state = "explore"
        else:
            state = "train"
        
        print("Timestep: ", self.timeStep, ", state: ", state, ", epsilon: ", self.epsilon, end="\r")
        self.currentState = newState
        self.currentMask  = newMask
        self.timeStep += 1
        
    def getAction(self):
        ''' Get Some Action on a_t (current state)
        '''
        # TODO: Need Epsilon to Prevent some Work
        # stateInput : Input Layer, 
        # qvalue = self.qvalue.eval(feed_dict={self.stateInput:[self.currentState]})[0]
        
        
        stateToFeed = np.zeros((self.config.batch_size, self.config.num_steps), dtype=np.int32)
        stateToFeed[0] = self.currentState
        maskToFeed  = np.zeros((self.config.batch_size, self.config.num_steps))
        maskToFeed[0]  = self.currentMask      
        qvalue = self.QValue.eval(feed_dict={
            self.stateInput: np.transpose(stateToFeed),
            self.stateMask : np.transpose(maskToFeed)
            })[0] # 0 is cuz by batch 

        # print("getAction - qvalue : ", qvalue) # checked [xx,xx,xx,xx]
        # One hot Encoding Our Action, e.g. Positive = [1, 0, 0, 0]
        action = np.zeros(self.actions)
        action_index = 0
        if random.random() <= self.epsilon:
            # Explore more possible action 
            action_index = random.randrange(self.actions)
            action[action_index] = 1
        else:
            # Action determined by model
            action_index = np.argmax(qvalue)
            action[action_index] = 1    
        # change epsilon
        if self.epsilon > FINAL_EPSILON and self.timeStep > self.observe:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON)/self.explore

        return action

    def setInitState(self, observation, mask):
        self.currentState = observation
        self.currentMask  = mask 
    def getCurrentPredict(self):
        stateToFeed = np.zeros((self.config.batch_size, self.config.num_steps), dtype=np.int32)
        stateToFeed[0] = self.currentState
        maskToFeed  = np.zeros((self.config.batch_size, self.config.num_steps))
        maskToFeed[0]  = self.currentMask      
        predR = self.pred.eval(feed_dict={
            self.stateInput: np.transpose(stateToFeed),
            self.stateMask : np.transpose(maskToFeed)
        })[0]
        return np.argmax(predR)

class Config(object):
    num_replay_memory = 50000 # number of previous transitions to remember

    # observe = 100. # timesteps to observe before training
    # explore = 200000. # frames over which to anneal epsilon
    observe = 2000. # timesteps to observe before training
    explore = 30000. # frames over which to anneal epsilon
    gamma = 0.80 # decay rate of past observations

    num_tokens=126
    update_time = 50 # update predict model after time .. 
    batch_size=32  # The batch size during training.
    
    init_scale = 0.05
    learning_rate = 1.0
    max_grad_norm = 5
    
    num_steps = 87 # windows size
    hidden_size = 128 # neurons in one hidden sate
    num_layers = 2 # number of neuron

    max_epoch = 20
    # max_epoch = 1 # for test
    dropout = 0.5
    lr_decay = 0.95

    re_max_epoch = 10

def main(unused_args):
    # Step 0: init BrainDQN
    # actions = 2  # Make Decision, MoveForward 
    config = Config()
    config.num_steps = 200
    config.hidden_size = 256 #256
    config.num_layers  = 2 #2
    config.re_max_epoch  = 10
    config.max_epoch = 40
    Brain = SentimentModel(is_training=True, config=config)

    # Step 1: Pre-train RNN part
    print("Pre-Train The Model.  ")
    train, valid = jsfeed.load_data_light(pklPath="SAMPLE200.pickle", n_tokens=config.num_tokens, valid_portion=0.05)
    train = jsfeed.prepare_data(train[0], train[1], maxlen=config.num_steps)
    valid = jsfeed.prepare_data(valid[0], valid[1], maxlen=config.num_steps)
    print("Pre-Train Data ...")
    for data in [train, valid]:
        print(data[0].shape, data[1].shape, data[2].shape)
        good = np.sum(data[2])
        bad  = len(data[2])-good
        print("- Good: ", good, ", Bad:", bad, ", bad ratio:", (bad/(good+bad)))

    print("State the Pre-Train Process: ")
    for i in range(config.max_epoch):
        lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
        Brain.assign_rlr(config.learning_rate * lr_decay)
        print("Epoch: %d Learning rate: %.3f" % (i + 1, Brain.session.run(Brain.rlr)))
        train_acc = Brain.preTrain_run_epoch(train, Brain.trainRNN) 
        print("Training Accuracy = %.4f" % train_acc)
        valid_acc = Brain.preTrain_run_epoch(valid, tf.no_op())
        print("Valid Accuracy = %.4f\n" % valid_acc)

    print("End of the Pre-Train Process.")
    Brain.assign_lr(1e-6)
    Brain.copyTargetQNetwork()
    # Step 2: init Flappy Script Game (Init all Input)
    jsGame = flappyScript(jsdb="SAMPLE.pickle",windowsSize=config.num_steps, step=7)
    # jsGame = flappyScript(jsdb="sample_slice_450.pickle",windowsSize=config.num_steps, step=5   )
    
    # Step 3: play game
    # Step 3.1: obtain init state (From Data)
    observation0, mask0 = jsGame.time_stop_machine_A__A()

    # Step 3.2: get initial_state
    Brain.setInitState(observation0, mask0)

    # Step 3.3: run the game
    print("Let's Go Play")
    so_many_money_to_play_more_game = True
    Brain.TP = 0
    Brain.TN = 0
    Brain.FP = 0
    Brain.FN = 0 
    Brain.total_num_step_played = 0
    decision = 0

    newSampleToTrain_X = []
    newSampleToTrain_y = []
    while so_many_money_to_play_more_game:
        action = Brain.getAction()
        if (np.argmax(action) == 0):
            # Make Decision
            decision = Brain.getCurrentPredict()
        elif (np.argmax(action) == 1):
            # MoveForward
            decision = 2
        # print("Model decision: %d, Take action: %d " % (decision, np.argmax(action)))
        
        '''
        if (np.argmax(action) == 0 or np.argmax(action) == 1):
            # get origin predict ... 
            pred = Brain.getCurrentPredict()
            print("Model think: %d, Take action: %d " % (pred, np.argmax(action)))
        '''
        # print("First action .. ", action)

        nextObservation,reward,terminal,nextMask = jsGame.frame_step(decision)
        Brain.num_action += 1
        if terminal:
            if (decision == 0 or decision == 1):
                Brain.total_num_step_played += jsGame.last_num_step_played
                if (reward == 100):
                    # print("Correct")
                    if (decision == 0):
                        Brain.TP += 1
                        Brain.TPL += jsGame.TPL
                    elif (decision == 1):
                        Brain.TN += 1
                        Brain.TNL += jsGame.TNL
                elif (reward == -120):
                    # print("Wrong")
                    newSampleToTrain_X.append(Brain.currentState)
                    if (decision == 0):
                        Brain.FP += 1
                        Brain.FPL += jsGame.FPL
                        newSampleToTrain_y.append(1)
                    elif (decision == 1):
                        Brain.FN += 1
                        Brain.FNL += jsGame.FNL
                        newSampleToTrain_y.append(0)
                    else:
                        print("Throw some Error")
                    # print("Wrong, shold be:", reward, ", but guess :", np.argmax(action))
                    # Brain.wrong += 1
            else:
                print("weired")

        if nextObservation is None and nextMask is None:
            print("No more money to play more game")
            # All sample done ... 
            break
        # this must larger then batch_size
        if len(newSampleToTrain_X) > 200:
            Brain.pt_keep_learning(newSampleToTrain_X, newSampleToTrain_y)
            newSampleToTrain_X[:] = []
            newSampleToTrain_y[:] = []

        Brain.setPerception(nextObservation,action,reward,terminal,nextMask)

    print("....END")

if __name__ == '__main__':
    tf.app.run()


