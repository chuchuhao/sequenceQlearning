import jsTokenEncoder
import numpy as np 
import random
class flappyScript(object):
    def __init__(self, jsdb=None, windowsSize=10, step=5, maxlen=None, minlen=50, n_tokens=126, drop_some_good=True, sort_by_len=True):
        self.TPL = 0.
        self.TNL = 0.
        self.FPL = 0.
        self.FNL = 0. 
        self.num_step_played = 0.
        self.last_num_step_played = 0.
        if jsdb is not None:
            js_set = jsTokenEncoder.loadSampleFromPickle(jsdb)
        else:
            js_set = jsTokenEncoder.loadSampleFromPickle()
        
        # Cut if set maxlen
        if maxlen is not None:
            new_set_x = []
            new_set_y = []
            for x,y in zip(js_set[0], js_set[1]):
                if len(x) < maxlen:
                    new_set_x.append(x)
                    new_set_y.append(y)
            js_set = (new_set_x, new_set_y)
            del new_set_x, new_set_y

        # Cut if set minLen
        if minlen is not None:
            new_set_x = []
            new_set_y = []
            for x,y in zip(js_set[0], js_set[1]):
                if len(x) > minlen:
                    new_set_x.append(x)
                    new_set_y.append(y)
            js_set = (new_set_x, new_set_y)
            del new_set_x, new_set_y
        
            
        # Drop Some Good if needed
        js_set_x, js_set_y = js_set
        def label_argsort(labels):
            return sorted(range(len(labels)), key=lambda x: labels[x])

        if drop_some_good:
            set_len = [len(s) for s in js_set_x]
            sorted_index = label_argsort(js_set_y)
            js_set_x = [js_set_x[i] for i in sorted_index]
            js_set_y = [js_set_y[i] for i in sorted_index]
            length_we_want = round(len(js_set_x) / 20)
            print(length_we_want)
            js_set_x = js_set_x[:length_we_want]
            js_set_y = js_set_y[:length_we_want]
        
        def len_argsort(seq):
            return sorted(range(len(seq)), key=lambda x: len(seq[x]))
        
        if sort_by_len:
            sorted_index = len_argsort(js_set_x)
            js_set_x = [js_set_x[i] for i in sorted_index]
            js_set_y = [js_set_y[i] for i in sorted_index]

        print("Loading Game ... total :", len(js_set_y))
        print("number of malicious are:", len(js_set_y) - np.sum(js_set_y))
        # property
        self.num_game = len(js_set_x)
        self.window_size = windowsSize
        self.js_set = (js_set_x, js_set_y)
        self.num_tokens = n_tokens 
        self.step = step

        # state
        self.cur_game = 0
        self.cur_length = len(self.js_set[0][self.cur_game])
        self.cur_index  = 0
        self.rate = 0
        self.all_done = False

    def next_game(self):
        ''' start next game
        '''
        # self.cur_game += 1
        # choose random number as cur_game
        self.cur_game = random.randrange(self.num_game)
        if self.cur_game >= self.num_game:
            print("ALL Game Done")
            self.all_done = True
            return True
        # Want Infinite Game ... 
        self.cur_length = len(self.js_set[0][self.cur_game])
        self.cur_index = 0
        self.rate = 0
        return False

    def time_stop_machine_A__A(self):
        ''' this only use when start a new game, which means no action
        '''
        if (self.cur_length <= self.window_size):
            # Halt (pad to window_size
            observation = np.zeros(self.window_size).astype('int64')
            mask        = np.zeros(self.window_size).astype('float32')
            for idx, t in enumerate(self.js_set[0][self.cur_game]):
                # TODO: DO I Need A MASK 
                observation[idx] = t
                mask[idx] = 1.
        else:
            observation = self.js_set[0][self.cur_game][0:self.window_size]
            mask = np.ones(self.window_size).astype('float32')
        return (observation, mask)

    def frame_step(self, action):
        ''' return (observation, reward, terminal)
        '''
        # note, good is 1, bad is 0
        # observation0, reward0, terminal (reward)
        reward   = 0
        terminal = False
        observation = []
        mask = []
        self.rate += 1
        self.num_step_played += 1
        if action == 0:   # Guess Malicious 
            if self.js_set[1][self.cur_game] == 0:
                # True Positive
                reward = 100
                self.TPL = len(self.js_set[0][self.cur_game])
            else:
                # False Positive
                reward = -120
                self.FPL = len(self.js_set[0][self.cur_game])
            terminal = True
            self.last_num_step_played = self.num_step_played
            self.num_step_played = 0
            if self.next_game():
                return (None, None, None, None)
            observation, mask = self.time_stop_machine_A__A()
        elif action == 1: # Guess Not Malicous
            if self.js_set[1][self.cur_game] == 1:
                # True Negative
                reward = 100
                self.TNL = len(self.js_set[0][self.cur_game])
            else:
                # False Negative
                reward = -120
                self.FNL = len(self.js_set[0][self.cur_game])
            terminal = True
            self.last_num_step_played = self.num_step_played
            self.num_step_played = 0
            if self.next_game():
                return (None, None, None, None)
            observation, mask = self.time_stop_machine_A__A()
        elif action == 2: # MoveForward
            reward = 10 + self.rate * (-1)
            terminal = False
            # check Length can move then move
            if (self.cur_length <= self.window_size):
                # Halt (pad to window_size
                observation = np.zeros(self.window_size).astype('int64')
                mask        = np.zeros(self.window_size).astype('float32')
                for idx, t in enumerate(self.js_set[0][self.cur_game]):
                    # TODO: DO I Need A MASK 
                    observation[idx] = t
                    mask[idx] = 1.
            else:
                if (self.cur_index+self.step+self.window_size) < self.cur_length:
                    # Moveforward
                    self.cur_index += self.step
                    observation = self.js_set[0][self.cur_game][self.cur_index:(self.cur_index+self.window_size)]
                else:
                    # Halt if to the end
                    self.cur_index = self.cur_length - self.window_size
                    observation = self.js_set[0][self.cur_game][-self.window_size:]
                mask = np.ones(self.window_size).astype('float32')
        elif action == 3: # MoveBackWard
            reward = self.rate * (-1)
            terminal = False
            # check Length can move then move
            if (self.cur_length <= self.window_size):
                # Halt (pad to window_size
                observation = np.zeros(self.window_size).astype('int64')
                mask        = np.zeros(self.window_size).astype('float32')
                for idx, t in enumerate(self.js_set[0][self.cur_game]):
                    # TODO: DO I Need A MASK 
                    observation[idx] = t
                    mask[idx] = 1.
            else:
                if ((self.cur_index-self.step) >= 0):
                    self.cur_index -= self.step
                    observation = self.js_set[0][self.cur_game][self.cur_index:(self.cur_index+self.window_size)]
                else:
                    self.cur_index = 0
                    observation = self.js_set[0][self.cur_game][0:self.window_size]
                mask = np.ones(self.window_size).astype('float32')
        else:
            # Invalid Action
            print("Invalid Action ...")
            return (None, None, None, None)            

        return (observation,reward,terminal,mask)

if __name__ == '__main__':
    game = flappyScript(jsdb="sample_slice_40to90.pickle",windowsSize=3)
    o,m = game.time_stop_machine_A__A()
    print("time_stop_machine_A__A")
    print(o,m)

    for i in range(0,30):
        o,r,t,m = game.frame_step(2)
    print("Do many Forward")
    print(o, r, t, m)
    print(game.cur_index)
    print(game.cur_length)
    o,r,t,m = game.frame_step(3)
    print("Do one BackWard")
    print(o, r, t, m)
    print(game.cur_index)
    print(game.cur_length)
    o,r,t,m = game.frame_step(0)
    print("Guess Malicious")
    print(o, r, t, m)
    print(game.cur_index)
    print(game.cur_length)