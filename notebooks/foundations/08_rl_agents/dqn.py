class DQN(tf.keras.Model):
    def __init__(self, states, actions):
        super(DQN, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(states,))
        self.hidden_layer = tf.keras.layers.Dense(128)
        self.output_layer = tf.keras.layers.Dense(actions)

    @tf.function
    def call(self, inputs):
        z = self.input_layer(inputs)
        z = self.hidden_layer(z)
        output = self.output_layer(z)
        return output

class Transition(object):
    
    def __init__(self, state, action, reward, next_state):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
    
class DataBuffer(object):

    def __init__(self, length):
        self.buffer = deque([], maxlen=length)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
