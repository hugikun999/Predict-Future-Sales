import mxnet as mx
from mxnet.gluon import nn

def conv_block(channel, kernel=1, stride=1, pad=0):
    blc = nn.HybridSequential()
    blc.add(
        nn.Conv1D(channel, kernel_size=kernel, strides=1, padding=pad),
        nn.BatchNorm(),
        nn.Activation('relu')
    )
    return blc

class CNN(nn.HybridBlock):
    def __init__(self):
        super(CNN, self).__init__()
        self.model = nn.HybridSequential()
        with self.model.name_scope():
            self.model.add(
                conv_block(64),
                nn.Flatten(),
                nn.Dense(10, activation='relu'),
                nn.Dense(1)
            )
    def hybrid_forward(self, F, x, *args, **kwargs):
        return self.model(x)

class LSTM(nn.HybridBlock):
    def __init__(self):
        super(LSTM, self).__init__()
        self.model = nn.HybridSequential()
        with self.model.name_scope():
            self.model.add(
                mx.gluon.rnn.LSTM(256),
                nn.Dropout(0.2),
                mx.gluon.rnn.LSTM(512),
                nn.Dropout(0.2),
                mx.gluon.rnn.LSTM(1024),
                nn.Dense(1)
            )
    def hybrid_forward(self, F, x, *args, **kwargs):
        return self.model(x)

if __name__=='__main__':
    fakein = mx.nd.ones((10, 38, 1))
    model = LSTM()
    model.initialize()
    fakeout = model(fakein)
    print(fakeout)