import jittor as jt
from jittor import Module
from jittor import nn
import numpy as np

import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model-dir', type=str, default='/opt/ml/model')  # os.environ['SM_MODEL_DIR']
    parser.add_argument('--output-dir', type=str, default='/opt/ml/output')  # os.environ['SM_MODEL_DIR']
    parser.add_argument('--train', type=str, default='/opt/ml/input/data/train')  # os.environ['SM_CHANNEL_TRAINING']
    parser.add_argument('--test', type=str, default='/opt/ml/input/data/test')  # os.environ['SM_CHANNEL_TEST']
    
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=50)
    parser.add_argument('--learning-rate', type=float, default=0.1)

    return parser.parse_args()

class Model(Module):
    def __init__(self):
        self.layer1 = nn.Linear(1, 10)
        self.relu = nn.Relu() 
        self.layer2 = nn.Linear(10, 1)
    def execute (self,x) :
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

def get_data(n): # generate random data for training test.
    for i in range(n):
        x = np.random.rand(batch_size, 1)
        y = x*x
        yield jt.float32(x), jt.float32(y)

args = parse_args()

epochs = args.epochs
learning_rate = args.learning_rate
batch_size = args.batch_size
n = 1000

model = Model()
optim = nn.SGD(model.parameters(), learning_rate)

for epoch in range(epochs):
    for i,(x,y) in enumerate(get_data(n)):
        pred_y = model(x)
        dy = pred_y - y
        loss = dy * dy
        loss_mean = loss.mean()
        optim.step(loss_mean)
        print(f"step {i}, loss = {loss_mean.data.sum()}")

if jt.rank == 0:
    model.save(os.path.join(args.model_dir, 'net.pkl'))