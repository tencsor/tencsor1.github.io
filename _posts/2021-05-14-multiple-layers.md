---
layout: post
title:  "How to add vary number of linear layers to Neural Network?"
author: "Ashish Singhal"
tags: PyTorch
comments: true
---

Recently I was implementing a library related to Graph Networks in PyTorch. There I encountered a requirement where the neural network model would have number of layers required as input. The user would specify a random number and the neural network model would need to add those many layers to the neural network. 

My first approach was to use a Python's `list` and store these linear layers in it which is wrong. We will see the correct method in this blog.

This blog talks about following things:
    1. How to add different numbers of linear layers aka feed forward neural network to the model?
    2. While adding different numbers of linear layers, shall we use a ```for``` loop and put them in a Python ```list```?

### Wrong Way
The wrong way to add variable numbers of linear layer is:


    class Net(nn.Module):
        def __init__(self, input_dim, output_dim, nos_linear_layer):
            super(Net, self).__init__
            self.nn_layers = []
            for i in range(0,nos_linear_layer):
                linear_layer = nn.Linear(input_dim, output_dim)
                self.nn_layers.append(linear_layer)

        def forward(self, input):
            outputs = None
            for i,layer in enumerate(self.nn_layers):
                outputs = layer(input)
        
            outputs = torch.nn.functional.Softmax(outputs, 1)
            return outputs


Generally above code would look correct and would be expected to run without any issue.nBut the mais issue with this code is that the linear layers stored in a simple Python `list` would not be trained.

On calling `model.parameters()`, PyTorch would simply ignore the parameters of linear layers stored in the Python `list`. Well, that's how PyTorch has been implemented. 

The solution is to use `nn.ModuleList` which is PyTorch's `list'.

### Correct Way
The correct code to add variable numbers of linear layers is:


    class Net(nn.Module):
        def __init__(self, input_dim, output_dim, nos_linear_layer):
            super(Net, self).__init__
            self.nn_layers = nn.ModuleList()
            for i in range(0,nos_linear_layer):
                linear_layer = nn.Linear(input_dim, output_dim)
                self.nn_layers.append(linear_layer)

        def forward(self, input):
            outputs = None
            for i,layer in enumerate(self.nn_layers):
                outputs = layer(input)
        
            outputs = torch.nn.functional.Softmax(outputs, 1)
            return outputs

