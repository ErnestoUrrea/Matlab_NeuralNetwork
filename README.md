# MATLAB Neural Network
A second (and more successful) attempt to build a Neural Network from scratch on MATLAB. To see the first attempt (not recommended) check the folder [`old_firstattempt`](/old_firstattempt) on this repository.

The objective of this project is learning about the basic work principles of Neural Networks and Machine Learning. This project was made using concepts from the Intermediate Mathematical Modeling (MA1029) class taken on March 2022 at the Tecnologico de Monterrey, in Monterrey, Mexico. Another great support resource for this project was the [Neural Network YouTube series](https://youtube.com/playlist?list=PL-Ogd76BhmcB9OjPucsnc2-piEE96jJDQ&si=unqUJHm6ifEYXdXz) by Dot CSV.

## File Description
pending...

## Development Log
In case I want to revisit this repository and don't fully understand my code or the concepts used, this is a log of my thought process during the development of this project.

To avoid the error made in my first attempt, of trying to start too big, and then getting lost in the complexity and lack of organization and documentation of my code, I decided to start small this time. My first objective was to understand the basic properties and structure of a Neural Network and develop a method of visualizing the results. This can be seen in the script [`Manually_Defined_Neural_Network_Visualization.m`](/Manually_Defined_Neural_Network_Visualization.m), where I define a 1 layer neural network to perform the classification of 4 sets of points in 2 groups.

After understanding neural networks and forward propagation, the next logical step into developing a system that is capable of learning is understanding backpropagation. To optimize a scalar function $f(x_1,x_2,...,x_n)$ to a local minimum we can use gradient descent, which consists in finding the direction on which the function increases more quickly (the gradient $\Delta f$) and move (change the inputs) in the oposite direction. In order to use this method to optimize a multiple-output function we need to find another function that evaluates this outputs and returns a scalar value that represents whatever we want to minimize, in this case "How bad is the neural network?". This function is called a Loss or a Cost function.

$$\frac{\partial C}{\partial W}$$

pending...

## Planned Changes / Features

pending...
