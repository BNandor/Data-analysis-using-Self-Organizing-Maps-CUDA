# Dimensionality reduction using Self-Organizing-Maps 
A self-organizing map (SOM) or self-organizing feature map (SOFM) is a type of artificial neural network (ANN) 
that is trained using unsupervised learning to produce a low-dimensional (typically two-dimensional), 
discretized representation of the input space of the training samples, called a map, and is therefore a method to do dimensionality reduction.

### Implementation
This repository contains a C++ implementation of self organizing maps. 
It is capable of the saving of the intermediate training states  thus allowing for better visualization. 
It is also capable of classification by calculating the kNN within classified maps. 
The learning rate is decreased along the slope of the normal distribution with the standard deviation of `max generation number / 3`  
![](media/8x8gif.gif) <img src="media/emnist28x28.gif"  width="180" height="180">

### Available flags  
`imagew, imageh` dimensions of the input data  
`mapw, maph` dimensions of the self organizing map  
`gen` training generation count  
`input, test, classCount` input and test filename, label count   
`outputPath` output file that will contain the resulting SOM  (`som.txt`)  

`animation, framecount` enable animation, frame save rate  
`animationPath` path to the directory in which to save the intermediary SOMs.  

### Usage examples  
`./a.out imagew=8 imageh=8 input=optdigits.tra mapw=10 maph=10 `  generates a 10x10 SOM in `som.txt`  
`./a.out imagew=8 imageh=8 input=optdigits.tra mapw=10 maph=10  animationPath=./ framecount=100` will save  100 intermediate SOMs to `.`  
`./a.out imagew=8 imageh=8 input=optdigits.tra test=optdigits.tes mapw=3 maph=3 classCount=10` for classification 
