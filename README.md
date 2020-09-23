# SGA-plant-learning

Modelling different phenomena such as the Baldwin effect and Lamarckian learning is useful in evolutionary algorithms 
to solve a wide array of optimization problems. Through this project, results regarding the effect of mutation rate 
and selection methods on efficiency in convergence in a small population of 50 individuals were confirmed. 
The interaction between evolution and learning has been studied from a molecular perspective2 as well as from a 
computational perspective to understand the processes, benefits, and costs involved. This interaction is relevant in 
the field of artificial intelligence as it improves our understanding of efficiency in adaptation in dynamic environments. 
Classical models implement genetic changes, however, behaviour such as learning can potentially influence fitness, survival, and 
thus, evolution. The aim of this project was to model the Baldwin effect in plants, as well as explore different plant-specific 
parameters of reproduction like self-reproduction and the lack of decision and classification problem for plants.

Evolutionary Algorithms (EAs) are population-based stochastic search algorithms developed from the principles of evolution 
as a tool for problem-solving, design, and optimization of those problems which cannot be solved by more traditional design and 
optimization tools. EAs model natural selection, by having specific steps (initialization, selection, genetic operators – crossover 
and mutation, termination) that closely parallel their biological counterpart. There exist four main EA types: 
1.	Evolutionary Strategies
2.	Evolutionary Programming 
3.	Genetic Algorithms
4.	Genetic Programming
They share similar basic design but differ in a few parameters like preferred selection strategies and relative importance of 
genetic operators. 

Overview of SGA: 

![sga-flowchart](https://github.com/aparnagopalakrishnan7/SGA-plant-learning/blob/master/sga-flowchart.png)

### Model details 

In my model, the influence of the following parameters was tested:
1.	Mutation rate:

    a.	Low (0.1%)
  
    b.	Moderate (1%)
    
    c.	High (5%)

2.	Parent population size percentage – number of individuals from previous generation used to produce offspring generation:

    a.	Low (10%) 

    b.	Moderate (20-50%) 
  
    c.	High (60-100%) 

3.	Tournament size percentage – number of individuals in a tournament such that “best” or most fit individual in this set of individuals chosen for crossover:
  
    a.	Low (10%) 
  
    b.	Moderate (20-50%) 
  
    c.	High (60-100%)
  
4. Phenotype length:
  
    a. 4 
  
    b. 8
  
    c. 16
  
  
### Results 
- Regardless of parent population size percentage and mutation rate, low tournament size percentage (10%) is not advisable 
as it provides very little to no advantage to the fitter individuals in the population. On the other hand, very high advantage to 
the fitter individuals, i.e., higher tournament size percentages (80-100%), as the probability of always choosing the best individual 
in the population is high, leading to quick loss in diversity of population and convergence to local optimum. 

- With the combination of higher parent population size and tournament size percentages (60-100%), quick convergence is due to the best 
individual having a higher probability of being chosen (seen by an initial sharp increase in average fitness) for crossover leading 
to quick loss in diversity in the population and convergence towards local optimum.

-	Moderate mutation rate, parent population, and tournament size percentages (20-40%) allow for slower convergence but avoid premature 
convergence toward the local optima.

- Extreme mutation rates are not suggested, as very low mutation rates allow for minimal exploration of the search space, and very high 
mutation rates result in inefficient convergence.

-	Combination of low parent population size percentage and tournament size percentage leads to no or inefficient convergence. 

- Low mutation rate (0.1%) leads to very quick convergence, convergence to local optimum, loss in diversity in population, and 
stagnation in the fitness of population. 

-	Combination of higher parent population size percentage and tournament size percentage provides more advantage to the fitter 
individuals, i.e., higher probability of being chosen for crossover, regardless of phenotype length. 

### Usage 
To run this algorithm, run `sga.py` with values for the customizable parameters: 
  1. Size of population `max_pop` (default: 50) 
  
  2. Parent and tournament population percentage `parent_percent`, `tournament_percent`
  
  3. Number of generations `max_gen` (default: 150) 
  
  4. Mutation rate 
  
  5. Phenotype length 
  
This algorithm also produces a plot at the end with convergence results (x axis: generation/iteration number, y axis: fitness).  
If you wish to reproduce your results, please set a suitable seed for the random number generators present throughout. 
  
