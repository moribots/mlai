# Machine Learning & Artificial Intelligence, ME 469
## Homework 1, Part A
### Maurice Rahme
### Student ID: 3219435

### A* Search Implementation, Naive and Online
### Assigned Data Set: ds1

****
## Dependencies
The code for this submission is written in Python 2.7 using PEP8 Styling. 
The libraries used for this code are:
* ```Numpy ```
* ```Matplotlib ```
* ```Pandas ```
* ```heapq``` for optimisation

**PLEASE MAKE SURE TO RUN THE FILE IN THE ```hw_1``` DIRECTORY ON YOUR TERMINAL**
****
## Coded Exercises
When you run ``` ./run.py ``` (after writing the command: ``` chmod +x run.py ```), you will be prompted to select an exercise: ```3```, ```5```, or ```7```. The numbering corresponds to the labeling in the homework description. You can also run the executable using ``` run run.py ```.

### Exercise 3:
#### Operation:
Enter ```3``` at the command prompt to plot for this exercise.

The Naive A* search is implemented here with a 1x1 grid. The path displayed here is often more optimal than that of the online search, as it (less realistically) allows the robot to back up and pick a better path if one is found. Plots can be printed for Start and Goal sets ```A```, ```B``` and ```C``` by inputting these values (upper or lower case) at the command prompt. Open and Closed list nodes are plotted to provide a visual of required computational power for each search. 

A Plot:

![3A](media/Homework/3A.png)

B Plot:

![3B](media/Homework/3B.png)

C Plot:

![3C](media/Homework/3C.png)

### Exercise 5:
#### Operation:
Enter ```5``` at the command prompt to plot for this exercise.

The Online A* search is implemented here with a 1x1 grid. The path displayed here is less optimal than that of the naive search, the realistic implementation prevents the robot from back-tracking by only adding the lowest-cost neighbour to the open list at each iteration. Plots can be printed for Start and Goal sets ```A```, ```B``` and ```C``` by inputting these values (upper or lower case) at the command prompt. Open and Closed list nodes are not plotted here as they simply consist of the followed path. 

A Plot:

![5A](media/Homework/5A.png)

B Plot:

![5B](media/Homework/5B.png)

C Plot:

![5C](media/Homework/5C.png)


### Exercise 7: 
#### Operation:
Enter ```7``` at the command prompt to plot for this exercise.

The Online A* search is implemented here with a 0.1x0.1 grid and obstacles inflated by .3m in each direction. The path displayed here is less optimal than that of the naive search, the realistic implementation prevents the robot from back-tracking by only adding the lowest-cost neighbour to the open list at each iteration. Plots can be printed for Start and Goal sets ```A```, ```B``` and ```C``` by inputting these values (upper or lower case) at the command prompt.

The same environment is navigated using the Naive A* for comparison purposes. At the command prompt, enter ```ONLINE``` or ```NAIVE``` (upper or lower case) to plot accordingly. As before, only the naive algorithm plots the expanded nodes.  

A Plot:

Online:
![7AO](media/PartA/7AOnline.png)
Naive:
![7AN](media/PartA/7ANaive.png)

B Plot:

Online:
![7BO](media/PartA/7BOnline.png)
Naive:
![7BN](media/PartA/7BNaive.png)

C Plot:

Online:
![7CO](media/PartA/7COnline.png)
Naive:
![7CN](media/PartA/7CNaive.png)

Note that in these screenshots, the path sometimes appear to intersect with an obstacle. If you zoom in after running the code, you will see that this is not actually the case. 