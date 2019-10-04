# Machine Learning & Artificial Intelligence, ME 469
## Homework 0, Part A
### Maurice Rahme
### Student ID: 3219435

### Assigned Filter: Particle Filter
### Assigned Data Set: ds0

****
## Dependencies
The code for this submission is written in Python 2.7 using PEP8 Styling. 
The libraries used for this code are:
* ``` Numpy ```
* ``` Matplotlib ```
* ``` Pandas ``` (mentioned in my email)

**PLEASE MAKE SURE TO RUN THE FILE IN THE ```hw_0``` DIRECTORY ON YOUR TERMINAL**
****
## Coded Exercises
When you run ``` ./part_a.py ``` (after writing the command: ``` chmod +x part_a.py ```), you will be prompted to select an exercise: 2, 3, or 6. The numbering corresponds to the labeling in the homework description. You can also run the executable using ``` run part_a.py ```.
### Exercise 2:
#### Noise:
If exercise 2 is selected, you will be asked whether or not to add noise to the motion model. This noise matrix generated is a gaussian based on the (arbitrary) standard deviations for x, y and theta: 5mm, 5mm and 0.012rad. These were chosen based on the Viscon's standard deviation of +/- 3mm for cartesian values. I took standard deviations slightly larger than that of the Vison. The standard deviation used for theta was also arbitrary, and was chosen experimentally based on behaviour.
#### Motion Model:
The motion model used in this exercise (and all exercises in this submission) is a nonlinear model mapping $$x_t$$, $$y_t$$, and $$\theta_t$$ as follows for each next iteration of time elapsed $$dt$$:
* $$x_t=x_{t-1}+v(cos(x))dt+\epsilon_x$$
* $$y_t=y_{t-1}+v(sin(x))dt+\epsilon_y$$
* $$\theta_t=\theta_{t-1}+\omega dt+\epsilon_{\theta}$$

where $$v$$ is linear velocity, $$\omega$$ is angular velocity, and $$\epsilon$$ is noise. 

#### Operation:
Funtion ```a2()```
After selecting your response for whether or not to include noise in the model, the ``` Robot ``` Class, which stores the ``` position ``` vector attribute of the robot, performs its ``` move ``` method iteratively for each command as given in the assignment by applying the motion model.

The plot is animated to show the subtle change in behaviour in case noise is implemented; you will notice that the noise-inclusive plot may change directions thanks to the added noise on $$\theta$$. The starting point is indicated by a black dot, the intermediate point by black diamonds, and the end point by a purple dot. 

![An example output for exercise 2]('ex2.png')

### Exercise 3:
#### Noise & Motion Models:
The noise and motion models used here are identical to the ones used in exercise 2. Note that in the ``` Robot ``` Class, $$dt$$ is defined as ``` future timestamp ``` - ``` current timestamp ```. However, to ensure that the same class can be used for exercises 2 and 3, the absolute value of this operation is taken instead. 
#### Operation:
Function ```a3()```
Upon selecting this exercise, you will also be asked whether or not noise should be added to the model. Subsequently, you will be asked whether to plot the full Dead Reckoning versus Ground Truth paths, or just the first 2000 iterations of the plot, where I have identified the start of the major divergence between the two. This functionality is achieved using list comprehension, by storing the first 2000 elements of the lists for the complete paths and plotting them if requested. 

The ``` Robot ``` Class is reused in this exercise, however, the ```controls``` list is fed data from the ```ds0_Odometry.dat``` file, which is read using ```Pandas``` and stored as a list of 64-bit floats using the function ```read_dat()``` which takes for arguments the index at which to start storing useful data (since the .dat file begins with headers), the file path, and the name of the columns. The ```read_dat()``` function is called in ```main()``` to read all the relevant data for this submission, including ```ds0_Groundtruth.dat```, and ```ds0_Landmark_Groundtruth.dat```. The former of these is used to be plotted alongside the dead reckoning data. 

In the resultant plot, the starting point for both the Dead Reckoning and Ground Truth states is shown as a yellow dot. The Dead Reckoning path is plotted in black, and the Ground Truth path in green. Both paths end with a purple marker in their 2000-iteration and full path plots. 

![An example output for exercise 3 with limited iterations to show divergence]('ex3_lim.png')
![An example output for exercise 3 with all iterations]('ex3.png')

### Exercise 6:
### Measurement Model:
The measurement model maps the acquired range and bearing to the cartersian position of the measured object (the landmark) relative to the robot, and vice versa. I will call the former model the *forward measurement model*, and the latter the *inverse measurement model*. 

The inverse measurement model:
* $$range=\sqrt{(x_t-xi-\epsilon_{xi})^2+(y_t-yi-\epsilon_{yi})^2}$$
* $$bearing=\arctan(\frac{y_i+\epsilon_{yi}-y_t}{x_i+\epsilon_{xi}-x_t})$$

where the $$i$$ subscripts denote the landmark coordinates.

The forward measurement model:
* $$x_i=x_t+range(\cos(bearing+heading))$$
* $$y_i=y_t+range(\sin(bearing+heading))$$

where heading is the $$\theta_t$$ for the robot, which is ```0``` for all evaluations in this exercise. Note that noise is only added in the inverse measurement model, as it is derived from the estimated cartersian coordinates according to the camera. 
### Operation:
Function ```a6()```
The ```Robot``` Class has an additional method, ```measure_a6()```, which computes the inverse model with (optional) measurement noise characterised by the standard deviations in $$x$$ and $$y$$ provided for landmarks 6, 13, and 17, followed by the forward model to quantify the error due to noise. For the error output, the resultant absolute cartesian landmark location is compared to the Ground Truth data. If the noise option is not selected, the executable will output ```0``` in some cases, but may also output a number with a factor of ```E-16``` due to the rounding associated with the ```Numpy``` trigonometric operations in the forward and reverse models. This function also outputs the range and bearing of each measurement, as requested.

The pose data used here as well as the chosen landmarks are provided in the homework assignment. 

![An example output for exercise 6 returning measurement error]('ex6_noise.png')

## Final Note:
There are various screenshots of code output in the hw_0 file with relevant names in case the code malfunctions. 