# A Spiking Multi Layer Perceptron

The code in this repo allows you to recreate our experiments in our new paper [Deep Spiking Networks](http://arxiv.org/pdf/1602.08323v1.pdf).  In this paper, we introduce a new way to implement Deep Feed-Forward networks using spiking neurons - that is, neurons that only communicate by sending an "I'm active" signal to their downstream neighbours.  Our architecture requires only addition, indexing, and comparison (no multiplication) operations, and has the interesting property that the amount of computation required per data point is a function of the contents of the data.

## The Code

The Spiking MLP is written in Java (for speed), but we call it from Python (for convenience), using JPype (as a bridge).  This repo depends on [Plato](https://github.com/petered/plato), which is a library of useful ML/Deep Learning stuff, mainly built on top of Theano, and [DeepSpike](https://github.com/petered/DeepSpike) which is our fancy new Java Spiking Deep Network repo.  

## Setup

This installation process has been tested on MacOS.  There may be some hiccups if you do it from linux, and almost defnitely some hiccups if you try running it from windows, but all should be possible.  

### Step 1: Clone this repo, open up Terminal, and run setup.sh

```
cd /my/projects/folder/
git clone https://github.com/petered/spiking-mlp.git
cd spiking-mlp
source setup.sh
```
This will install a bunch of things, hopefully successfully.  Check the output to see if there were any installation errors.  If so, get Googling.

You'll notice a little `(venv)` on the left of your terminal prompt.  This means you're inside the "virtual environment" for this project, which means if you run Python you can import all the modules in the project. If you leave the venv, you can type `source venv/bin/activate` from the spiking-mlp directory to get back in.

### Step 2: Compile the Java.

There are a few ways to do this, here I list the one that I'm most familiar with, though there may be easier ways.

- Download [IntelliJ IDEA](https://www.jetbrains.com/idea/) (IntelliJ is a Java IDE, like Eclipse or Netbeans).  
- `File > Open...`, then navigate to `<your projects folder>/spiking-mlp/venv/src/deepspike/pom.xml` and click "Choose".  This will open the DeepSpike project.
- From the top toolbar, select `Build > Make Project`.  It should compile without error, and you should notice a new folder called "target" in your project root.  Good, you're done with this part.

### Step 3: Run Experiments

First verify that the experiments are working.  In terminal, from the `spiking-mlp` directory, while in the `(venv)` (see step 1), run:

```
py.test
```
If the tests pass, you're good to go.  To run the experiments, you can tinker with the paremeters at the bottom of `demo_spiking_mlp_experiments.py`, and run the experiments.

### Troubleshooting

- Can't locate Java?  For now we have a hardcoded path to Java.  If you're not on mac, you'll have to edit the file `spiking_mlp.jpype_connect.py`, function `_guess_java_loc` to include the new path.  

Please email me at peter \<dot\> ed \<dot\> oconnor at google's email service if you have any questions/comments/bug-reports.
