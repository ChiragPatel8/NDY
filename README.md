# NDY

NDY Machine learning project for M.Tech Software Systems Sem4.
==============================================================

This Project is meant for dissertation for M.Tech Software System, FS-20-21 of Birla Institute of Science and Technology.

![Alt text](data/logo.gif?raw=true)

# Installation

First, you will need to install [git](https://git-scm.com/), if you don't have it already.

Next, clone this repository by opening a terminal and typing the following commands:

    $ cd $HOME  # or any other development directory you prefer
    $ git clone https://github.com/ChiragPatel8/NDY.git
    $ cd NDY

## Python and dependent libraries
You will need Python. Python 3 is already readily available on many systems. You can check the version of python by typing the following command (you may need to replace `python3` with `python`):

    $ python3 --version  # for Python it should be 3

Any Python 3 version is fine. If you do not have Python 3, I recommend installing it.

To install python 3: on Windows or MacOSX, see [python.org](https://www.python.org/downloads/). On MacOSX, you can also use [MacPorts](https://www.macports.org/) or [Homebrew](https://brew.sh/). If the python version is 3.6 on MacOSX, you need to execute the following command to install the `certifi` package of certificates. (see this [StackOverflow question](https://stackoverflow.com/questions/27835619/urllib-and-ssl-certificate-verify-failed-error)):

    $ /Applications/Python\ 3.6/Install\ Certificates.command

On linux, you can use system's packaging system. For example, on Ubuntu:

    $ sudo apt-get update
    $ sudo apt-get install python3 python3-pip

## Using pip
There are several scientific Python libraries needed for this project which can be installed using Python's integrated packaging system, pip.

First check if you have the latest version of pip:

    $ python3 -m pip install --user --upgrade pip

Next, you need to create an isolated environment:

    $ python3 -m pip install --user --upgrade virtualenv
    $ python3 -m virtualenv -p `which python3` env

Now to activate this environment. Note: You need to run this command every time to use this environment.

    $ source ./env/bin/activate

On Windows:

    $ .\env\Scripts\activate

Then, use pip to install the required python packages.

    $ python3 -m pip install --upgrade -r depends/requirements.txt

## How to run.

Go to source directory

    $ cd source

run runner.py

    $python3 ./runner.py

There are already trained model in cache dir.
If you need to re-train the model, get the training data and uncomment a line in runner.py and re-run the same command, it will pre-process the data and train the model.
