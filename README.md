# INF01017-K-means
A Python implementation of the K-means algorithm for the INF01017 class

## Requirements
 - Python 3+

## Installation

- Install [pandas](https://pandas.pydata.org/)
- Install [matplotlib](https://matplotlib.org/stable/index.html)
- Install [numpy](https://numpy.org/install/)
- Install [scikit-learn](https://scikit-learn.org/)

## Running the algorithm

To generate the model, use the *main.py* script:

```python
usage: main.py [-h]

Run the K-means algorithm.

positional arguments:
  h              The hypothesis to be used. The default is 1.
  k_clusters     The number of clusters used in algorithm. The default is 2.
  centroids_iterations     Number of iterations to find the best centroids initialization. The efault is 50.
  elbow_plot     If the execution should plot the Elbow method showing the optimal K. The default is False.
  
optional arguments:
  -h, --help  show this help message and exit
```