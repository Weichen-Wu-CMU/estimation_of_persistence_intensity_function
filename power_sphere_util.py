import numpy as np

def generate_uniform_sphere(num_samples:int,
                            num_pts_per_sample:int,
                            dimension: int,
                            turbulance_std: float):
    '''
    Function to generate samples from the unit sphere distribution
    Inputs:
    num_samples: number of samples
    num_pts_per_samples: number of points per sample
    dimension: dimension of the sphere; e.g. dimension=2 represents a circle
    turbulance_std: standard deviation of the turbulance
    Output:
    samples: numpy.array of dimension (num_samples,num_pts_per_sample,dimension)
    '''
    u = np.random.normal(0, 1, (num_samples,num_pts_per_sample,dimension))
    x = u / (np.sqrt(np.sum(u * u,axis = 2, keepdims = True)))
    eps = np.random.normal(0, turbulance_std, (num_samples,num_pts_per_sample,dimension))
    samples = x + eps
    return samples

def generate_power_sphere(num_samples:int,
                          num_pts_per_sample:int,
                          dimension:int, 
                          param: float,
                          turbulance_std:float):
    '''
    Function to generate samples from the power sphere distribution
    Inputs:
    num_samples: number of samples
    num_pts_per_samples: number of points per sample
    dimension: dimension of the sphere; e.g. dimension=2 represents a circle
    param: the parameter kappa
    turbulance_std: standard deviation of the turbulance
    Output:
    samples: numpy.array of dimension (num_samples,num_pts_per_sample,dimension)
    '''
    z = np.random.beta(a = (dimension-1)/2 + param, 
                       b = (dimension-1)/2, 
                       size = (num_samples,num_pts_per_sample,1))
    v = 2 * np.random.binomial(n = 1, 
                               p = 0.5, 
                               size = (num_samples,num_pts_per_sample,1)) - 1
    t = 2 * z - 1
    y = np.concatenate((t,np.sqrt(1 - t * t) * v), axis = 2)
    samples = y + np.random.normal(0, turbulance_std, (num_samples,num_pts_per_sample,dimension))
    return samples
