import numpy as np

def _gen_orbit(num_pts_per_orbit:int,
               param:float):
    '''
    Function to generate an orbit
    Inputs:
        num_pts_per_orbit: numebr of points per orbit
        param: parameter for the orbit
    Output: 
        X: numpy.array of dimension (num_pts_per_orbit,2)
           represents num_pts_per_orbit points on the 2-d plane
    '''
    X = np.zeros([num_pts_per_orbit, 2])
    xcur, ycur = np.random.rand(), np.random.rand()
    for iPt in range(num_pts_per_orbit):
        xcur = (xcur + param * ycur * (1. - ycur)) % 1
        ycur = (ycur + param * xcur * (1. - xcur)) % 1
        X[iPt, :] = [xcur, ycur]
    return X

def generate_orbits(param_list:list, 
                    num_orbit_per_param:int, 
                    num_pts_per_orbit:int):
    '''
    Function to generate a number of orbits
    Inputs:
        param_list: list of parameters
        num_orbit_per_param: number of orbits generated per parameter
        num_pts_per_orbit: number of points in each orbit
    Output:
        X:numpy.array of dimension (num_orbit_per_param * len(param_list), num_pts_per_orbit, 2)
        The generated orbits
        X[i]------ the i-th orbit
        y:numpy.array of dimension num_orbits_per_param * len(param_list)
        The labels
        y[i] = r means X[i] corresponds to param r
    '''
    count = 0
    X = np.zeros((num_orbit_per_param * len(param_list), num_pts_per_orbit, 2))
    y = np.zeros(num_orbit_per_param * len(param_list), dtype = 'int')
    for lab, r in enumerate(param_list):
        print("Generating", num_orbit_per_param, "orbits and diagrams for r = ", r, "...")
        for orbit in range(num_orbit_per_param):
            X[count, :, :] = _gen_orbit(num_pts_per_orbit=num_pts_per_orbit,
                  param=r)
            y[count] = lab
            count += 1
    return (X, y)