import numpy as np
import scipy.stats as st

def clear_bottom_right(matrix:np.array):
    '''
    Function to set the bottom-right part of a matrix to 0
    '''
    n = matrix.shape[0]
    for i in range(n):
        matrix[i:n,i] = 0
    return matrix

def estimate_persistence_intensity(dgm_flist:list,
                                   xmax:float,
                                   resolution = 100):
    '''
    Function to estimate persistence intensity from a list of persistence diagrams
    '''
    xx, yy = np.mgrid[0:xmax:complex(0,resolution),0:xmax:complex(0,resolution)]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    f = np.zeros((resolution,resolution))
    n = len(dgm_flist)
    for j in range(n):
        dgm = np.load(dgm_flist[j])
        if dgm.shape[0] <= 2:
            continue
        values = np.vstack([dgm[:,0],dgm[:,1]])
        kernel = st.gaussian_kde(values)
        f = f + (dgm.shape[0] * np.reshape(kernel(positions).T, xx.shape)-f)/(j+1)
    #f = np.log(f+1)
    f = clear_bottom_right(f)
    return f

def estimate_persistence_density(dgm_flist:list,
                                 xmax:float,
                                 resolution = 100):
    '''
    Function to estimate persistence density from a list of persistence diagrams
    '''
    xx, yy = np.mgrid[0:xmax:complex(0,resolution),0:xmax:complex(0,resolution)]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    f = np.zeros((resolution,resolution))
    n = len(dgm_flist)
    for j in range(n):
        dgm = np.load(dgm_flist[j])
        if dgm.shape[0] <= 2:
            continue
        values = np.vstack([dgm[:,0],dgm[:,1]])
        kernel = st.gaussian_kde(values)
        f = f + (np.reshape(kernel(positions).T, xx.shape)-f)/(j+1)
    f = clear_bottom_right(f)
    return f

def kernel_betti_number(dgm_flist:list,
                        xmax:float,
                        resolution = 100):
    '''
    Function to estimate the betti number using kernel method
    '''
    xx, yy = np.mgrid[0:xmax:complex(0,resolution),0:xmax:complex(0,resolution)]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    n = len(dgm_flist)
    ker_bettis = np.zeros((n,resolution))
    for j in range(n):
        dgm = np.load(dgm_flist[j])
        if dgm.shape[0] <= 2:
            continue
        values = np.vstack([dgm[:,0],dgm[:,1]])
        kernel = st.gaussian_kde(values)
        intensity = dgm.shape[0] * density
        for k in range(resolution):
            ker_bettis[j,k] = np.sum(intensity[0:(k+1),k:100])
            
    return ker_bettis

def plot_density(density, xmax, resolution = 100):
    xx, yy = np.mgrid[0:xmax:complex(0,resolution),0:xmax:complex(0,resolution)]
    plt.pcolormesh(xx,yy,density,cmap='Blues')
    plt.colorbar()
    x = np.arange(0, xmax, xmax/resolution)
    y = np.arange(0, xmax, xmax/resolution)
    plt.plot(x,y,'--',color = 'black')
