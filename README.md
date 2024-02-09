# CUDA-Kernels-In-Py-using-NUMBA
//////////////////////////////////////////
from numba import cuda

@cuda.jit
def add_kernel(x, y, out):

    idx = cuda.grid(1)          
    out[idx] = x[idx] + y[idx]
//////////////////////////////////////////
    
    import numpy as np

n = 4096
x = np.arange(n).astype(np.int32)
y = np.ones_like(x)               

d_x = cuda.to_device(x) 
d_y = cuda.to_device(y) 
d_out = cuda.device_array_like(d_x) 
threads_per_block = 128
blocks_per_grid = 32
/////////////////////////////////////////

add_kernel[blocks_per_grid, threads_per_block](d_x, d_y, d_out)
cuda.synchronize()
print(d_out.copy_to_host())

/////////////////////////////////////////

import numpy as np

n = 4096
x = np.arange(n).astype(np.int32) 
y = np.ones_like(x)              

d_x = cuda.to_device(x) 
d_y = cuda.to_device(y) 
d_out = cuda.device_array_like(d_x) 
threads_per_block = 32
blocks_per_grid = 32

/////////////////////////////////////////

import numpy as np

n = 4096
x = np.arange(n).astype(np.int32) 
y = np.ones_like(x)              

d_x = cuda.to_device(x) 
d_y = cuda.to_device(y) 
d_out = cuda.device_array_like(d_x) 
threads_per_block = 32
blocks_per_grid = 32

///////////////////////////////////////

add_kernel[blocks_per_grid, threads_per_block](d_x, d_y, d_out)
cuda.synchronize()
print(d_out.copy_to_host())
print(d_out.copy_to_host()[1023], d_out.copy_to_host()[1024]) 

//////////////////////////////////////

import numpy as np

n = 4096
x = np.arange(n).astype(np.int32) 
y = np.ones_like(x)  

d_x = cuda.to_device(x) 
d_y = cuda.to_device(y)
d_out = cuda.device_array_like(d_x) 
threads_per_block = 128
blocks_per_grid = 16

//////////////////////////////////////
