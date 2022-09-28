## follow this steps to reproduce the error
File: ```src/sycl/plugin-PixelTriplets/gpuPixelDoubletsAlgos.h```

1) run the code as it is. It should work and print the correct value (30996).

2) uncomment line 81 and comment line 89. 
Change all the ```ntot``` to ```*ntot```, because now ```ntot``` is a pointer.
It should not work any more.

3) uncomment also lines from 98 to 101. It will not work anyway but now the value of ntot is different.

**_Note:_** after printing ```ntot``` the program will crash giving the error out of resources (related to a problem later in the code, so just interrupt it)