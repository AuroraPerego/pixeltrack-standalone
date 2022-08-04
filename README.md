This version works only with compiling the GPU code.

Put this in Makefile on DPCT_CXXFLAGS:
-fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device xe_hp_sdv"

Without precompile flags and selecting CPU on chooseDevice.h, we get seg SEG fault.
