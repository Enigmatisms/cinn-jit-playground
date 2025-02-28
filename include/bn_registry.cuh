#pragma once

// initialize instantiated kernel table
void init_kernel_table();

// call compiled kernel given the input shape
void lookup_kernel_call(int n, int w, int c);
