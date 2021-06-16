//MIT License
//
//Copyright (c) 2021 Mobius Authors

//Permission is hereby granted, free of charge, to any person obtaining a copy
//of this software and associated documentation files (the "Software"), to deal
//in the Software without restriction, including without limitation the rights
//to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//copies of the Software, and to permit persons to whom the Software is
//furnished to do so, subject to the following conditions:

//The above copyright notice and this permission notice shall be included in all
//copies or substantial portions of the Software.

//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//SOFTWARE.


//from https://github.com/sunbelbd/mobius/blob/e2d166547d61d791da8f06747a63b9cd38f02c71/config.h

#pragma once


typedef float value_t; 
//typedef double dist_t; 
typedef float dist_t;
typedef size_t idx_t;
typedef int UINT;


#define ACC_BATCH_SIZE 4096
#define FIXED_DEGREE 31
#define FIXED_DEGREE_SHIFT 5


//for construction
#define SEARCH_DEGREE 15
#define CONSTRUCT_SEARCH_BUDGET 150
