#define DEBUG_x86
#define DEBUG_PRINT

#ifdef DEBUG_x86
#include "tensorflow/lite/micro/examples/ml_on_risc/main_functions_local.h"
#include <thread>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <cmath>
#include <sstream>
#include <vector>
#include <stdint.h>
#else
#include "tensorflow/lite/micro/examples/ml_on_risc/main_functions.h"
#endif

#include <cstdint>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stddef.h>       // per NULL

#ifdef DEBUG_PRINT
#define DPRINTF(...) printf(__VA_ARGS__)
#else
#define DPRINTF(...) // niente
#endif

#define INPUT_FEATURE_SIZE 1024
#define OUTPUT_FEATURE_SIZE 1024