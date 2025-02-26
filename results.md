## project_1_run_1:

this boi was just the og project run (B=32)

## run_1:

this boi was just the og project run, however with small matrices for comparison of other runs:

```cpp
#define B   (4)      // Batch size
// #define N   (100)     // Sequence length
// #define dk  (128)     // Key/Query dimension
// #define dv  (128)     // Value dimension
#define N   (10)     // Sequence length
#define dk  (12)     // Key/Query dimension
#define dv  (12)     // Value dimension
```

## run_3

the first stage of optimization.

optimizations performed:

allocated BRAM upfront and copied all inputs to BRAM, stored result in BRAM and then copied out of BRAM

performed all batches at once:

https://github.com/RCMast3r/lab2_individual/blob/62fe53d2054624d7cb46c25f6c9ff16ac40b49e1/top.cpp 


TODO: optimize softmax_HLS function next as it seems to be taking a lot of cycles internally

## run_4

2nd stage of optimization. (FAIL)

optimizations performed:

- allocated BRAM upfront and copied all inputs to BRAM, stored result in BRAM and then copied out of BRAM

- attempted streaming out of softmax HLS function into output attention function -> failure due to attempting read before fifo has data (?)

## run_5

3rd stage of optimization. result: decrease from 3154 to 2825 cycles

optimizations performed:

- allocated BRAM upfront and copied all inputs to BRAM, stored result in BRAM and then copied out of BRAM

- performed all batches at once:

https://github.com/RCMast3r/lab2_individual/blob/62fe53d2054624d7cb46c25f6c9ff16ac40b49e1/top.cpp 

- unrolled batch outer loop to handle every stage on each batch at once

- TODO: optimize softmax_HLS function next as it seems to be taking a lot of cycles internally

## run_6

4th stage of optimization. result: (3830 to 3836) did not help 

optimizations performed:

- allocated BRAM upfront and copied all inputs to BRAM, stored result in BRAM and then copied out of BRAM

https://github.com/RCMast3r/lab2_individual/commit/b47e72492092ab71b743ebe82a79ae5470a209c8

- unrolled batch outer loop to handle every stage on each batch at once

- unrolling loops that are loading into BRAM as they seem to be taking a fair amount of cycles and we have access to more BRAM

- TODO: try task level parallelism

the issue i was having previously was due to attempting to read an empty stream, probably because it was attempting to perform an operation before it had the data to do so. 

## run_7

4th stage of optimization. result: (3830 to 3836) did not help 

optimizations performed:

- allocated BRAM upfront and copied all inputs to BRAM, stored result in BRAM and then copied out of BRAM

https://github.com/RCMast3r/lab2_individual/commit/b47e72492092ab71b743ebe82a79ae5470a209c8

- unrolled batch outer loop to handle every stage on each batch at once

- unrolling loops that are loading into BRAM as they seem to be taking a fair amount of cycles and we have access to more BRAM

- TODO: try task level parallelism

the issue i was having previously was due to attempting to read an empty stream, probably because it was attempting to perform an operation before it had the data to do so. 

## run_8 (in progress under ./project_1)

back to og, however with smaller batch size and full other sizes back