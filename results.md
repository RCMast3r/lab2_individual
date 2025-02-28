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

## run_9 
back to smol runs because I need progress damnit

optimization: 2525 (smoll est one yet)

optimized output attention dimensioning 1

softmax really taking up a lot of resources, will need to cut down the optimizations within it -> partial unrolling will be needed

at this stage I am relying on synthesis clock cycle estimations to guess at optimization levels
## run_10

optimization result: 2521

- attempted pipelining of jth loop for loading in Q K and V

## run_11

- attempting unroll of jth loop for Q K and V

optimization result: 2579 :skull:

new plan: instead of attempting to optimize reading all from DRAM at once, we read in batches instead

- each batch gets pipeline read in and the batch

## run_12

- attempting batch reading for each of the matrices from the tensors, reading in 1 batch at a time in a for loop

optimization result: 2831 however we do have a win here since we reduced resource utilization for LUTs and DSPs significantly

## run_13

- attempting more complete unroll of `scaled_dot_product` by adding a new inlined function for taking the sums vector and scaling it in an unrolled operation

- optimiation result: 3479 

but why? I dont even see it in the synthesis report. I believe that this eliminated some other automatic optimization operation that was being performed

I am going to allow for the softmax HLS internal loop to have an II of 5 in its pipeline and unroll the second matrix op that handles normalization

the LOAD_LOOP_OUTER is also still the highest cycle count, next is outer loop within `scaled_dot_product`. unrolling and annotating this one

loop at (line 106?) is the highest cycle count

## run_14

- attempting more unrolls within `scaled_dot_product` and adding more annotations

- optimization result: 3275 

i dont even think it was even full correct / synthesized. going to fix pipeline warning within softmax by removing pipeline of outer loop

## run_15
- removed some attempted pipelines

- optimization result: 3119

am going to attempt to move back into runs with full size, going to solve resource usage issues with loop at line 60 (exp dsp usages)

attempted to remove unroll, however it seems that hls inline attempts to force it?

forcing `#pragma HLS pipeline off`, automatic loop pipelining is kinda killing me
    - this is what forced this off

still having dsp scaling issues, need to turn off auto pipelining of outer loop (BIG_BATCH_OP).

TODO: increase bus data transfer to allow for bigger / smarter burst reads (will be done in run_17 since right now i want to fix scaling issues)

## run_16

fixing 128% DSP usage by turning off pipelining of inner loop of `scaled_dot_product`

also, i am now looking at the datasheet to find exactly how many resources I have:

dsp: 360 slices

    