`ifndef RL_ACCEL_PARAMS_VH
`define RL_ACCEL_PARAMS_VH

// Datapath format used by the first implementation.
`define RL_DATA_W          16
`define RL_ACC_W           48
`define RL_LANES           8
`define RL_VEC_W           128
`define RL_WEIGHT_BUS_W    1024
`define RL_ACT_FRAC        8
`define RL_RAW_FRAC        6
`define RL_WEIGHT_FRAC     14

// Large enough for the current 620-tile policy and compatible policies.
`define RL_WEIGHT_DEPTH    1280
`define RL_WEIGHT_ADDR_W   11
`define RL_VECTOR_DEPTH    512
`define RL_VECTOR_ADDR_W   9
`define RL_INSTRUCTION_DEPTH  32
`define RL_INSTRUCTION_ADDR_W 5

// RoboAccel-v2: descriptor word3 carries the DDR weight stream controls.
`define RL_WEIGHT_STREAM_WORD_W 31

`define RL_OP_GEMM         3'd1
`define RL_OP_ELU          3'd2
// v3: opcode 3 was NORM, which no policy in the 23-model corpus used and whose
// per-feature affine is a strict subset of VECTOR's AFFINE sub-op. The vector
// engine reuses the retired NORM engine's eight DSP48E1.
`define RL_OP_VECTOR       3'd3
`define RL_OP_CONCAT       3'd4
`define RL_OP_SFU          3'd5

// VECTOR sub-ops and SFU curves, both carried in descriptor word1[31:26].
`define RL_VEC_AFFINE      3'd0
`define RL_VEC_MUL         3'd1
`define RL_VEC_CLIP        3'd2
`define RL_VEC_RSUM        3'd3
`define RL_VEC_RSUMSQ      3'd4
`define RL_SFU_ELU         2'd0
`define RL_SFU_TANH        2'd1
`define RL_SFU_SIGMOID     2'd2
`define RL_SFU_RSQRT       2'd3

`endif
