#ifndef RL_ACCEL_REGS_H
#define RL_ACCEL_REGS_H

#include <stdint.h>

/* AXI4-Lite register offsets. */
#define RL_REG_CONTROL          0x000u
#define RL_REG_STATUS           0x004u
#define RL_REG_OPCODE           0x008u
#define RL_REG_SRC_BASE         0x00cu
#define RL_REG_DST_BASE         0x010u
#define RL_REG_AUX0_BASE        0x014u
#define RL_REG_AUX1_BASE        0x018u
#define RL_REG_WEIGHT_BASE      0x01cu
#define RL_REG_DIM_K            0x020u
#define RL_REG_DIM_N            0x024u
#define RL_REG_OUTPUT_SHIFT     0x028u
#define RL_REG_IRQ_ENABLE       0x02cu

#define RL_REG_VEC_ADDR         0x040u
#define RL_REG_VEC_STAGE0       0x044u
#define RL_REG_VEC_STAGE1       0x048u
#define RL_REG_VEC_STAGE2       0x04cu
#define RL_REG_VEC_STAGE3       0x050u
#define RL_REG_VEC_COMMAND      0x054u
#define RL_REG_VEC_READ0        0x058u
#define RL_REG_VEC_READ1        0x05cu
#define RL_REG_VEC_READ2        0x060u
#define RL_REG_VEC_READ3        0x064u

#define RL_REG_WEIGHT_ADDR      0x080u
#define RL_REG_WEIGHT_STAGE0    0x084u
#define RL_REG_WEIGHT_STAGE1    0x088u
#define RL_REG_WEIGHT_STAGE2    0x08cu
#define RL_REG_WEIGHT_STAGE3    0x090u
#define RL_REG_WEIGHT_COMMAND   0x094u
#define RL_REG_WEIGHT_READ0     0x098u
#define RL_REG_WEIGHT_READ1     0x09cu
#define RL_REG_WEIGHT_READ2     0x0a0u
#define RL_REG_WEIGHT_READ3     0x0a4u

#define RL_REG_INSTR_ADDR       0x0b0u
#define RL_REG_INSTR_STAGE0     0x0b4u
#define RL_REG_INSTR_STAGE1     0x0b8u
#define RL_REG_INSTR_STAGE2     0x0bcu
#define RL_REG_INSTR_STAGE3     0x0c0u
#define RL_REG_INSTR_COMMAND    0x0c4u
#define RL_REG_INSTR_READ0      0x0c8u
#define RL_REG_INSTR_READ1      0x0ccu
#define RL_REG_INSTR_READ2      0x0d0u
#define RL_REG_INSTR_READ3      0x0d4u
#define RL_REG_SEQUENCE_LENGTH  0x0d8u
#define RL_REG_SEQUENCE_CONTROL 0x0dcu
#define RL_REG_SEQUENCE_STATUS  0x0e0u
#define RL_REG_WEIGHT_DDR_BASE  0x0e4u
#define RL_REG_VERSION          0x0fcu

#define RL_CONTROL_START        (1u << 0)
#define RL_CONTROL_CLEAR_DONE   (1u << 1)
#define RL_STATUS_BUSY          (1u << 0)
#define RL_STATUS_DONE          (1u << 1)
#define RL_STATUS_ERROR         (1u << 2)
#define RL_CACHE_COMMIT         (1u << 0)
#define RL_CACHE_READ           (1u << 1)
#define RL_SEQUENCE_START       (1u << 0)
#define RL_SEQUENCE_BUSY        (1u << 0)
#define RL_SEQUENCE_ERROR       (1u << 1)
#define RL_SEQUENCE_INDEX_MASK  (0x1fu << 8)

#define RL_OP_GEMM              1u
#define RL_OP_ELU               2u
/* v3 retired NORM (unused by all 23 corpus policies and a strict subset of
   VECTOR.AFFINE) and reused opcode 3 for the VECTOR engine. */
#define RL_OP_VECTOR            3u
#define RL_OP_CONCAT            4u
#define RL_OP_SFU               5u

/* Descriptor word1[31:26] selects the sub-function. */
#define RL_VEC_AFFINE           0u
#define RL_VEC_MUL              1u
#define RL_VEC_CLIP             2u
#define RL_VEC_RSUM             3u
#define RL_VEC_RSUMSQ           4u
#define RL_SFU_ELU              0u
#define RL_SFU_TANH             1u
#define RL_SFU_SIGMOID          2u
#define RL_SFU_RSQRT            3u
#define RL_DESC1_SUBOP(v)       (((uint32_t)(v) & 0x3fu) << 26)

#define RL_VECTOR_WORD_BYTES    16u
#define RL_VECTOR_WORDS         512u
#define RL_WEIGHT_BANKS         8u
#define RL_WEIGHT_WORDS         1280u
#define RL_INSTRUCTION_WORDS    32u
#define RL_WEIGHT_ADDR(bank, word) \
    ((((uint32_t)(bank) & 7u) << 16) | ((uint32_t)(word) & 0x7ffu))

/* RoboAccel-v2: descriptor word3. word3 == 0 selects the v1 on-chip weight
 * cache, so a v1 program is unchanged. */
#define RL_DESC3_WEIGHT_STREAM  (1u << 0)
#define RL_DESC3_WEIGHT_WORD(w) (((uint32_t)(w) & 0x7fffffffu) << 1)
#define RL_ACCEL_VERSION_V2     0x00010003u

typedef struct {
    uint32_t opcode;
    uint32_t src_base;
    uint32_t dst_base;
    uint32_t aux0_base;
    uint32_t aux1_base;
    uint32_t weight_base;
    uint32_t dim_k;
    uint32_t dim_n;
    uint32_t output_shift;
} rl_accel_descriptor_t;

#endif
