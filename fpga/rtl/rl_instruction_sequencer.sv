`timescale 1ns/1ps
`include "rl_accel_params.vh"

// Stores a programmable network as 128-bit operator descriptors and submits
// them to rl_accel_core without PS intervention between layers.
//
// Descriptor layout (little-endian 32-bit words):
//   word0: opcode[2:0], src[11:3], dst[20:12], aux0[29:21]
//   word1: aux1[8:0], weight[19:9], shift[25:20]
//   word2: dim_k[15:0], dim_n[31:16]
//   word3: reserved, written as zero
module rl_instruction_sequencer (
    input  wire                              clk,
    input  wire                              reset_n,

    input  wire                              host_wr_en,
    input  wire [`RL_INSTRUCTION_ADDR_W-1:0] host_addr,
    input  wire [127:0]                      host_wr_data,
    input  wire                              host_rd_en,
    output reg                               host_rd_valid,
    output reg  [127:0]                      host_rd_data,

    input  wire                              start,
    input  wire [5:0]                        instruction_count,
    output reg                               busy,
    output reg                               done,
    output reg                               error,
    output reg  [`RL_INSTRUCTION_ADDR_W-1:0] current_index,

    output reg                               cmd_start,
    output wire [2:0]                        cmd_opcode,
    output wire [`RL_VECTOR_ADDR_W-1:0]      cmd_src_base,
    output wire [`RL_VECTOR_ADDR_W-1:0]      cmd_dst_base,
    output wire [`RL_VECTOR_ADDR_W-1:0]      cmd_aux0_base,
    output wire [`RL_VECTOR_ADDR_W-1:0]      cmd_aux1_base,
    output wire [`RL_WEIGHT_ADDR_W-1:0]      cmd_weight_base,
    output wire [15:0]                       cmd_dim_k,
    output wire [15:0]                       cmd_dim_n,
    output wire [5:0]                        cmd_shift,
    output wire [5:0]                        cmd_subop,
    output wire                              cmd_weight_stream,
    output wire [`RL_WEIGHT_STREAM_WORD_W-1:0] cmd_weight_word,
    input  wire                              core_busy,
    input  wire                              core_done,
    input  wire                              core_error
);
    localparam [1:0] S_IDLE = 2'd0;
    localparam [1:0] S_FETCH = 2'd1;
    localparam [1:0] S_DISPATCH = 2'd2;
    localparam [1:0] S_WAIT = 2'd3;

    (* ram_style = "distributed" *) reg [127:0] instruction_mem
        [0:`RL_INSTRUCTION_DEPTH-1];
    reg [1:0] state;
    reg [5:0] count_latched;
    reg [127:0] current_descriptor;

    assign cmd_opcode = current_descriptor[2:0];
    assign cmd_src_base = current_descriptor[11:3];
    assign cmd_dst_base = current_descriptor[20:12];
    assign cmd_aux0_base = current_descriptor[29:21];
    assign cmd_aux1_base = current_descriptor[40:32];
    assign cmd_weight_base = current_descriptor[51:41];
    assign cmd_shift = current_descriptor[57:52];
    assign cmd_dim_k = current_descriptor[79:64];
    assign cmd_dim_n = current_descriptor[95:80];
    // word3 was reserved-zero in v1, so a v1 program decodes as stream=0.
    // word1[31:26] was reserved zero in v1/v2; v3 uses it as a sub-function
    // selector, so an older program decodes as sub-op 0.
    assign cmd_subop = current_descriptor[63:58];
    assign cmd_weight_stream = current_descriptor[96];
    assign cmd_weight_word = current_descriptor[127:97];

    wire opcode_valid = (cmd_opcode == `RL_OP_GEMM) ||
                        (cmd_opcode == `RL_OP_ELU) ||
                        (cmd_opcode == `RL_OP_VECTOR) ||
                        (cmd_opcode == `RL_OP_CONCAT) ||
                        (cmd_opcode == `RL_OP_SFU);
    wire dimensions_valid =
        ((cmd_opcode == `RL_OP_GEMM) && (cmd_dim_k != 0) && (cmd_dim_n != 0)) ||
        (((cmd_opcode == `RL_OP_ELU) || (cmd_opcode == `RL_OP_VECTOR) ||
          (cmd_opcode == `RL_OP_SFU)) && (cmd_dim_n != 0)) ||
        ((cmd_opcode == `RL_OP_CONCAT) &&
            ((cmd_dim_k != 0) || (cmd_dim_n != 0)));
    wire descriptor_valid = opcode_valid && dimensions_valid;

    always @(posedge clk) begin
        host_rd_valid <= host_rd_en;
        if (host_wr_en)
            instruction_mem[host_addr] <= host_wr_data;
        if (host_rd_en)
            host_rd_data <= instruction_mem[host_addr];

        if (!reset_n) begin
            state <= S_IDLE;
            count_latched <= 0;
            current_descriptor <= 0;
            current_index <= 0;
            busy <= 1'b0;
            done <= 1'b0;
            error <= 1'b0;
            cmd_start <= 1'b0;
            host_rd_valid <= 1'b0;
            host_rd_data <= 0;
        end else begin
            done <= 1'b0;
            cmd_start <= 1'b0;
            case (state)
                S_IDLE: begin
                    busy <= 1'b0;
                    if (start) begin
                        error <= 1'b0;
                        current_index <= 0;
                        if ((instruction_count == 0) ||
                            (instruction_count > `RL_INSTRUCTION_DEPTH)) begin
                            error <= 1'b1;
                            done <= 1'b1;
                        end else begin
                            count_latched <= instruction_count;
                            busy <= 1'b1;
                            state <= S_FETCH;
                        end
                    end
                end
                S_FETCH: begin
                    current_descriptor <= instruction_mem[current_index];
                    state <= S_DISPATCH;
                end
                S_DISPATCH: begin
                    if (!descriptor_valid) begin
                        busy <= 1'b0;
                        done <= 1'b1;
                        error <= 1'b1;
                        state <= S_IDLE;
                    end else if (!core_busy) begin
                        cmd_start <= 1'b1;
                        state <= S_WAIT;
                    end
                end
                S_WAIT: begin
                    if (core_done) begin
                        if (core_error) begin
                            busy <= 1'b0;
                            done <= 1'b1;
                            error <= 1'b1;
                            state <= S_IDLE;
                        end else if (({1'b0, current_index} + 6'd1) >=
                                     count_latched) begin
                            busy <= 1'b0;
                            done <= 1'b1;
                            error <= 1'b0;
                            state <= S_IDLE;
                        end else begin
                            current_index <= current_index + 1'b1;
                            state <= S_FETCH;
                        end
                    end
                end
                default: begin
                    busy <= 1'b0;
                    done <= 1'b1;
                    error <= 1'b1;
                    state <= S_IDLE;
                end
            endcase
        end
    end
endmodule
