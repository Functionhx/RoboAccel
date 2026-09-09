`timescale 1ns/1ps
`include "rl_accel_params.vh"

// RoboAccel-v3 vector engine.
//
// Replaces rl_norm_engine, whose per-feature affine is a strict subset of what
// this does and which no policy in the 23-model corpus ever used.
//
//   AFFINE   dst = sat((src * a + b) >> shift)      Sub, Div-by-constant, Mul, Add
//   MUL      dst = sat((src * z) >> shift)          LSTM gating, elementwise Mul
//   CLIP     dst = clamp(src, lo, hi)               Clip, Relu
//   RSUM     dst[0] = sat(sum(src) >> shift)        ReduceSum
//   RSUMSQ   dst[0] = sat(sum(src*src) >> shift)    ReduceL2, with SFU rsqrt
//
// Lanes are processed one per cycle rather than eight at once. A first
// lane-parallel version cost 4,470 LUT and 15 DSP and pushed the device to 93%
// LUT, where placement failed. These primitives are O(N) over short vectors --
// tens of elements against the GEMM's tens of thousands of MACs -- so the
// parallel datapath bought nothing that mattered and the serial one costs a
// single multiplier and a single requantizer.
module rl_vector_engine #(
    parameter integer VEC_ADDR_W = 9
) (
    input  wire                    clk,
    input  wire                    reset_n,
    input  wire                    start,
    input  wire [2:0]              subop,
    input  wire [VEC_ADDR_W-1:0]   src_base,
    input  wire [VEC_ADDR_W-1:0]   dst_base,
    input  wire [VEC_ADDR_W-1:0]   aux0_base,
    input  wire [VEC_ADDR_W-1:0]   aux1_base,
    input  wire [15:0]             element_count,
    input  wire [5:0]              shift,
    output reg                     busy,
    output reg                     done,

    output reg                     vec_en,
    output reg                     vec_we,
    output reg [VEC_ADDR_W-1:0]    vec_addr,
    output reg [127:0]             vec_wr_data,
    input  wire                    vec_rd_valid,
    input  wire [127:0]            vec_rd_data
);
    localparam [2:0] OP_AFFINE = 3'd0;
    localparam [2:0] OP_MUL    = 3'd1;
    localparam [2:0] OP_CLIP   = 3'd2;
    localparam [2:0] OP_RSUM   = 3'd3;
    localparam [2:0] OP_RSUMSQ = 3'd4;

    localparam [3:0] S_IDLE      = 4'd0;
    localparam [3:0] S_SRC_REQ   = 4'd1;
    localparam [3:0] S_SRC_WAIT  = 4'd2;
    localparam [3:0] S_AUX0_REQ  = 4'd3;
    localparam [3:0] S_AUX0_WAIT = 4'd4;
    localparam [3:0] S_AUX1_REQ  = 4'd5;
    localparam [3:0] S_AUX1_WAIT = 4'd6;
    // The lane datapath is pipelined for the same reason the SFU is: a single
    // cycle doing lane mux -> multiply -> variable shift -> bias -> saturate
    // closed at -13.2 ns.
    localparam [3:0] S_SEL       = 4'd7;
    localparam [3:0] S_MUL       = 4'd8;
    localparam [3:0] S_SHIFT     = 4'd9;
    localparam [3:0] S_RES       = 4'd10;
    localparam [3:0] S_OUT       = 4'd11;
    localparam [3:0] S_RFIN      = 4'd12;
    localparam [3:0] S_RFIN2     = 4'd13;
    localparam [3:0] S_WRITE     = 4'd14;

    reg [3:0]  state;
    reg [2:0]  active_op;
    reg [5:0]  active_shift;
    reg [15:0] group_count, group_index, total_elements;
    reg [2:0]  lane;
    reg [127:0] src_word, aux0_word, aux1_word, out_word;
    reg signed [47:0] accumulator;

    wire needs_aux0 = (active_op == OP_AFFINE) || (active_op == OP_MUL) ||
                      (active_op == OP_CLIP);
    wire needs_aux1 = (active_op == OP_AFFINE) || (active_op == OP_CLIP);
    wire is_reduce  = (active_op == OP_RSUM) || (active_op == OP_RSUMSQ);
    wire last_group = (group_index == group_count - 16'd1);

    // One multiplier and one requantizer, shared by every sub-op.
    reg signed [15:0] s_q, a_q, b_q;
    reg signed [31:0] product_q;
    reg signed [32:0] shifted_q;
    reg signed [32:0] round_half;      // registered once per instruction
    reg signed [32:0] wide;
    reg signed [15:0] result_q;
    reg               in_range_q;
    reg signed [47:0] reduce_next;
    reg signed [47:0] final_sum;
    reg signed [47:0] reduce_half;
    reg signed [47:0] rounded_sum;

    function automatic signed [32:0] round_shift33(input signed [32:0] v,
                                                  input [5:0] sh);
        reg signed [32:0] half;
        begin
            if (sh == 6'd0) round_shift33 = v;
            else begin
                half = 33'sd1 <<< (sh - 6'd1);
                round_shift33 = (v[32] ? (v - half) : (v + half)) >>> sh;
            end
        end
    endfunction

    function automatic signed [15:0] sat16_33(input signed [32:0] v);
        begin
            if (v > 33'sd32767)       sat16_33 = 16'sh7fff;
            else if (v < -33'sd32768) sat16_33 = 16'sh8000;
            else                      sat16_33 = v[15:0];
        end
    endfunction

    function automatic signed [15:0] sat16_48(input signed [47:0] v);
        begin
            if (v > 48'sd32767)       sat16_48 = 16'sh7fff;
            else if (v < -48'sd32768) sat16_48 = 16'sh8000;
            else                      sat16_48 = v[15:0];
        end
    endfunction

    always @* begin
        vec_en = 1'b0;
        vec_we = 1'b0;
        vec_addr = {VEC_ADDR_W{1'b0}};
        vec_wr_data = out_word;
        case (state)
            S_SRC_REQ:  begin vec_en = 1'b1; vec_addr = src_base  + group_index[VEC_ADDR_W-1:0]; end
            S_AUX0_REQ: begin vec_en = 1'b1; vec_addr = aux0_base + group_index[VEC_ADDR_W-1:0]; end
            S_AUX1_REQ: begin vec_en = 1'b1; vec_addr = aux1_base + group_index[VEC_ADDR_W-1:0]; end
            S_WRITE: begin
                vec_en = 1'b1; vec_we = 1'b1;
                vec_addr = dst_base + (is_reduce ? {VEC_ADDR_W{1'b0}}
                                                 : group_index[VEC_ADDR_W-1:0]);
                vec_wr_data = out_word;
            end
            default: begin end
        endcase
    end

    always @(posedge clk) begin
        if (!reset_n) begin
            state <= S_IDLE; busy <= 1'b0; done <= 1'b0;
            group_count <= 0; group_index <= 0; total_elements <= 0; lane <= 0;
            src_word <= 0; aux0_word <= 0; aux1_word <= 0; out_word <= 0;
            accumulator <= 0; active_op <= 0; active_shift <= 0;
            s_q <= 0; a_q <= 0; b_q <= 0; product_q <= 0;
            shifted_q <= 0; round_half <= 0; in_range_q <= 1'b0;
            wide <= 0; result_q <= 0; reduce_half <= 0; rounded_sum <= 0;
        end else begin
            done <= 1'b0;
            case (state)
                S_IDLE: begin
                    busy <= 1'b0;
                    if (start) begin
                        busy <= 1'b1;
                        active_op <= subop;
                        active_shift <= shift;
                        total_elements <= element_count;
                        group_count <= (element_count + 16'd7) >> 3;
                        group_index <= 0; lane <= 0;
                        accumulator <= 0; out_word <= 0;
                        round_half <= (shift == 6'd0) ? 33'sd0
                                                      : (33'sd1 <<< (shift - 6'd1));
                        reduce_half <= (shift == 6'd0) ? 48'sd0
                                                       : (48'sd1 <<< (shift - 6'd1));
                        state <= S_SRC_REQ;
                    end
                end

                S_SRC_REQ:  state <= S_SRC_WAIT;
                S_SRC_WAIT: if (vec_rd_valid) begin
                    src_word <= vec_rd_data;
                    lane <= 3'd0;
                    state <= needs_aux0 ? S_AUX0_REQ : S_SEL;
                end
                S_AUX0_REQ:  state <= S_AUX0_WAIT;
                S_AUX0_WAIT: if (vec_rd_valid) begin
                    aux0_word <= vec_rd_data;
                    state <= needs_aux1 ? S_AUX1_REQ : S_SEL;
                end
                S_AUX1_REQ:  state <= S_AUX1_WAIT;
                S_AUX1_WAIT: if (vec_rd_valid) begin
                    aux1_word <= vec_rd_data;
                    state <= S_SEL;
                end

                S_SEL: begin
                    s_q <= $signed(src_word[{2'b0, lane} * 16 +: 16]);
                    a_q <= $signed(aux0_word[{2'b0, lane} * 16 +: 16]);
                    b_q <= $signed(aux1_word[{2'b0, lane} * 16 +: 16]);
                    in_range_q <= (({group_index, 3'b0} | {13'd0, lane}) < total_elements);
                    state <= S_MUL;
                end

                S_MUL: begin
                    // One shared multiplier: src*aux for AFFINE/MUL, src*src
                    // for RSUMSQ.
                    product_q <= s_q * ((active_op == OP_RSUMSQ) ? s_q : a_q);
                    state <= S_SHIFT;
                end

                S_SHIFT: begin
                    // $signed() is load-bearing: a bare concatenation is
                    // unsigned, which turns >>> into a logical shift and makes
                    // every negative product saturate high.
                    wide = $signed({{1{product_q[31]}}, product_q});
                    shifted_q <= (active_shift == 6'd0) ? wide
                        : (((product_q[31] ? (wide - round_half) : (wide + round_half)))
                           >>> active_shift);
                    state <= S_RES;
                end

                S_RES: begin
                    // Compute the lane result and update the reduction here,
                    // so the 128-bit indexed write sits alone in the next
                    // stage. Doing both in one cycle cost -4.7 ns.
                    reduce_next = accumulator;
                    case (active_op)
                        OP_AFFINE: result_q <= sat16_33(shifted_q +
                                                        $signed({{17{b_q[15]}}, b_q}));
                        OP_MUL:    result_q <= sat16_33(shifted_q);
                        OP_CLIP:   result_q <= (s_q < a_q) ? a_q
                                             : ((s_q > b_q) ? b_q : s_q);
                        OP_RSUM: begin
                            result_q <= 16'sd0;
                            if (in_range_q)
                                reduce_next = accumulator + {{32{s_q[15]}}, s_q};
                        end
                        OP_RSUMSQ: begin
                            result_q <= 16'sd0;
                            if (in_range_q)
                                reduce_next = accumulator + {{16{1'b0}}, product_q};
                        end
                        default: result_q <= 16'sd0;
                    endcase
                    accumulator <= reduce_next;
                    state <= S_OUT;
                end

                S_OUT: begin
                    if (!is_reduce)
                        out_word[{2'b0, lane} * 16 +: 16] <= in_range_q ? result_q : 16'sd0;

                    if (lane == 3'd7) begin
                        if (is_reduce) begin
                            if (last_group) begin
                                // The 48-bit variable shift gets its own stage;
                                // folded into the write it cost -1.1 ns.
                                state <= S_RFIN;
                            end else begin
                                group_index <= group_index + 16'd1;
                                state <= S_SRC_REQ;
                            end
                        end else begin
                            state <= S_WRITE;
                        end
                    end else begin
                        lane <= lane + 3'd1;
                        state <= S_SEL;
                    end
                end

                S_RFIN: begin
                    // 48-bit add, variable shift and saturate are three stages:
                    // together they were the last -0.2 ns.
                    rounded_sum <= accumulator + reduce_half;
                    state <= S_RFIN2;
                end

                S_RFIN2: begin
                    final_sum = (active_shift == 6'd0) ? accumulator
                                                       : (rounded_sum >>> active_shift);
                    out_word <= 128'd0;
                    out_word[15:0] <= sat16_48(final_sum);
                    state <= S_WRITE;
                end

                S_WRITE: begin
                    if (is_reduce || last_group) begin
                        busy <= 1'b0; done <= 1'b1; state <= S_IDLE;
                    end else begin
                        group_index <= group_index + 16'd1;
                        state <= S_SRC_REQ;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
