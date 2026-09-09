`timescale 1ns/1ps
`include "rl_accel_params.vh"

// RoboAccel-v3 shared special-function unit.
//
// One curve ROM and one interpolator, time-shared across the eight lanes of a
// vector word. Activation and transcendental work is a small fraction of a
// control policy's cycles, so serialising the lanes buys a 4-curve SFU for
// roughly the area of a single lane instead of eight.
//
//   curve 0 ELU      (bit-identical to rl_elu_array8, kept for compatibility)
//   curve 1 TANH     LSTM cell/output activation
//   curve 2 SIGMOID  LSTM gates
//   curve 3 RSQRT    L2 normalisation, with the reduction done by the vector engine
//
// Q8.8 in and out, index = min(|x| >> 3, 255), 3-bit linear interpolation --
// the same numeric contract the ELU table already uses.
module rl_sfu_engine #(
    parameter integer VEC_ADDR_W = 9
) (
    input  wire                    clk,
    input  wire                    reset_n,
    input  wire                    start,
    input  wire [1:0]              curve,
    input  wire [VEC_ADDR_W-1:0]   src_base,
    input  wire [VEC_ADDR_W-1:0]   dst_base,
    input  wire [15:0]             element_count,
    output reg                     busy,
    output reg                     done,

    output reg                     vec_en,
    output reg                     vec_we,
    output reg [VEC_ADDR_W-1:0]    vec_addr,
    output reg [127:0]             vec_wr_data,
    input  wire                    vec_rd_valid,
    input  wire [127:0]            vec_rd_data
);
    // The lookup is pipelined: address, registered ROM read, interpolate,
    // then curve select and saturate. A single-cycle version put a 31-level
    // combinational chain (ROM mux -> subtract -> multiply -> shift -> add ->
    // saturate) on the critical path and closed at -13.8 ns.
    localparam [3:0] S_IDLE   = 4'd0;
    localparam [3:0] S_READ   = 4'd1;
    localparam [3:0] S_WAIT   = 4'd2;
    localparam [3:0] S_LOOK   = 4'd3;
    localparam [3:0] S_ROM    = 4'd4;
    localparam [3:0] S_DIFF   = 4'd5;
    localparam [3:0] S_INTERP = 4'd6;
    localparam [3:0] S_SELECT = 4'd7;
    localparam [3:0] S_WRITE  = 4'd8;

    (* rom_style = "block" *) reg [15:0] curve_rom [0:1023];
    initial $readmemh("rl_sfu_table.hex", curve_rom);

    reg [3:0]  state;
    reg [1:0]  active_curve;
    reg [15:0] group_count, group_index, total_elements;
    reg [2:0]  lane;
    reg [127:0] src_word, out_word;
    reg [15:0] rom_a, rom_b;
    reg signed [15:0] x;
    reg [15:0] magnitude;
    reg [7:0]  index;
    reg [2:0]  remainder;
    reg        negative, saturated;

    wire [7:0] next_index = (index == 8'd255) ? 8'd255 : (index + 8'd1);

    always @* begin
        vec_en = 1'b0; vec_we = 1'b0;
        vec_addr = {VEC_ADDR_W{1'b0}};
        vec_wr_data = out_word;
        case (state)
            S_READ: begin vec_en = 1'b1; vec_addr = src_base + group_index[VEC_ADDR_W-1:0]; end
            S_WRITE: begin
                vec_en = 1'b1; vec_we = 1'b1;
                vec_addr = dst_base + group_index[VEC_ADDR_W-1:0];
                vec_wr_data = out_word;
            end
            default: begin end
        endcase
    end

    function automatic signed [15:0] sat16_32(input signed [31:0] v);
        begin
            if (v > 32'sd32767)       sat16_32 = 16'sh7fff;
            else if (v < -32'sd32768) sat16_32 = 16'sh8000;
            else                      sat16_32 = v[15:0];
        end
    endfunction

    reg signed [31:0] interp_q;
    reg signed [17:0] diff_q;
    reg signed [17:0] base_q;
    reg signed [15:0] lane_x;

    always @(posedge clk) begin
        if (!reset_n) begin
            state <= S_IDLE; busy <= 1'b0; done <= 1'b0;
            group_count <= 0; group_index <= 0; total_elements <= 0;
            lane <= 0; src_word <= 0; out_word <= 0;
            active_curve <= 0; index <= 0; remainder <= 0;
            interp_q <= 0; lane_x <= 0; diff_q <= 0; base_q <= 0;
            negative <= 1'b0; saturated <= 1'b0; magnitude <= 0;
            rom_a <= 0; rom_b <= 0; x <= 0;
        end else begin
            done <= 1'b0;
            case (state)
                S_IDLE: begin
                    busy <= 1'b0;
                    if (start) begin
                        busy <= 1'b1;
                        active_curve <= curve;
                        total_elements <= element_count;
                        group_count <= (element_count + 16'd7) >> 3;
                        group_index <= 0; lane <= 0;
                        state <= S_READ;
                    end
                end
                S_READ: state <= S_WAIT;
                S_WAIT: if (vec_rd_valid) begin
                    src_word <= vec_rd_data;
                    lane <= 3'd0;
                    state <= S_LOOK;
                end
                S_LOOK: begin
                    x = $signed(src_word[{2'b0, lane} * 16 +: 16]);
                    lane_x    <= x;
                    negative  <= x[15];
                    saturated <= (x <= -(16'sd8 * 16'sd256));
                    magnitude = x[15] ? (~x + 16'sd1) : x;
                    index     <= (magnitude[15:3] > 13'd255) ? 8'd255 : magnitude[10:3];
                    remainder <= magnitude[2:0];
                    state <= S_ROM;
                end

                S_ROM: begin
                    // Registered reads so Vivado infers a block ROM.
                    rom_a <= curve_rom[{active_curve, index}];
                    rom_b <= curve_rom[{active_curve, next_index}];
                    state <= S_DIFF;
                end

                S_DIFF: begin
                    // The BRAM clock-to-out plus subtract plus multiply plus
                    // add did not fit in one cycle (-3.7 ns); the subtract gets
                    // its own stage.
                    base_q <= $signed({{2{rom_a[15]}}, rom_a});
                    diff_q <= $signed({{2{rom_b[15]}}, rom_b}) -
                              $signed({{2{rom_a[15]}}, rom_a});
                    state <= S_INTERP;
                end

                S_INTERP: begin
                    interp_q <= $signed({{14{base_q[17]}}, base_q}) +
                                (((diff_q * $signed({1'b0, remainder})) + 18'sd4) >>> 3);
                    state <= S_SELECT;
                end

                S_SELECT: begin
                    if (({group_index, 3'b0} | {13'd0, lane}) >= total_elements)
                        out_word[{2'b0, lane} * 16 +: 16] <= 16'sd0;
                    else begin
                        case (active_curve)
                            2'd0: out_word[{2'b0, lane} * 16 +: 16] <=
                                      saturated ? -16'sd256
                                                : (negative ? sat16_32(interp_q) : lane_x);
                            2'd1: out_word[{2'b0, lane} * 16 +: 16] <=
                                      negative ? sat16_32(-interp_q) : sat16_32(interp_q);
                            2'd2: out_word[{2'b0, lane} * 16 +: 16] <=
                                      negative ? sat16_32(32'sd256 - interp_q)
                                               : sat16_32(interp_q);
                            default: out_word[{2'b0, lane} * 16 +: 16] <= sat16_32(interp_q);
                        endcase
                    end
                    if (lane == 3'd7) state <= S_WRITE;
                    else begin lane <= lane + 3'd1; state <= S_LOOK; end
                end

                S_WRITE: begin
                    if (group_index == group_count - 16'd1) begin
                        busy <= 1'b0; done <= 1'b1; state <= S_IDLE;
                    end else begin
                        group_index <= group_index + 16'd1;
                        state <= S_READ;
                    end
                end
                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
