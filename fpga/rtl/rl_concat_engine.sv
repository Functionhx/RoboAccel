`timescale 1ns/1ps

// Concatenate two INT16 vectors even when the first vector is not word aligned.
// dim_k is the number of elements in source 0; dim_n is the number in source 1.
//
// src0_offset shifts the start of source 0 by an arbitrary element count. With
// count1 = 0 that turns this engine into an unaligned SLICE, which is what the
// estimator-style policies in the corpus need, at the cost of one adder. It is
// zero for every v1/v2 program, so existing behaviour is unchanged.
module rl_concat_engine #(
    parameter integer VEC_ADDR_W = 9
) (
    input  wire                       clk,
    input  wire                       reset_n,
    input  wire                       start,
    input  wire [VEC_ADDR_W-1:0]      src0_base,
    input  wire [VEC_ADDR_W-1:0]      src1_base,
    input  wire [VEC_ADDR_W-1:0]      dst_base,
    input  wire [15:0]                count0,
    input  wire [15:0]                count1,
    input  wire [VEC_ADDR_W-1:0]      src0_offset,
    output reg                        busy,
    output reg                        done,
    output reg                        vec_en,
    output reg                        vec_we,
    output reg [VEC_ADDR_W-1:0]       vec_addr,
    output reg [127:0]                vec_wr_data,
    input  wire                       vec_rd_valid,
    input  wire [127:0]               vec_rd_data
);
    localparam [2:0] S_IDLE = 3'd0;
    localparam [2:0] S_READ_REQ = 3'd1;
    localparam [2:0] S_READ_WAIT = 3'd2;
    localparam [2:0] S_STORE = 3'd3;
    localparam [2:0] S_WRITE = 3'd4;

    reg [2:0] state;
    reg [16:0] total_count;
    reg [16:0] element_index;
    reg [VEC_ADDR_W-1:0] output_word_index;
    reg [2:0] output_lane;
    reg [127:0] source_word;
    reg [127:0] output_word;
    reg final_write;

    wire select_source1 = (element_index >= {1'b0, count0});
    wire [16:0] source_index = select_source1 ?
        (element_index - {1'b0, count0})
      : (element_index + {{(17-VEC_ADDR_W){1'b0}}, src0_offset});
    wire [2:0] source_lane = source_index[2:0];
    wire [15:0] source_value = source_word[source_lane*16 +: 16];

    always @* begin
        vec_en = 1'b0;
        vec_we = 1'b0;
        vec_addr = {VEC_ADDR_W{1'b0}};
        vec_wr_data = output_word;
        case (state)
            S_READ_REQ: begin
                vec_en = 1'b1;
                if (select_source1)
                    vec_addr = src1_base + source_index[VEC_ADDR_W+2:3];
                else
                    vec_addr = src0_base + source_index[VEC_ADDR_W+2:3];
            end
            S_WRITE: begin
                vec_en = 1'b1;
                vec_we = 1'b1;
                vec_addr = dst_base + output_word_index;
                vec_wr_data = output_word;
            end
            default: begin end
        endcase
    end

    always @(posedge clk) begin
        if (!reset_n) begin
            state <= S_IDLE;
            total_count <= 0;
            element_index <= 0;
            output_word_index <= 0;
            output_lane <= 0;
            source_word <= 0;
            output_word <= 0;
            final_write <= 1'b0;
            busy <= 1'b0;
            done <= 1'b0;
        end else begin
            done <= 1'b0;
            case (state)
                S_IDLE: begin
                    busy <= 1'b0;
                    if (start) begin
                        busy <= 1'b1;
                        total_count <= {1'b0, count0} + {1'b0, count1};
                        element_index <= 0;
                        output_word_index <= 0;
                        output_lane <= 0;
                        output_word <= 0;
                        final_write <= 1'b0;
                        state <= S_READ_REQ;
                    end
                end
                S_READ_REQ: state <= S_READ_WAIT;
                S_READ_WAIT: begin
                    if (vec_rd_valid) begin
                        source_word <= vec_rd_data;
                        state <= S_STORE;
                    end
                end
                S_STORE: begin
                    output_word[output_lane*16 +: 16] <= source_value;
                    final_write <= (element_index + 1'b1 == total_count);
                    element_index <= element_index + 1'b1;
                    if ((output_lane == 3'd7) ||
                        (element_index + 1'b1 == total_count)) begin
                        state <= S_WRITE;
                    end else begin
                        output_lane <= output_lane + 1'b1;
                        state <= S_READ_REQ;
                    end
                end
                S_WRITE: begin
                    if (final_write) begin
                        busy <= 1'b0;
                        done <= 1'b1;
                        state <= S_IDLE;
                    end else begin
                        output_word_index <= output_word_index + 1'b1;
                        output_lane <= 0;
                        output_word <= 0;
                        state <= S_READ_REQ;
                    end
                end
                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
