`timescale 1ns/1ps

module rl_norm_engine #(
    parameter integer VEC_ADDR_W = 9
) (
    input  wire                    clk,
    input  wire                    reset_n,
    input  wire                    start,
    input  wire [VEC_ADDR_W-1:0]   src_base,
    input  wire [VEC_ADDR_W-1:0]   dst_base,
    input  wire [VEC_ADDR_W-1:0]   mean_base,
    input  wire [VEC_ADDR_W-1:0]   inv_std_base,
    input  wire [15:0]             element_count,
    input  wire [5:0]              output_shift,
    output reg                     busy,
    output reg                     done,
    output reg                     vec_en,
    output reg                     vec_we,
    output reg [VEC_ADDR_W-1:0]    vec_addr,
    output reg [127:0]             vec_wr_data,
    input  wire                    vec_rd_valid,
    input  wire [127:0]            vec_rd_data
);
    localparam S_IDLE       = 4'd0;
    localparam S_SRC_REQ    = 4'd1;
    localparam S_SRC_WAIT   = 4'd2;
    localparam S_MEAN_REQ   = 4'd3;
    localparam S_MEAN_WAIT  = 4'd4;
    localparam S_INV_REQ    = 4'd5;
    localparam S_INV_WAIT   = 4'd6;
    localparam S_EXEC       = 4'd7;
    localparam S_RESULT     = 4'd8;
    localparam S_WRITE      = 4'd9;

    reg [3:0] state;
    reg [15:0] group_count;
    reg [15:0] group_index;
    reg [127:0] src_word;
    reg [127:0] mean_word;
    reg [127:0] inv_word;
    reg [127:0] output_word;
    reg norm_valid;
    wire norm_out_valid;
    wire [127:0] norm_out_data;
    integer lane;

    rl_norm_array8 u_norm (
        .clk(clk), .reset_n(reset_n), .i_valid(norm_valid),
        .i_data(src_word), .i_mean(mean_word), .i_inv_std(inv_word),
        .i_shift(output_shift), .o_valid(norm_out_valid), .o_data(norm_out_data)
    );

    always @* begin
        vec_en = 1'b0;
        vec_we = 1'b0;
        vec_addr = 0;
        vec_wr_data = output_word;
        norm_valid = 1'b0;
        case (state)
            S_SRC_REQ: begin
                vec_en = 1'b1;
                vec_addr = src_base + group_index[VEC_ADDR_W-1:0];
            end
            S_MEAN_REQ: begin
                vec_en = 1'b1;
                vec_addr = mean_base + group_index[VEC_ADDR_W-1:0];
            end
            S_INV_REQ: begin
                vec_en = 1'b1;
                vec_addr = inv_std_base + group_index[VEC_ADDR_W-1:0];
            end
            S_EXEC: norm_valid = 1'b1;
            S_WRITE: begin
                vec_en = 1'b1;
                vec_we = 1'b1;
                vec_addr = dst_base + group_index[VEC_ADDR_W-1:0];
                vec_wr_data = output_word;
            end
            default: begin end
        endcase
    end

    always @(posedge clk) begin
        if (!reset_n) begin
            state <= S_IDLE;
            busy <= 1'b0;
            done <= 1'b0;
            group_count <= 0;
            group_index <= 0;
            src_word <= 0;
            mean_word <= 0;
            inv_word <= 0;
            output_word <= 0;
        end else begin
            done <= 1'b0;
            case (state)
                S_IDLE: begin
                    busy <= 1'b0;
                    if (start) begin
                        busy <= 1'b1;
                        group_count <= (element_count + 16'd7) >> 3;
                        group_index <= 0;
                        state <= S_SRC_REQ;
                    end
                end
                S_SRC_REQ: state <= S_SRC_WAIT;
                S_SRC_WAIT: if (vec_rd_valid) begin
                    src_word <= vec_rd_data;
                    state <= S_MEAN_REQ;
                end
                S_MEAN_REQ: state <= S_MEAN_WAIT;
                S_MEAN_WAIT: if (vec_rd_valid) begin
                    mean_word <= vec_rd_data;
                    state <= S_INV_REQ;
                end
                S_INV_REQ: state <= S_INV_WAIT;
                S_INV_WAIT: if (vec_rd_valid) begin
                    inv_word <= vec_rd_data;
                    state <= S_EXEC;
                end
                S_EXEC: state <= S_RESULT;
                S_RESULT: if (norm_out_valid) begin
                    for (lane = 0; lane < 8; lane = lane + 1) begin
                        if ((group_index * 8 + lane) < element_count)
                            output_word[lane*16 +: 16] <= norm_out_data[lane*16 +: 16];
                        else
                            output_word[lane*16 +: 16] <= 0;
                    end
                    state <= S_WRITE;
                end
                S_WRITE: begin
                    if (group_index == group_count - 1'b1) begin
                        busy <= 1'b0;
                        done <= 1'b1;
                        state <= S_IDLE;
                    end else begin
                        group_index <= group_index + 1'b1;
                        state <= S_SRC_REQ;
                    end
                end
                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
