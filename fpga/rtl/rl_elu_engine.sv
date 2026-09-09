`timescale 1ns/1ps

module rl_elu_engine #(
    parameter integer VEC_ADDR_W = 9
) (
    input  wire                    clk,
    input  wire                    reset_n,
    input  wire                    start,
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
    localparam S_IDLE   = 3'd0;
    localparam S_READ   = 3'd1;
    localparam S_WAIT   = 3'd2;
    localparam S_EXEC   = 3'd3;
    localparam S_RESULT = 3'd4;
    localparam S_WRITE  = 3'd5;

    reg [2:0] state;
    reg [15:0] group_count;
    reg [15:0] group_index;
    reg [127:0] input_word;
    reg [127:0] output_word;
    reg elu_valid;
    wire elu_out_valid;
    wire [127:0] elu_out_data;
    integer lane;

    rl_elu_array8 u_elu (
        .clk(clk), .reset_n(reset_n), .i_valid(elu_valid),
        .i_data(input_word), .o_valid(elu_out_valid), .o_data(elu_out_data)
    );

    always @* begin
        vec_en = 1'b0;
        vec_we = 1'b0;
        vec_addr = 0;
        vec_wr_data = output_word;
        elu_valid = 1'b0;
        case (state)
            S_READ: begin
                vec_en = 1'b1;
                vec_addr = src_base + group_index[VEC_ADDR_W-1:0];
            end
            S_EXEC: elu_valid = 1'b1;
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
            input_word <= 0;
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
                        state <= S_READ;
                    end
                end
                S_READ: state <= S_WAIT;
                S_WAIT: if (vec_rd_valid) begin
                    input_word <= vec_rd_data;
                    state <= S_EXEC;
                end
                S_EXEC: state <= S_RESULT;
                S_RESULT: if (elu_out_valid) begin
                    for (lane = 0; lane < 8; lane = lane + 1) begin
                        if ((group_index * 8 + lane) < element_count)
                            output_word[lane*16 +: 16] <= elu_out_data[lane*16 +: 16];
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
                        state <= S_READ;
                    end
                end
                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
