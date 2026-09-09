`timescale 1ns/1ps

module rl_gemm_engine #(
    parameter integer VEC_ADDR_W = 9,
    parameter integer WT_ADDR_W = 11
) (
    input  wire                    clk,
    input  wire                    reset_n,
    input  wire                    start,
    input  wire [VEC_ADDR_W-1:0]   src_base,
    input  wire [VEC_ADDR_W-1:0]   dst_base,
    input  wire [VEC_ADDR_W-1:0]   bias_base,
    input  wire [WT_ADDR_W-1:0]    weight_base,
    input  wire [15:0]             dim_k,
    input  wire [15:0]             dim_n,
    input  wire [5:0]              output_shift,
    output reg                     busy,
    output reg                     done,

    output reg                     vec_en,
    output reg                     vec_we,
    output reg [VEC_ADDR_W-1:0]    vec_addr,
    output reg [127:0]             vec_wr_data,
    input  wire                    vec_rd_valid,
    input  wire [127:0]            vec_rd_data,

    output reg                     wt_rd_en,
    output reg [WT_ADDR_W-1:0]     wt_rd_addr,
    input  wire                    wt_rd_valid,
    input  wire [1023:0]           wt_rd_data,

    // RoboAccel-v2: the weight source may stall. The on-chip cache ties this
    // high, so the cache path behaves exactly as it did in v1.
    input  wire                    wt_ready
);
    localparam S_IDLE        = 4'd0;
    localparam S_BIAS_REQ    = 4'd1;
    localparam S_BIAS_WAIT   = 4'd2;
    localparam S_STREAM      = 4'd3;
    localparam S_QUANT_ADJ   = 4'd4;
    localparam S_QUANT_SHIFT = 4'd5;
    localparam S_QUANT_BIAS  = 4'd6;
    localparam S_QUANT_SAT   = 4'd7;
    localparam S_WRITE       = 4'd8;

    reg [3:0] state;
    reg [15:0] k_groups;
    reg [15:0] n_groups;
    reg [15:0] out_group;
    reg [15:0] request_count;
    reg [15:0] result_count;
    reg [WT_ADDR_W-1:0] group_weight_base;
    reg [127:0] bias_word;
    reg signed [47:0] accumulator [0:7];
    reg signed [48:0] quant_adjusted [0:7];
    reg signed [48:0] quant_shifted [0:7];
    reg signed [48:0] quant_biased [0:7];
    reg [127:0] output_word;

    wire mac_in_valid;
    wire mac_out_valid;
    wire [383:0] mac_row_sums;
    wire signed [47:0] total_value [0:7];

    integer lane;

    assign mac_in_valid = (state == S_STREAM) && vec_rd_valid && wt_rd_valid;

    genvar gl;
    generate
        for (gl = 0; gl < 8; gl = gl + 1) begin : g_round
            assign total_value[gl] = accumulator[gl] +
                $signed(mac_row_sums[gl*48 +: 48]);
        end
    endgenerate

    rl_mac_array_8x8 u_mac (
        .clk(clk), .reset_n(reset_n), .i_valid(mac_in_valid),
        .i_activations(vec_rd_data), .i_weights(wt_rd_data),
        .o_valid(mac_out_valid), .o_row_sums(mac_row_sums)
    );

    always @* begin
        vec_en = 1'b0;
        vec_we = 1'b0;
        vec_addr = {VEC_ADDR_W{1'b0}};
        vec_wr_data = output_word;
        wt_rd_en = 1'b0;
        wt_rd_addr = {WT_ADDR_W{1'b0}};

        case (state)
            S_BIAS_REQ: begin
                vec_en = 1'b1;
                vec_addr = bias_base + out_group[VEC_ADDR_W-1:0];
            end
            S_STREAM: begin
                // Activations and weights are requested together so their
                // valids stay aligned; if the weight source cannot deliver
                // this cycle, neither request is issued.
                if ((request_count < k_groups) && wt_ready) begin
                    vec_en = 1'b1;
                    vec_addr = src_base + request_count[VEC_ADDR_W-1:0];
                    wt_rd_en = 1'b1;
                    wt_rd_addr = group_weight_base + request_count[WT_ADDR_W-1:0];
                end
            end
            S_WRITE: begin
                vec_en = 1'b1;
                vec_we = 1'b1;
                vec_addr = dst_base + out_group[VEC_ADDR_W-1:0];
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
            k_groups <= 0;
            n_groups <= 0;
            out_group <= 0;
            request_count <= 0;
            result_count <= 0;
            group_weight_base <= 0;
            bias_word <= 0;
            output_word <= 0;
            for (lane = 0; lane < 8; lane = lane + 1)
                accumulator[lane] <= 0;
            for (lane = 0; lane < 8; lane = lane + 1) begin
                quant_adjusted[lane] <= 0;
                quant_shifted[lane] <= 0;
                quant_biased[lane] <= 0;
            end
        end else begin
            done <= 1'b0;
            case (state)
                S_IDLE: begin
                    busy <= 1'b0;
                    if (start) begin
                        busy <= 1'b1;
                        k_groups <= (dim_k + 16'd7) >> 3;
                        n_groups <= (dim_n + 16'd7) >> 3;
                        out_group <= 0;
                        group_weight_base <= weight_base;
                        state <= S_BIAS_REQ;
                    end
                end

                S_BIAS_REQ: state <= S_BIAS_WAIT;

                S_BIAS_WAIT: begin
                    if (vec_rd_valid) begin
                        bias_word <= vec_rd_data;
                        request_count <= 0;
                        result_count <= 0;
                        for (lane = 0; lane < 8; lane = lane + 1)
                            accumulator[lane] <= 0;
                        state <= S_STREAM;
                    end
                end

                S_STREAM: begin
                    if ((request_count < k_groups) && wt_ready)
                        request_count <= request_count + 1'b1;

                    if (mac_out_valid) begin
                        if (result_count == k_groups - 1'b1) begin
                            for (lane = 0; lane < 8; lane = lane + 1)
                                accumulator[lane] <= total_value[lane];
                            state <= S_QUANT_ADJ;
                        end else begin
                            for (lane = 0; lane < 8; lane = lane + 1)
                                accumulator[lane] <= total_value[lane];
                            result_count <= result_count + 1'b1;
                        end
                    end
                end

                // The final requantization is deliberately split into four
                // stages.  This removes a 48-bit add + barrel shift + bias add
                // + saturation chain from the 100 MHz critical path.
                S_QUANT_ADJ: begin
                    for (lane = 0; lane < 8; lane = lane + 1) begin
                        if (output_shift == 0)
                            quant_adjusted[lane] <= {accumulator[lane][47], accumulator[lane]};
                        else if (accumulator[lane][47])
                            quant_adjusted[lane] <= {accumulator[lane][47], accumulator[lane]} -
                                (49'sd1 <<< (output_shift - 1'b1));
                        else
                            quant_adjusted[lane] <= {accumulator[lane][47], accumulator[lane]} +
                                (49'sd1 <<< (output_shift - 1'b1));
                    end
                    state <= S_QUANT_SHIFT;
                end

                S_QUANT_SHIFT: begin
                    for (lane = 0; lane < 8; lane = lane + 1)
                        quant_shifted[lane] <= quant_adjusted[lane] >>> output_shift;
                    state <= S_QUANT_BIAS;
                end

                S_QUANT_BIAS: begin
                    for (lane = 0; lane < 8; lane = lane + 1)
                        quant_biased[lane] <= quant_shifted[lane] +
                            {{33{bias_word[lane*16+15]}}, bias_word[lane*16 +: 16]};
                    state <= S_QUANT_SAT;
                end

                S_QUANT_SAT: begin
                    for (lane = 0; lane < 8; lane = lane + 1) begin
                        if ((out_group * 8 + lane) >= dim_n)
                            output_word[lane*16 +: 16] <= 16'b0;
                        else if (quant_biased[lane] > 49'sd32767)
                            output_word[lane*16 +: 16] <= 16'sh7fff;
                        else if (quant_biased[lane] < -49'sd32768)
                            output_word[lane*16 +: 16] <= 16'sh8000;
                        else
                            output_word[lane*16 +: 16] <= quant_biased[lane][15:0];
                    end
                    state <= S_WRITE;
                end

                S_WRITE: begin
                    if (out_group == n_groups - 1'b1) begin
                        busy <= 1'b0;
                        done <= 1'b1;
                        state <= S_IDLE;
                    end else begin
                        out_group <= out_group + 1'b1;
                        group_weight_base <= group_weight_base + k_groups[WT_ADDR_W-1:0];
                        state <= S_BIAS_REQ;
                    end
                end

                default: state <= S_IDLE;
            endcase
        end
    end
endmodule
