`timescale 1ns/1ps

// Eight dot products are evaluated in parallel.  Each dot product consumes
// eight signed INT16 activation/weight pairs per accepted cycle.
module rl_mac_array_8x8 (
    input  wire           clk,
    input  wire           reset_n,
    input  wire           i_valid,
    input  wire [127:0]   i_activations,
    // Bank j occupies bits [j*128 +: 128]; lane k is [k*16 +: 16].
    input  wire [1023:0]  i_weights,
    output wire           o_valid,
    output wire [383:0]   o_row_sums
);
    integer j;
    integer k;

    (* use_dsp = "yes" *) reg signed [31:0] product [0:7][0:7];
    // Keep the reduction tree in LUT/CARRY logic.  Otherwise Vivado absorbs
    // eight adders into DSP48E1 slices and exhausts every DSP on XC7Z010.
    (* use_dsp = "no" *) reg signed [32:0] sum2 [0:7][0:3];
    (* use_dsp = "no" *) reg signed [33:0] sum4 [0:7][0:1];
    (* use_dsp = "no" *) reg signed [34:0] sum8 [0:7];
    reg [3:0] valid_pipe;

    wire signed [15:0] act_lane [0:7];
    wire signed [15:0] weight_lane [0:7][0:7];

    genvar gj;
    genvar gk;
    generate
        for (gk = 0; gk < 8; gk = gk + 1) begin : g_act
            assign act_lane[gk] = i_activations[gk*16 +: 16];
        end
        for (gj = 0; gj < 8; gj = gj + 1) begin : g_w_row
            for (gk = 0; gk < 8; gk = gk + 1) begin : g_w_col
                assign weight_lane[gj][gk] = i_weights[gj*128 + gk*16 +: 16];
            end
            assign o_row_sums[gj*48 +: 48] = {{13{sum8[gj][34]}}, sum8[gj]};
        end
    endgenerate

    assign o_valid = valid_pipe[3];

    always @(posedge clk) begin
        if (!reset_n) begin
            valid_pipe <= 4'b0;
            for (j = 0; j < 8; j = j + 1) begin
                sum8[j] <= 0;
                for (k = 0; k < 8; k = k + 1)
                    product[j][k] <= 0;
                for (k = 0; k < 4; k = k + 1)
                    sum2[j][k] <= 0;
                for (k = 0; k < 2; k = k + 1)
                    sum4[j][k] <= 0;
            end
        end else begin
            valid_pipe <= {valid_pipe[2:0], i_valid};

            if (i_valid) begin
                for (j = 0; j < 8; j = j + 1)
                    for (k = 0; k < 8; k = k + 1)
                        product[j][k] <= act_lane[k] * weight_lane[j][k];
            end

            if (valid_pipe[0]) begin
                for (j = 0; j < 8; j = j + 1)
                    for (k = 0; k < 4; k = k + 1)
                        sum2[j][k] <= $signed(product[j][2*k]) +
                                      $signed(product[j][2*k+1]);
            end

            if (valid_pipe[1]) begin
                for (j = 0; j < 8; j = j + 1) begin
                    sum4[j][0] <= $signed(sum2[j][0]) + $signed(sum2[j][1]);
                    sum4[j][1] <= $signed(sum2[j][2]) + $signed(sum2[j][3]);
                end
            end

            if (valid_pipe[2]) begin
                for (j = 0; j < 8; j = j + 1)
                    sum8[j] <= $signed(sum4[j][0]) + $signed(sum4[j][1]);
            end
        end
    end
endmodule
