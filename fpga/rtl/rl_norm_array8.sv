`timescale 1ns/1ps

module rl_norm_array8 (
    input  wire          clk,
    input  wire          reset_n,
    input  wire          i_valid,
    input  wire [127:0]  i_data,
    input  wire [127:0]  i_mean,
    input  wire [127:0]  i_inv_std,
    input  wire [5:0]    i_shift,
    output reg           o_valid,
    output reg  [127:0]  o_data
);
    (* use_dsp = "yes" *) reg signed [28:0] product_s1 [0:7];
    reg [5:0] shift_s1 [0:7];
    (* use_dsp = "no" *) reg signed [29:0] adjusted_s2 [0:7];
    reg [5:0] shift_s2 [0:7];
    (* use_dsp = "no" *) reg signed [29:0] shifted_s3 [0:7];
    reg valid_s1;
    reg valid_s2;
    reg valid_s3;
    integer lane;

    wire signed [16:0] difference [0:7];
    wire signed [11:0] mantissa [0:7];
    wire [4:0] exponent [0:7];

    genvar gl;
    generate
        for (gl = 0; gl < 8; gl = gl + 1) begin : g_lane
            assign difference[gl] = $signed(i_data[gl*16 +: 16]) -
                                    $signed(i_mean[gl*16 +: 16]);
            assign mantissa[gl] = {1'b0, i_inv_std[gl*16 +: 11]};
            assign exponent[gl] = i_inv_std[gl*16 + 11 +: 5];
        end
    endgenerate

    always @(posedge clk) begin
        if (!reset_n) begin
            valid_s1 <= 1'b0;
            valid_s2 <= 1'b0;
            valid_s3 <= 1'b0;
            o_valid <= 1'b0;
            o_data <= 0;
        end else begin
            valid_s1 <= i_valid;
            valid_s2 <= valid_s1;
            valid_s3 <= valid_s2;
            o_valid <= valid_s3;

            if (i_valid) begin
                for (lane = 0; lane < 8; lane = lane + 1) begin
                    product_s1[lane] <= difference[lane] * mantissa[lane];
                    if (exponent[lane] > i_shift)
                        shift_s1[lane] <= exponent[lane] - i_shift;
                    else
                        shift_s1[lane] <= 0;
                end
            end

            if (valid_s1) begin
                for (lane = 0; lane < 8; lane = lane + 1) begin
                    if (shift_s1[lane] == 0) begin
                        adjusted_s2[lane] <= $signed(product_s1[lane]);
                    end else if (product_s1[lane][28]) begin
                        adjusted_s2[lane] <= $signed(product_s1[lane]) -
                            (30'sd1 <<< (shift_s1[lane] - 1'b1));
                    end else begin
                        adjusted_s2[lane] <= $signed(product_s1[lane]) +
                            (30'sd1 <<< (shift_s1[lane] - 1'b1));
                    end
                    shift_s2[lane] <= shift_s1[lane];
                end
            end

            if (valid_s2)
                for (lane = 0; lane < 8; lane = lane + 1)
                    shifted_s3[lane] <= adjusted_s2[lane] >>> shift_s2[lane];

            if (valid_s3) begin
                for (lane = 0; lane < 8; lane = lane + 1) begin
                    if (shifted_s3[lane] > 32767)
                        o_data[lane*16 +: 16] <= 16'sh7fff;
                    else if (shifted_s3[lane] < -32768)
                        o_data[lane*16 +: 16] <= -16'sh8000;
                    else
                        o_data[lane*16 +: 16] <= shifted_s3[lane][15:0];
                end
            end
        end
    end
endmodule
