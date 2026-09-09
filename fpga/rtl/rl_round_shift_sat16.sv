`timescale 1ns/1ps

module rl_round_shift_sat16 #(
    parameter integer ACC_W = 48
) (
    input  wire signed [ACC_W-1:0] i_value,
    input  wire signed [15:0]      i_bias,
    input  wire        [5:0]       i_shift,
    output reg  signed [15:0]      o_value
);
    reg signed [ACC_W:0] adjusted;
    reg signed [ACC_W:0] shifted;
    reg signed [ACC_W:0] biased;
    reg signed [ACC_W:0] half_lsb;

    always @* begin
        adjusted = {i_value[ACC_W-1], i_value};
        half_lsb = 0;
        if (i_shift != 0) begin
            half_lsb = ({{ACC_W{1'b0}}, 1'b1} <<< (i_shift - 1'b1));
            if (i_value[ACC_W-1])
                adjusted = adjusted - half_lsb;
            else
                adjusted = adjusted + half_lsb;
        end

        shifted = adjusted >>> i_shift;
        biased = shifted + {{(ACC_W-15){i_bias[15]}}, i_bias};

        if (biased > 32767)
            o_value = 16'sh7fff;
        else if (biased < -32768)
            o_value = -16'sh8000;
        else
            o_value = biased[15:0];
    end
endmodule
