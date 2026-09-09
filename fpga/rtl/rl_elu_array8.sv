`timescale 1ns/1ps

// ELU(alpha=1) for Q8.8 data. Negative inputs use a 256-entry table over
// [-8, 0] with a 1/32 step; values below -8 saturate to -1.0.
module rl_elu_array8 (
    input  wire          clk,
    input  wire          reset_n,
    input  wire          i_valid,
    input  wire [127:0]  i_data,
    output reg           o_valid,
    output reg  [127:0]  o_data
);
    integer lane;

    function automatic signed [15:0] elu_rom;
        input [7:0] index;
        begin
            case (index)
+                8'd0: elu_rom = 16'sd0;
                8'd1: elu_rom = -16'sd8;
                8'd2: elu_rom = -16'sd16;
                8'd3: elu_rom = -16'sd23;
                8'd4: elu_rom = -16'sd30;
                8'd5: elu_rom = -16'sd37;
                8'd6: elu_rom = -16'sd44;
                8'd7: elu_rom = -16'sd50;
                8'd8: elu_rom = -16'sd57;
                8'd9: elu_rom = -16'sd63;
                8'd10: elu_rom = -16'sd69;
                8'd11: elu_rom = -16'sd74;
                8'd12: elu_rom = -16'sd80;
                8'd13: elu_rom = -16'sd85;
                8'd14: elu_rom = -16'sd91;
                8'd15: elu_rom = -16'sd96;
                8'd16: elu_rom = -16'sd101;
                8'd17: elu_rom = -16'sd106;
                8'd18: elu_rom = -16'sd110;
                8'd19: elu_rom = -16'sd115;
                8'd20: elu_rom = -16'sd119;
                8'd21: elu_rom = -16'sd123;
                8'd22: elu_rom = -16'sd127;
                8'd23: elu_rom = -16'sd131;
                8'd24: elu_rom = -16'sd135;
                8'd25: elu_rom = -16'sd139;
                8'd26: elu_rom = -16'sd142;
                8'd27: elu_rom = -16'sd146;
                8'd28: elu_rom = -16'sd149;
                8'd29: elu_rom = -16'sd153;
                8'd30: elu_rom = -16'sd156;
                8'd31: elu_rom = -16'sd159;
                8'd32: elu_rom = -16'sd162;
                8'd33: elu_rom = -16'sd165;
                8'd34: elu_rom = -16'sd168;
                8'd35: elu_rom = -16'sd170;
                8'd36: elu_rom = -16'sd173;
                8'd37: elu_rom = -16'sd175;
                8'd38: elu_rom = -16'sd178;
                8'd39: elu_rom = -16'sd180;
                8'd40: elu_rom = -16'sd183;
                8'd41: elu_rom = -16'sd185;
                8'd42: elu_rom = -16'sd187;
                8'd43: elu_rom = -16'sd189;
                8'd44: elu_rom = -16'sd191;
                8'd45: elu_rom = -16'sd193;
                8'd46: elu_rom = -16'sd195;
                8'd47: elu_rom = -16'sd197;
                8'd48: elu_rom = -16'sd199;
                8'd49: elu_rom = -16'sd201;
                8'd50: elu_rom = -16'sd202;
                8'd51: elu_rom = -16'sd204;
                8'd52: elu_rom = -16'sd206;
                8'd53: elu_rom = -16'sd207;
                8'd54: elu_rom = -16'sd209;
                8'd55: elu_rom = -16'sd210;
                8'd56: elu_rom = -16'sd212;
                8'd57: elu_rom = -16'sd213;
                8'd58: elu_rom = -16'sd214;
                8'd59: elu_rom = -16'sd215;
                8'd60: elu_rom = -16'sd217;
                8'd61: elu_rom = -16'sd218;
                8'd62: elu_rom = -16'sd219;
                8'd63: elu_rom = -16'sd220;
                8'd64: elu_rom = -16'sd221;
                8'd65: elu_rom = -16'sd222;
                8'd66: elu_rom = -16'sd223;
                8'd67: elu_rom = -16'sd224;
                8'd68: elu_rom = -16'sd225;
                8'd69: elu_rom = -16'sd226;
                8'd70: elu_rom = -16'sd227;
                8'd71: elu_rom = -16'sd228;
                8'd72: elu_rom = -16'sd229;
                8'd73: elu_rom = -16'sd230;
                8'd74: elu_rom = -16'sd231;
                8'd75: elu_rom = -16'sd231;
                8'd76: elu_rom = -16'sd232;
                8'd77: elu_rom = -16'sd233;
                8'd78: elu_rom = -16'sd234;
                8'd79: elu_rom = -16'sd234;
                8'd80: elu_rom = -16'sd235;
                8'd81: elu_rom = -16'sd236;
                8'd82: elu_rom = -16'sd236;
                8'd83: elu_rom = -16'sd237;
                8'd84: elu_rom = -16'sd237;
                8'd85: elu_rom = -16'sd238;
                8'd86: elu_rom = -16'sd239;
                8'd87: elu_rom = -16'sd239;
                8'd88: elu_rom = -16'sd240;
                8'd89: elu_rom = -16'sd240;
                8'd90: elu_rom = -16'sd241;
                8'd91: elu_rom = -16'sd241;
                8'd92: elu_rom = -16'sd242;
                8'd93: elu_rom = -16'sd242;
                8'd94: elu_rom = -16'sd242;
                8'd95: elu_rom = -16'sd243;
                8'd96: elu_rom = -16'sd243;
                8'd97: elu_rom = -16'sd244;
                8'd98: elu_rom = -16'sd244;
                8'd99: elu_rom = -16'sd244;
                8'd100: elu_rom = -16'sd245;
                8'd101: elu_rom = -16'sd245;
                8'd102: elu_rom = -16'sd245;
                8'd103: elu_rom = -16'sd246;
                8'd104: elu_rom = -16'sd246;
                8'd105: elu_rom = -16'sd246;
                8'd106: elu_rom = -16'sd247;
                8'd107: elu_rom = -16'sd247;
                8'd108: elu_rom = -16'sd247;
                8'd109: elu_rom = -16'sd248;
                8'd110: elu_rom = -16'sd248;
                8'd111: elu_rom = -16'sd248;
                8'd112: elu_rom = -16'sd248;
                8'd113: elu_rom = -16'sd249;
                8'd114: elu_rom = -16'sd249;
                8'd115: elu_rom = -16'sd249;
                8'd116: elu_rom = -16'sd249;
                8'd117: elu_rom = -16'sd249;
                8'd118: elu_rom = -16'sd250;
                8'd119: elu_rom = -16'sd250;
                8'd120: elu_rom = -16'sd250;
                8'd121: elu_rom = -16'sd250;
                8'd122: elu_rom = -16'sd250;
                8'd123: elu_rom = -16'sd251;
                8'd124: elu_rom = -16'sd251;
                8'd125: elu_rom = -16'sd251;
                8'd126: elu_rom = -16'sd251;
                8'd127: elu_rom = -16'sd251;
                8'd128: elu_rom = -16'sd251;
                8'd129: elu_rom = -16'sd251;
                8'd130: elu_rom = -16'sd252;
                8'd131: elu_rom = -16'sd252;
                8'd132: elu_rom = -16'sd252;
                8'd133: elu_rom = -16'sd252;
                8'd134: elu_rom = -16'sd252;
                8'd135: elu_rom = -16'sd252;
                8'd136: elu_rom = -16'sd252;
                8'd137: elu_rom = -16'sd252;
                8'd138: elu_rom = -16'sd253;
                8'd139: elu_rom = -16'sd253;
                8'd140: elu_rom = -16'sd253;
                8'd141: elu_rom = -16'sd253;
                8'd142: elu_rom = -16'sd253;
                8'd143: elu_rom = -16'sd253;
                8'd144: elu_rom = -16'sd253;
                8'd145: elu_rom = -16'sd253;
                8'd146: elu_rom = -16'sd253;
                8'd147: elu_rom = -16'sd253;
                8'd148: elu_rom = -16'sd253;
                8'd149: elu_rom = -16'sd254;
                8'd150: elu_rom = -16'sd254;
                8'd151: elu_rom = -16'sd254;
                8'd152: elu_rom = -16'sd254;
                8'd153: elu_rom = -16'sd254;
                8'd154: elu_rom = -16'sd254;
                8'd155: elu_rom = -16'sd254;
                8'd156: elu_rom = -16'sd254;
                8'd157: elu_rom = -16'sd254;
                8'd158: elu_rom = -16'sd254;
                8'd159: elu_rom = -16'sd254;
                8'd160: elu_rom = -16'sd254;
                8'd161: elu_rom = -16'sd254;
                8'd162: elu_rom = -16'sd254;
                8'd163: elu_rom = -16'sd254;
                8'd164: elu_rom = -16'sd254;
                8'd165: elu_rom = -16'sd255;
                8'd166: elu_rom = -16'sd255;
                8'd167: elu_rom = -16'sd255;
                8'd168: elu_rom = -16'sd255;
                8'd169: elu_rom = -16'sd255;
                8'd170: elu_rom = -16'sd255;
                8'd171: elu_rom = -16'sd255;
                8'd172: elu_rom = -16'sd255;
                8'd173: elu_rom = -16'sd255;
                8'd174: elu_rom = -16'sd255;
                8'd175: elu_rom = -16'sd255;
                8'd176: elu_rom = -16'sd255;
                8'd177: elu_rom = -16'sd255;
                8'd178: elu_rom = -16'sd255;
                8'd179: elu_rom = -16'sd255;
                8'd180: elu_rom = -16'sd255;
                8'd181: elu_rom = -16'sd255;
                8'd182: elu_rom = -16'sd255;
                8'd183: elu_rom = -16'sd255;
                8'd184: elu_rom = -16'sd255;
                8'd185: elu_rom = -16'sd255;
                8'd186: elu_rom = -16'sd255;
                8'd187: elu_rom = -16'sd255;
                8'd188: elu_rom = -16'sd255;
                8'd189: elu_rom = -16'sd255;
                8'd190: elu_rom = -16'sd255;
                8'd191: elu_rom = -16'sd255;
                8'd192: elu_rom = -16'sd255;
                8'd193: elu_rom = -16'sd255;
                8'd194: elu_rom = -16'sd255;
                8'd195: elu_rom = -16'sd255;
                8'd196: elu_rom = -16'sd255;
                8'd197: elu_rom = -16'sd255;
                8'd198: elu_rom = -16'sd255;
                8'd199: elu_rom = -16'sd255;
                8'd200: elu_rom = -16'sd256;
                8'd201: elu_rom = -16'sd256;
                8'd202: elu_rom = -16'sd256;
                8'd203: elu_rom = -16'sd256;
                8'd204: elu_rom = -16'sd256;
                8'd205: elu_rom = -16'sd256;
                8'd206: elu_rom = -16'sd256;
                8'd207: elu_rom = -16'sd256;
                8'd208: elu_rom = -16'sd256;
                8'd209: elu_rom = -16'sd256;
                8'd210: elu_rom = -16'sd256;
                8'd211: elu_rom = -16'sd256;
                8'd212: elu_rom = -16'sd256;
                8'd213: elu_rom = -16'sd256;
                8'd214: elu_rom = -16'sd256;
                8'd215: elu_rom = -16'sd256;
                8'd216: elu_rom = -16'sd256;
                8'd217: elu_rom = -16'sd256;
                8'd218: elu_rom = -16'sd256;
                8'd219: elu_rom = -16'sd256;
                8'd220: elu_rom = -16'sd256;
                8'd221: elu_rom = -16'sd256;
                8'd222: elu_rom = -16'sd256;
                8'd223: elu_rom = -16'sd256;
                8'd224: elu_rom = -16'sd256;
                8'd225: elu_rom = -16'sd256;
                8'd226: elu_rom = -16'sd256;
                8'd227: elu_rom = -16'sd256;
                8'd228: elu_rom = -16'sd256;
                8'd229: elu_rom = -16'sd256;
                8'd230: elu_rom = -16'sd256;
                8'd231: elu_rom = -16'sd256;
                8'd232: elu_rom = -16'sd256;
                8'd233: elu_rom = -16'sd256;
                8'd234: elu_rom = -16'sd256;
                8'd235: elu_rom = -16'sd256;
                8'd236: elu_rom = -16'sd256;
                8'd237: elu_rom = -16'sd256;
                8'd238: elu_rom = -16'sd256;
                8'd239: elu_rom = -16'sd256;
                8'd240: elu_rom = -16'sd256;
                8'd241: elu_rom = -16'sd256;
                8'd242: elu_rom = -16'sd256;
                8'd243: elu_rom = -16'sd256;
                8'd244: elu_rom = -16'sd256;
                8'd245: elu_rom = -16'sd256;
                8'd246: elu_rom = -16'sd256;
                8'd247: elu_rom = -16'sd256;
                8'd248: elu_rom = -16'sd256;
                8'd249: elu_rom = -16'sd256;
                8'd250: elu_rom = -16'sd256;
                8'd251: elu_rom = -16'sd256;
                8'd252: elu_rom = -16'sd256;
                8'd253: elu_rom = -16'sd256;
                8'd254: elu_rom = -16'sd256;
                8'd255: elu_rom = -16'sd256;
                default: elu_rom = -16'sd256;
            endcase
        end
    endfunction

    reg valid_s1;
    reg valid_s2;
    reg signed [15:0] pass_s1 [0:7];
    reg signed [15:0] pass_s2 [0:7];
    reg [7:0] index_s1 [0:7];
    reg [2:0] remainder_s1 [0:7];
    reg [2:0] remainder_s2 [0:7];
    reg nonnegative_s1 [0:7];
    reg nonnegative_s2 [0:7];
    reg saturated_s1 [0:7];
    reg saturated_s2 [0:7];
    reg signed [15:0] base_s2 [0:7];
    reg signed [15:0] next_s2 [0:7];
    integer magnitude_tmp;
    integer delta_tmp;
    integer scaled_tmp;
    integer interpolated_tmp;

    always @(posedge clk) begin
        if (!reset_n) begin
            o_valid <= 1'b0;
            valid_s1 <= 1'b0;
            valid_s2 <= 1'b0;
            o_data <= 0;
        end else begin
            valid_s1 <= i_valid;
            valid_s2 <= valid_s1;
            o_valid <= valid_s2;

            if (i_valid) begin
                for (lane = 0; lane < 8; lane = lane + 1) begin
                    pass_s1[lane] <= i_data[lane*16 +: 16];
                    nonnegative_s1[lane] <= ($signed(i_data[lane*16 +: 16]) >= 0);
                    saturated_s1[lane] <= ($signed(i_data[lane*16 +: 16]) <= -16'sd2048);
                    magnitude_tmp = -$signed(i_data[lane*16 +: 16]);
                    index_s1[lane] <= magnitude_tmp[10:3];
                    remainder_s1[lane] <= magnitude_tmp[2:0];
                end
            end

            if (valid_s1) begin
                for (lane = 0; lane < 8; lane = lane + 1) begin
                    pass_s2[lane] <= pass_s1[lane];
                    nonnegative_s2[lane] <= nonnegative_s1[lane];
                    saturated_s2[lane] <= saturated_s1[lane];
                    remainder_s2[lane] <= remainder_s1[lane];
                    base_s2[lane] <= elu_rom(index_s1[lane]);
                    if (index_s1[lane] == 8'd255)
                        next_s2[lane] <= -16'sd256;
                    else
                        next_s2[lane] <= elu_rom(index_s1[lane] + 1'b1);
                end
            end

            if (valid_s2) begin
                for (lane = 0; lane < 8; lane = lane + 1) begin
                    delta_tmp = $signed(next_s2[lane]) - $signed(base_s2[lane]);
                    case (remainder_s2[lane])
                        0: scaled_tmp = 0;
                        1: scaled_tmp = delta_tmp;
                        2: scaled_tmp = delta_tmp <<< 1;
                        3: scaled_tmp = (delta_tmp <<< 1) + delta_tmp;
                        4: scaled_tmp = delta_tmp <<< 2;
                        5: scaled_tmp = (delta_tmp <<< 2) + delta_tmp;
                        6: scaled_tmp = (delta_tmp <<< 2) + (delta_tmp <<< 1);
                        default: scaled_tmp = (delta_tmp <<< 3) - delta_tmp;
                    endcase
                    interpolated_tmp = $signed(base_s2[lane]) + ((scaled_tmp + 4) >>> 3);
                    if (nonnegative_s2[lane])
                        o_data[lane*16 +: 16] <= pass_s2[lane];
                    else if (saturated_s2[lane])
                        o_data[lane*16 +: 16] <= -16'sd256;
                    else
                        o_data[lane*16 +: 16] <= interpolated_tmp[15:0];
                end
            end
        end
    end
endmodule
