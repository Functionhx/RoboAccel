`timescale 1ns/1ps

module tb_primitives;
    reg clk = 0;
    always #5 clk = ~clk;

    reg reset_n = 0;

    reg mac_valid;
    reg [127:0] mac_act;
    reg [1023:0] mac_weight;
    wire mac_out_valid;
    wire [383:0] mac_sum;

    reg elu_valid;
    reg [127:0] elu_in;
    wire elu_out_valid;
    wire [127:0] elu_out;

    reg signed [47:0] round_in;
    reg signed [15:0] round_bias;
    reg [5:0] round_shift;
    wire signed [15:0] round_out;

    integer i;
    integer j;
    integer errors = 0;

    rl_mac_array_8x8 u_mac (
        .clk(clk), .reset_n(reset_n), .i_valid(mac_valid),
        .i_activations(mac_act), .i_weights(mac_weight),
        .o_valid(mac_out_valid), .o_row_sums(mac_sum)
    );

    rl_elu_array8 u_elu (
        .clk(clk), .reset_n(reset_n), .i_valid(elu_valid),
        .i_data(elu_in), .o_valid(elu_out_valid), .o_data(elu_out)
    );


    rl_round_shift_sat16 u_round (
        .i_value(round_in), .i_bias(round_bias),
        .i_shift(round_shift), .o_value(round_out)
    );

    task check_equal;
        input signed [63:0] got;
        input signed [63:0] expected;
        input [255:0] label;
        begin
            if (got !== expected) begin
                $display("FAIL %0s: got %0d expected %0d", label, got, expected);
                errors = errors + 1;
            end
        end
    endtask

    initial begin
        mac_valid = 0;
        mac_act = 0;
        mac_weight = 0;
        elu_valid = 0;
        elu_in = 0;
        round_in = 0;
        round_bias = 0;
        round_shift = 0;

        repeat (3) @(posedge clk);
        reset_n <= 1;
        @(posedge clk);

        // Round/shift/saturation.
        round_in = 48'sd1638400; // 100 * 2^14
        round_bias = -16'sd3;
        round_shift = 14;
        #1 check_equal(round_out, 97, "round_shift");
        round_in = 48'sd1073741824;
        #1 check_equal(round_out, 32767, "positive_saturation");

        // One 8x8 MAC packet. Activations are 1..8. Row j has weight j+1.
        for (i = 0; i < 8; i = i + 1)
            mac_act[i*16 +: 16] = i + 1;
        for (j = 0; j < 8; j = j + 1)
            for (i = 0; i < 8; i = i + 1)
                mac_weight[j*128 + i*16 +: 16] = j + 1;
        mac_valid = 1;
        @(posedge clk);
        #1 mac_valid = 0;
        wait (mac_out_valid);
        #1;
        for (j = 0; j < 8; j = j + 1)
            check_equal($signed(mac_sum[j*48 +: 48]), 36*(j+1), "mac_row_sum");

        // Q8.8 ELU samples: positive passthrough, ELU(-1) ~= -0.6321.
        elu_in[0*16 +: 16] = 16'sd256;
        elu_in[1*16 +: 16] = 16'sd0;
        elu_in[2*16 +: 16] = -16'sd128;
        elu_in[3*16 +: 16] = -16'sd256;
        elu_in[4*16 +: 16] = -16'sd512;
        elu_in[5*16 +: 16] = -16'sd1024;
        elu_in[6*16 +: 16] = -16'sd2048;
        elu_in[7*16 +: 16] = -16'sd4096;
        elu_valid = 1;
        @(posedge clk);
        #1 elu_valid = 0;
        wait (elu_out_valid);
        #1;
        check_equal($signed(elu_out[0*16 +: 16]), 256, "elu_positive");
        check_equal($signed(elu_out[1*16 +: 16]), 0, "elu_zero");
        check_equal($signed(elu_out[3*16 +: 16]), -162, "elu_minus_one");
        check_equal($signed(elu_out[6*16 +: 16]), -256, "elu_minus_eight");

        // Raw input/mean are Q10.6. The folded coefficient is
        // inv_std*2^(8-6)=4.0, encoded as E5M11 e=8,m=1024.
        for (i = 0; i < 8; i = i + 1) begin
        end
        @(posedge clk);
        #1;
        for (i = 0; i < 8; i = i + 1)

        if (errors == 0)
            $display("PASS tb_primitives");
        else
            $fatal(1, "tb_primitives failed with %0d errors", errors);
        $finish;
    end
endmodule
