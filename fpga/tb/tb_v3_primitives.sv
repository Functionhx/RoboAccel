`timescale 1ns/1ps
`include "rl_accel_params.vh"

// Verifies every v3 primitive against a golden produced independently in
// tools/gen_v3_vectors.py: VECTOR affine / mul / clip / reductions, all four
// SFU curves, and the unaligned SLICE that CONCAT gained.
module tb_v3_primitives;
    reg clk = 0;
    always #5 clk = ~clk;
    reg reset_n = 0;

    reg  [2:0]   cmd_opcode = 0;
    reg  [8:0]   cmd_src = 0, cmd_dst = 0, cmd_aux0 = 0, cmd_aux1 = 0;
    reg  [15:0]  cmd_dim_k = 0, cmd_dim_n = 0;
    reg  [5:0]   cmd_shift = 0, cmd_subop = 0;
    reg          cmd_start = 0;
    wire         busy, done, error;

    reg          host_we = 0, host_re = 0;
    reg  [8:0]   host_addr = 0;
    reg  [127:0] host_wdata = 0;
    wire         host_rvalid;
    wire [127:0] host_rdata;

    rl_accel_core u_dut (
        .clk(clk), .reset_n(reset_n), .cmd_start(cmd_start),
        .cmd_opcode(cmd_opcode), .cmd_src_base(cmd_src), .cmd_dst_base(cmd_dst),
        .cmd_aux0_base(cmd_aux0), .cmd_aux1_base(cmd_aux1),
        .cmd_weight_base(11'd0), .cmd_dim_k(cmd_dim_k), .cmd_dim_n(cmd_dim_n),
        .cmd_shift(cmd_shift), .cmd_subop(cmd_subop),
        .cmd_weight_stream(1'b0),
        .cmd_weight_word({`RL_WEIGHT_STREAM_WORD_W{1'b0}}), .weight_ddr_base(32'b0),
        .busy(busy), .done(done), .error(error),
        .host_vec_wr_en(host_we), .host_vec_addr(host_addr),
        .host_vec_wr_data(host_wdata), .host_vec_rd_en(host_re),
        .host_vec_rd_valid(host_rvalid), .host_vec_rd_data(host_rdata),
        .host_wt_wr_en(1'b0), .host_wt_bank(3'b0), .host_wt_addr(11'b0),
        .host_wt_wr_data(128'b0), .host_wt_rd_en(1'b0),
        .host_wt_rd_valid(), .host_wt_rd_data(),
        .m_axi_araddr(), .m_axi_arlen(), .m_axi_arsize(), .m_axi_arburst(),
        .m_axi_arvalid(), .m_axi_arready(1'b0), .m_axi_rdata(64'b0),
        .m_axi_rresp(2'b00), .m_axi_rlast(1'b0), .m_axi_rvalid(1'b0),
        .m_axi_rready()
    );

    reg [127:0] vin  [0:63];
    reg [127:0] vexp [0:31];
    reg [127:0] observed;
    integer c, lane, errors;
    localparam integer N_CASES = 9;

    task vec_write(input [8:0] a, input [127:0] d);
        begin
            @(negedge clk); host_addr = a; host_wdata = d; host_we = 1;
            @(negedge clk); host_we = 0;
        end
    endtask

    task vec_read(input [8:0] a);
        begin
            @(negedge clk); host_addr = a; host_re = 1;
            @(negedge clk); host_re = 0;
            wait (host_rvalid); observed = host_rdata; @(negedge clk);
        end
    endtask

    task run_cmd(input [2:0] op, input [5:0] sub, input [5:0] sh,
                 input [15:0] n, input [15:0] k);
        begin
            @(negedge clk);
            cmd_opcode = op; cmd_subop = sub; cmd_shift = sh;
            cmd_dim_n = n; cmd_dim_k = k;
            cmd_src = 9'd0; cmd_aux0 = 9'd1; cmd_aux1 = 9'd2; cmd_dst = 9'd8;
            cmd_start = 1;
            @(negedge clk);
            cmd_start = 0;
            wait (done);
            @(negedge clk);
        end
    endtask

    initial begin
        errors = 0;
        $readmemh("../generated_v3_in.hex", vin);
        $readmemh("../generated_v3_exp.hex", vexp);
        repeat (4) @(negedge clk);
        reset_n = 1;
        repeat (2) @(negedge clk);

        for (c = 0; c < N_CASES; c = c + 1) begin
            vec_write(9'd0, vin[c*3 + 0]);
            vec_write(9'd1, vin[c*3 + 1]);
            vec_write(9'd2, vin[c*3 + 2]);
            if (c < 5)
                run_cmd(`RL_OP_VECTOR, c[5:0], (c == 0) ? 6'd6 :
                        (c == 1) ? 6'd8 : (c == 4) ? 6'd4 : 6'd0, 16'd8, 16'd0);
            else
                run_cmd(`RL_OP_SFU, (c - 5), 6'd0, 16'd8, 16'd0);
            vec_read(9'd8);
            for (lane = 0; lane < 8; lane = lane + 1)
                if (observed[lane*16 +: 16] !== vexp[c][lane*16 +: 16]) begin
                    $display("FAIL case %0d lane %0d got %0d expected %0d", c, lane,
                             $signed(observed[lane*16 +: 16]),
                             $signed(vexp[c][lane*16 +: 16]));
                    errors = errors + 1;
                end
        end

        // Unaligned SLICE: copy 5 elements starting at element 3 of word 0.
        vec_write(9'd0, 128'h0008_0007_0006_0005_0004_0003_0002_0001);
        vec_write(9'd1, 128'h0010_000f_000e_000d_000c_000b_000a_0009);
        @(negedge clk);
        cmd_opcode = `RL_OP_CONCAT; cmd_subop = 0; cmd_shift = 0;
        cmd_src = 9'd0; cmd_aux0 = 9'd1; cmd_aux1 = 9'd3;   // offset 3 elements
        cmd_dst = 9'd8; cmd_dim_k = 16'd5; cmd_dim_n = 16'd0;
        cmd_start = 1; @(negedge clk); cmd_start = 0;
        wait (done); @(negedge clk);
        vec_read(9'd8);
        for (lane = 0; lane < 5; lane = lane + 1)
            if ($signed(observed[lane*16 +: 16]) !== (lane + 4)) begin
                $display("FAIL slice lane %0d got %0d expected %0d", lane,
                         $signed(observed[lane*16 +: 16]), lane + 4);
                errors = errors + 1;
            end

        if (errors == 0) $display("PASS tb_v3_primitives");
        else $display("FAIL tb_v3_primitives errors=%0d", errors);
        $finish;
    end
endmodule
