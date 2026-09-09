`timescale 1ns/1ps
`include "rl_accel_params.vh"

// Measures the exact cycle cost of a single-operator sequence for a sweep of
// dimensions.  The portability analyzer's latency model is fitted to and
// validated against these numbers rather than hand-derived from the FSMs.
module tb_cycle_model;
    reg clk = 0;
    always #5 clk = ~clk;
    reg reset_n = 0;

    reg instruction_we = 0;
    reg [4:0] instruction_addr = 0;
    reg [127:0] instruction_wdata = 0;
    reg sequence_start = 0;
    reg [5:0] instruction_count = 1;
    wire sequence_busy, sequence_done, sequence_error;
    wire [4:0] sequence_index;
    wire cmd_start;
    wire [2:0] cmd_opcode;
    wire [8:0] cmd_src_base, cmd_dst_base, cmd_aux0_base, cmd_aux1_base;
    wire [10:0] cmd_weight_base;
    wire [15:0] cmd_dim_k, cmd_dim_n;
    wire [5:0] cmd_shift;
    wire busy, done, error;

    integer cycle_counter = 0;
    integer start_cycle;
    integer k, n;
    always @(posedge clk) cycle_counter <= cycle_counter + 1;

    rl_instruction_sequencer u_sequencer (
        .clk(clk), .reset_n(reset_n),
        .host_wr_en(instruction_we), .host_addr(instruction_addr),
        .host_wr_data(instruction_wdata), .host_rd_en(1'b0),
        .host_rd_valid(), .host_rd_data(),
        .start(sequence_start), .instruction_count(instruction_count),
        .busy(sequence_busy), .done(sequence_done), .error(sequence_error),
        .current_index(sequence_index), .cmd_start(cmd_start),
        .cmd_opcode(cmd_opcode), .cmd_src_base(cmd_src_base),
        .cmd_dst_base(cmd_dst_base), .cmd_aux0_base(cmd_aux0_base),
        .cmd_aux1_base(cmd_aux1_base), .cmd_weight_base(cmd_weight_base),
        .cmd_dim_k(cmd_dim_k), .cmd_dim_n(cmd_dim_n), .cmd_shift(cmd_shift),
        .core_busy(busy), .core_done(done), .core_error(error)
    );

    rl_accel_core u_dut (
        .clk(clk), .reset_n(reset_n), .cmd_start(cmd_start),
        .cmd_opcode(cmd_opcode), .cmd_src_base(cmd_src_base),
        .cmd_dst_base(cmd_dst_base), .cmd_aux0_base(cmd_aux0_base),
        .cmd_aux1_base(cmd_aux1_base), .cmd_weight_base(cmd_weight_base),
        .cmd_dim_k(cmd_dim_k), .cmd_dim_n(cmd_dim_n), .cmd_shift(cmd_shift),
        .busy(busy), .done(done), .error(error),
        .host_vec_wr_en(1'b0), .host_vec_addr(9'b0),
        .host_vec_wr_data(128'b0), .host_vec_rd_en(1'b0),
        .host_vec_rd_valid(), .host_vec_rd_data(),
        .host_wt_wr_en(1'b0), .host_wt_bank(3'b0), .host_wt_addr(11'b0),
        .host_wt_wr_data(128'b0), .host_wt_rd_en(1'b0),
        .host_wt_rd_valid(), .host_wt_rd_data(),
        .cmd_weight_stream(1'b0),
        .cmd_weight_word({`RL_WEIGHT_STREAM_WORD_W{1'b0}}),
        .weight_ddr_base(32'b0),
        .m_axi_araddr(), .m_axi_arlen(), .m_axi_arsize(), .m_axi_arburst(),
        .m_axi_arvalid(), .m_axi_arready(1'b0), .m_axi_rdata(64'b0),
        .m_axi_rresp(2'b00), .m_axi_rlast(1'b0), .m_axi_rvalid(1'b0),
        .m_axi_rready()
    );

    task load_one;
        input [2:0] opcode;
        input [15:0] dim_k;
        input [15:0] dim_n;
        begin
            @(negedge clk);
            instruction_addr = 0;
            // word0: opcode | src<<3 | dst<<12 | aux0<<21, all bases zero here.
            // word2 (bits 95:64): dim_k | dim_n<<16
            instruction_wdata = {32'd0, {dim_n, dim_k}, 32'd0, {29'd0, opcode}};
            instruction_we = 1;
            @(negedge clk);
            instruction_we = 0;
            instruction_count = 1;
        end
    endtask

    task run_and_report;
        input [127:0] label;
        input integer dim_k;
        input integer dim_n;
        begin
            @(negedge clk);
            sequence_start = 1;
            @(negedge clk);
            sequence_start = 0;
            start_cycle = cycle_counter;
            wait (sequence_done == 1'b1);
            @(posedge clk);
            $display("CYCLES %0s k=%0d n=%0d cycles=%0d err=%0d",
                     label, dim_k, dim_n, cycle_counter - start_cycle, sequence_error);
        end
    endtask

    initial begin
        repeat (4) @(negedge clk);
        reset_n = 1;
        repeat (4) @(negedge clk);

        // GEMM sweep: cost should be affine in ceil(k/8) and ceil(n/8).
        for (k = 8; k <= 64; k = k + 8) begin
            load_one(`RL_OP_GEMM, k[15:0], 16'd8);
            run_and_report("GEMM", k, 8);
        end
        for (n = 8; n <= 64; n = n + 8) begin
            load_one(`RL_OP_GEMM, 16'd8, n[15:0]);
            run_and_report("GEMM", 8, n);
        end
        load_one(`RL_OP_GEMM, 16'd125, 16'd128);
        run_and_report("GEMM", 125, 128);

        // ELU / NORM / CONCAT sweeps.
        for (n = 8; n <= 64; n = n + 8) begin
            load_one(`RL_OP_ELU, 16'd0, n[15:0]);
            run_and_report("ELU", 0, n);
        end
        for (n = 8; n <= 64; n = n + 8) begin
            load_one(`RL_OP_NORM, 16'd0, n[15:0]);
            run_and_report("NORM", 0, n);
        end
        for (n = 8; n <= 64; n = n + 8) begin
            load_one(`RL_OP_CONCAT, 16'd25, n[15:0]);
            run_and_report("CONCAT", 25, n);
        end
        load_one(`RL_OP_CONCAT, 16'd25, 16'd3);
        run_and_report("CONCAT", 25, 3);

        // Multi-instruction sequences: separates the once-per-sequence cost
        // from the per-instruction dispatch cost.
        for (k = 1; k <= 4; k = k + 1) begin
            @(negedge clk);
            for (n = 0; n < k; n = n + 1) begin
                instruction_addr = n[4:0];
                instruction_wdata = {32'd0, {16'd8, 16'd8}, 32'd0, {29'd0, `RL_OP_GEMM}};
                instruction_we = 1;
                @(negedge clk);
            end
            instruction_we = 0;
            instruction_count = k[5:0];
            run_and_report("MULTIGEMM", k, 8);
        end

        // CONCAT left-operand sweep at fixed right operand.
        for (k = 8; k <= 64; k = k + 8) begin
            load_one(`RL_OP_CONCAT, k[15:0], 16'd8);
            run_and_report("CONCATL", k, 8);
        end
        $display("PASS tb_cycle_model");
        $finish;
    end
endmodule
