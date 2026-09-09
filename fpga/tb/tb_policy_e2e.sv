`timescale 1ns/1ps
`include "rl_accel_params.vh"

module tb_policy_e2e;
    reg clk = 0;
    always #5 clk = ~clk;
    reg reset_n = 0;

    reg instruction_we = 0;
    reg instruction_re = 0;
    reg [4:0] instruction_addr = 0;
    reg [127:0] instruction_wdata = 0;
    wire instruction_rvalid;
    wire [127:0] instruction_rdata;
    reg sequence_start = 0;
    wire sequence_busy;
    wire sequence_done;
    wire sequence_error;
    wire [4:0] sequence_index;
    wire cmd_start;
    wire [2:0] cmd_opcode;
    wire [8:0] cmd_src_base;
    wire [8:0] cmd_dst_base;
    wire [8:0] cmd_aux0_base;
    wire [8:0] cmd_aux1_base;
    wire [10:0] cmd_weight_base;
    wire [15:0] cmd_dim_k;
    wire [15:0] cmd_dim_n;
    wire [5:0] cmd_shift;
    wire busy;
    wire done;
    wire error;

    reg host_vec_we = 0;
    reg [8:0] host_vec_addr = 0;
    reg [127:0] host_vec_wdata = 0;
    reg host_vec_re = 0;
    wire host_vec_rvalid;
    wire [127:0] host_vec_rdata;

    reg [127:0] selftest_obs [0:3];
    reg [127:0] selftest_history [0:15];
    reg [127:0] selftest_actions [0:0];
    integer cycle_counter = 0;
    integer start_cycle;
    integer group;
    integer lane;
    integer errors = 0;

    always @(posedge clk)
        cycle_counter <= cycle_counter + 1;

    rl_instruction_sequencer u_sequencer (
        .clk(clk), .reset_n(reset_n),
        .host_wr_en(instruction_we), .host_addr(instruction_addr),
        .host_wr_data(instruction_wdata), .host_rd_en(instruction_re),
        .host_rd_valid(instruction_rvalid), .host_rd_data(instruction_rdata),
        .start(sequence_start), .instruction_count(6'd13),
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
        .host_vec_wr_en(host_vec_we), .host_vec_addr(host_vec_addr),
        .host_vec_wr_data(host_vec_wdata), .host_vec_rd_en(host_vec_re),
        .host_vec_rd_valid(host_vec_rvalid), .host_vec_rd_data(host_vec_rdata),
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

    task vec_write;
        input [8:0] address;
        input [127:0] data;
        begin
            @(negedge clk);
            host_vec_addr = address;
            host_vec_wdata = data;
            host_vec_we = 1;
            @(negedge clk);
            host_vec_we = 0;
        end
    endtask

    function automatic [127:0] descriptor;
        input [2:0] opcode;
        input [8:0] src;
        input [8:0] dst;
        input [8:0] aux0;
        input [8:0] aux1;
        input [10:0] wt_base;
        input [15:0] dim_k;
        input [15:0] dim_n;
        input [5:0] shift;
        reg [127:0] descriptor_bits;
        begin
            descriptor_bits = 0;
            descriptor_bits[2:0] = opcode;
            descriptor_bits[11:3] = src;
            descriptor_bits[20:12] = dst;
            descriptor_bits[29:21] = aux0;
            descriptor_bits[40:32] = aux1;
            descriptor_bits[51:41] = wt_base;
            descriptor_bits[57:52] = shift;
            descriptor_bits[79:64] = dim_k;
            descriptor_bits[95:80] = dim_n;
            descriptor = descriptor_bits;
        end
    endfunction

    task instruction_write;
        input [4:0] address;
        input [127:0] data;
        begin
            @(negedge clk);
            instruction_addr = address;
            instruction_wdata = data;
            instruction_we = 1;
            @(negedge clk);
            instruction_we = 0;
        end
    endtask

    initial begin
        $readmemh("../generated/vector_cache.hex", u_dut.u_vector_cache.mem);
        $readmemh("../generated/weight_bank0_main.hex", u_dut.u_weight_cache.g_bank[0].mem_main);
        $readmemh("../generated/weight_bank0_tail.hex", u_dut.u_weight_cache.g_bank[0].mem_tail);
        $readmemh("../generated/weight_bank1_main.hex", u_dut.u_weight_cache.g_bank[1].mem_main);
        $readmemh("../generated/weight_bank1_tail.hex", u_dut.u_weight_cache.g_bank[1].mem_tail);
        $readmemh("../generated/weight_bank2_main.hex", u_dut.u_weight_cache.g_bank[2].mem_main);
        $readmemh("../generated/weight_bank2_tail.hex", u_dut.u_weight_cache.g_bank[2].mem_tail);
        $readmemh("../generated/weight_bank3_main.hex", u_dut.u_weight_cache.g_bank[3].mem_main);
        $readmemh("../generated/weight_bank3_tail.hex", u_dut.u_weight_cache.g_bank[3].mem_tail);
        $readmemh("../generated/weight_bank4_main.hex", u_dut.u_weight_cache.g_bank[4].mem_main);
        $readmemh("../generated/weight_bank4_tail.hex", u_dut.u_weight_cache.g_bank[4].mem_tail);
        $readmemh("../generated/weight_bank5_main.hex", u_dut.u_weight_cache.g_bank[5].mem_main);
        $readmemh("../generated/weight_bank5_tail.hex", u_dut.u_weight_cache.g_bank[5].mem_tail);
        $readmemh("../generated/weight_bank6_main.hex", u_dut.u_weight_cache.g_bank[6].mem_main);
        $readmemh("../generated/weight_bank6_tail.hex", u_dut.u_weight_cache.g_bank[6].mem_tail);
        $readmemh("../generated/weight_bank7_main.hex", u_dut.u_weight_cache.g_bank[7].mem_main);
        $readmemh("../generated/weight_bank7_tail.hex", u_dut.u_weight_cache.g_bank[7].mem_tail);
        $readmemh("../generated/selftest_input0.hex", selftest_obs);
        $readmemh("../generated/selftest_input1.hex", selftest_history);
        $readmemh("../generated/selftest_actions.hex", selftest_actions);

        repeat (4) @(posedge clk);
        reset_n <= 1;
        repeat (2) @(posedge clk);
        for (group = 0; group < 4; group = group + 1)
            vec_write(group[8:0], selftest_obs[group]);
        for (group = 0; group < 16; group = group + 1)
            vec_write(9'd4 + group[8:0], selftest_history[group]);

        instruction_write(0, descriptor(`RL_OP_GEMM, 4, 20, 36, 0, 0, 125, 128, 11));
        instruction_write(1, descriptor(`RL_OP_ELU, 20, 20, 0, 0, 0, 0, 128, 0));
        instruction_write(2, descriptor(`RL_OP_GEMM, 20, 52, 60, 0, 256, 128, 64, 12));
        instruction_write(3, descriptor(`RL_OP_ELU, 52, 52, 0, 0, 0, 0, 64, 0));
        instruction_write(4, descriptor(`RL_OP_GEMM, 52, 68, 69, 0, 384, 64, 3, 14));
        instruction_write(5, descriptor(`RL_OP_CONCAT, 0, 70, 68, 0, 0, 25, 3, 0));
        instruction_write(6, descriptor(`RL_OP_GEMM, 70, 74, 90, 0, 392, 28, 128, 13));
        instruction_write(7, descriptor(`RL_OP_ELU, 74, 74, 0, 0, 0, 0, 128, 0));
        instruction_write(8, descriptor(`RL_OP_GEMM, 74, 106, 114, 0, 456, 128, 64, 14));
        instruction_write(9, descriptor(`RL_OP_ELU, 106, 106, 0, 0, 0, 0, 64, 0));
        instruction_write(10, descriptor(`RL_OP_GEMM, 106, 122, 126, 0, 584, 64, 32, 14));
        instruction_write(11, descriptor(`RL_OP_ELU, 122, 122, 0, 0, 0, 0, 32, 0));
        instruction_write(12, descriptor(`RL_OP_GEMM, 122, 130, 131, 0, 616, 32, 6, 14));

        @(negedge clk);
        start_cycle = cycle_counter;
        sequence_start = 1;
        @(negedge clk);
        sequence_start = 0;
        wait(sequence_done);
        if (sequence_error) begin
            $display("FAIL automatic policy sequence reported error at %0d",
                     sequence_index);
            errors = errors + 1;
        end
        $display("Policy automatic sequence cycles: %0d",
                 cycle_counter - start_cycle);

        @(negedge clk);
        host_vec_addr = 130;
        host_vec_re = 1;
        @(negedge clk);
        host_vec_re = 0;
        wait(host_vec_rvalid);
        #1;
        for (lane = 0; lane < 6; lane = lane + 1) begin
            if ($signed(host_vec_rdata[lane*16 +: 16]) !==
                $signed(selftest_actions[0][lane*16 +: 16])) begin
                $display("FAIL action %0d got %0d expected %0d", lane,
                    $signed(host_vec_rdata[lane*16 +: 16]),
                    $signed(selftest_actions[0][lane*16 +: 16]));
                errors = errors + 1;
            end
        end

        if (errors == 0)
            $display("PASS tb_policy_e2e");
        else
            $fatal(1, "tb_policy_e2e failed with %0d errors", errors);
        $finish;
    end
endmodule
