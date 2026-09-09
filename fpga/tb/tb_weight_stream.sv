`timescale 1ns/1ps
`include "rl_accel_params.vh"

// RoboAccel-v2 end-to-end test: a policy whose weights do not fit on chip is
// executed straight out of a DDR model through the accelerator's AXI4 master,
// and its output is compared against the Python fixed-point golden.
//
// Selected at run time with plusargs so the same testbench covers Go2 and G1:
//   +dir=../generated_go2 +ninstr=7 +inwords=6 +actbase=230 +actwords=2
//   +nactions=12 +latency=30
module tb_weight_stream;
    reg clk = 0;
    always #5 clk = ~clk;
    reg reset_n = 0;

    localparam integer DDR_WORDS = 8192;      // 1 MiB of 1024-bit weight words

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
    wire cmd_weight_stream;
    wire [`RL_WEIGHT_STREAM_WORD_W-1:0] cmd_weight_word;
    wire busy, done, error;

    reg host_vec_we = 0;
    reg [8:0] host_vec_addr = 0;
    reg [127:0] host_vec_wdata = 0;
    reg host_vec_re = 0;
    wire host_vec_rvalid;
    wire [127:0] host_vec_rdata;

    // AXI4 read-only master under test.
    wire [31:0] m_araddr;
    wire [7:0]  m_arlen;
    wire [2:0]  m_arsize;
    wire [1:0]  m_arburst;
    wire        m_arvalid;
    reg         m_arready = 0;
    reg  [63:0] m_rdata = 0;
    reg  [1:0]  m_rresp = 0;
    reg         m_rlast = 0;
    reg         m_rvalid = 0;
    wire        m_rready;

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
        .cmd_weight_stream(cmd_weight_stream), .cmd_weight_word(cmd_weight_word),
        .core_busy(busy), .core_done(done), .core_error(error)
    );

    rl_accel_core u_dut (
        .clk(clk), .reset_n(reset_n), .cmd_start(cmd_start),
        .cmd_opcode(cmd_opcode), .cmd_src_base(cmd_src_base),
        .cmd_dst_base(cmd_dst_base), .cmd_aux0_base(cmd_aux0_base),
        .cmd_aux1_base(cmd_aux1_base), .cmd_weight_base(cmd_weight_base),
        .cmd_dim_k(cmd_dim_k), .cmd_dim_n(cmd_dim_n), .cmd_shift(cmd_shift),
        .cmd_weight_stream(cmd_weight_stream), .cmd_weight_word(cmd_weight_word),
        .weight_ddr_base(32'd0),
        .busy(busy), .done(done), .error(error),
        .host_vec_wr_en(host_vec_we), .host_vec_addr(host_vec_addr),
        .host_vec_wr_data(host_vec_wdata), .host_vec_rd_en(host_vec_re),
        .host_vec_rd_valid(host_vec_rvalid), .host_vec_rd_data(host_vec_rdata),
        .host_wt_wr_en(1'b0), .host_wt_bank(3'b0), .host_wt_addr(11'b0),
        .host_wt_wr_data(128'b0), .host_wt_rd_en(1'b0),
        .host_wt_rd_valid(), .host_wt_rd_data(),
        .m_axi_araddr(m_araddr), .m_axi_arlen(m_arlen), .m_axi_arsize(m_arsize),
        .m_axi_arburst(m_arburst), .m_axi_arvalid(m_arvalid),
        .m_axi_arready(m_arready), .m_axi_rdata(m_rdata), .m_axi_rresp(m_rresp),
        .m_axi_rlast(m_rlast), .m_axi_rvalid(m_rvalid), .m_axi_rready(m_rready)
    );

    // ---------------- DDR model ----------------
    reg [1023:0] ddr [0:DDR_WORDS-1];
    integer read_latency = 30;
    integer axi_bursts = 0;
    integer axi_beats = 0;

    reg [31:0] pending_addr [0:3];
    integer pending_count = 0;
    integer wr_ptr = 0;
    integer rd_ptr = 0;
    integer beat;
    integer wait_i;
    reg [31:0] serve_addr;

    // Address channel: accept an AR whenever the queue has room.
    always @(posedge clk) begin
        if (!reset_n) begin
            m_arready <= 1'b0;
            pending_count <= 0;
            wr_ptr <= 0;
        end else begin
            m_arready <= (pending_count < 3);
            if (m_arvalid && m_arready) begin
                if (m_arlen != 8'd15 || m_arsize != 3'b011 || m_arburst != 2'b01) begin
                    $display("FAIL unexpected burst shape len=%0d size=%0d burst=%0d",
                             m_arlen, m_arsize, m_arburst);
                    $finish;
                end
                if (m_araddr[6:0] != 7'b0) begin
                    $display("FAIL burst address 0x%08x is not 128-byte aligned", m_araddr);
                    $finish;
                end
                pending_addr[wr_ptr[1:0]] <= m_araddr;
                wr_ptr <= wr_ptr + 1;
                pending_count <= pending_count + 1;
                axi_bursts = axi_bursts + 1;
            end
        end
    end

    // Read data channel: serve queued bursts in order, after a fixed latency.
    initial begin
        m_rvalid = 0; m_rlast = 0; m_rresp = 0; m_rdata = 0;
        forever begin
            @(posedge clk);
            if (reset_n && pending_count > 0) begin
                for (wait_i = 0; wait_i < read_latency; wait_i = wait_i + 1)
                    @(posedge clk);
                serve_addr = pending_addr[rd_ptr[1:0]];
                for (beat = 0; beat < 16; beat = beat + 1) begin
                    m_rdata <= ddr[serve_addr[31:7]][beat*64 +: 64];
                    m_rresp <= 2'b00;
                    m_rlast <= (beat == 15);
                    m_rvalid <= 1'b1;
                    @(posedge clk);
                    axi_beats = axi_beats + 1;
                end
                m_rvalid <= 1'b0;
                m_rlast <= 1'b0;
                rd_ptr = rd_ptr + 1;
                pending_count = pending_count - 1;
            end
        end
    end

    // ---------------- stimulus ----------------
    reg [1023:0] dir_arg;
    reg [8*256-1:0] dir;
    integer ninstr, inwords, actbase, actwords, nactions;
    reg [127:0] program_words [0:31];
    reg [127:0] selftest_in [0:63];
    reg [127:0] golden [0:7];
    reg [127:0] observed;
    integer group, lane, errors, idx;
    integer cycle_counter = 0;
    integer start_cycle;
    reg [8*256-1:0] path;
    always @(posedge clk) cycle_counter <= cycle_counter + 1;

    task vec_write;
        input [8:0] address;
        input [127:0] data;
        begin
            @(negedge clk);
            host_vec_addr = address; host_vec_wdata = data; host_vec_we = 1;
            @(negedge clk);
            host_vec_we = 0;
        end
    endtask

    task vec_read;
        input [8:0] address;
        begin
            @(negedge clk);
            host_vec_addr = address; host_vec_re = 1;
            @(negedge clk);
            host_vec_re = 0;
            wait (host_vec_rvalid);
            observed = host_vec_rdata;
            @(negedge clk);
        end
    endtask

    initial begin
        errors = 0;
        if (!$value$plusargs("dir=%s", dir)) dir = "../generated_go2";
        if (!$value$plusargs("ninstr=%d", ninstr)) ninstr = 7;
        if (!$value$plusargs("inwords=%d", inwords)) inwords = 6;
        if (!$value$plusargs("actbase=%d", actbase)) actbase = 230;
        if (!$value$plusargs("actwords=%d", actwords)) actwords = 2;
        if (!$value$plusargs("nactions=%d", nactions)) nactions = 12;
        if (!$value$plusargs("latency=%d", read_latency)) read_latency = 30;

        $sformat(path, "%0s/vector_cache.hex", dir);
        $readmemh(path, u_dut.u_vector_cache.mem);
        $sformat(path, "%0s/weights_ddr.hex", dir);
        $readmemh(path, ddr);
        $sformat(path, "%0s/instruction_program.hex", dir);
        $readmemh(path, program_words);
        $sformat(path, "%0s/selftest_input0.hex", dir);
        $readmemh(path, selftest_in);
        $sformat(path, "%0s/selftest_actions.hex", dir);
        $readmemh(path, golden);

        repeat (4) @(posedge clk);
        reset_n <= 1;
        repeat (2) @(posedge clk);

        for (group = 0; group < inwords; group = group + 1)
            vec_write(group[8:0], selftest_in[group]);
        for (group = 0; group < ninstr; group = group + 1) begin
            @(negedge clk);
            instruction_addr = group[4:0];
            instruction_wdata = program_words[group];
            instruction_we = 1;
            @(negedge clk);
            instruction_we = 0;
        end
        instruction_count = ninstr[5:0];

        @(negedge clk);
        start_cycle = cycle_counter;
        sequence_start = 1;
        @(negedge clk);
        sequence_start = 0;
        wait (sequence_done);
        if (sequence_error) begin
            $display("FAIL sequence reported error at instruction %0d", sequence_index);
            errors = errors + 1;
        end
        $display("STREAM cycles=%0d bursts=%0d beats=%0d latency=%0d",
                 cycle_counter - start_cycle, axi_bursts, axi_beats, read_latency);

        for (group = 0; group < actwords; group = group + 1) begin
            vec_read(actbase[8:0] + group[8:0]);
            for (lane = 0; lane < 8; lane = lane + 1) begin
                idx = group * 8 + lane;
                if (idx < nactions) begin
                    if (observed[lane*16 +: 16] !== golden[group][lane*16 +: 16]) begin
                        $display("FAIL action[%0d] got %0d expected %0d", idx,
                                 $signed(observed[lane*16 +: 16]),
                                 $signed(golden[group][lane*16 +: 16]));
                        errors = errors + 1;
                    end
                end
            end
        end

        if (errors == 0) $display("PASS tb_weight_stream");
        else $display("FAIL tb_weight_stream errors=%0d", errors);
        $finish;
    end
endmodule
