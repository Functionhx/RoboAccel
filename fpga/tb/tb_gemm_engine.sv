`timescale 1ns/1ps

module tb_gemm_engine;
    reg clk = 0;
    always #5 clk = ~clk;
    reg reset_n = 0;

    reg host_vec_we = 0;
    reg [8:0] host_vec_waddr = 0;
    reg [127:0] host_vec_wdata = 0;
    reg host_vec_re = 0;
    reg [8:0] host_vec_raddr = 0;
    wire host_vec_rvalid;
    wire [127:0] host_vec_rdata;

    reg host_wt_we = 0;
    reg [2:0] host_wt_bank = 0;
    reg [10:0] host_wt_addr = 0;
    reg [127:0] host_wt_wdata = 0;

    reg start = 0;
    wire busy;
    wire done;
    wire vec_en;
    wire vec_we;
    wire [8:0] vec_addr;
    wire [127:0] vec_wdata;
    wire vec_rvalid;
    wire [127:0] vec_rdata;
    wire wt_re;
    wire [10:0] wt_raddr;
    wire wt_rvalid;
    wire [1023:0] wt_rdata;

    integer lane;
    integer errors = 0;

    rl_vector_cache u_vec (
        .clk(clk),
        .host_wr_en(host_vec_we), .host_wr_addr(host_vec_waddr),
        .host_wr_data(host_vec_wdata), .host_rd_en(host_vec_re),
        .host_rd_addr(host_vec_raddr), .host_rd_valid(host_vec_rvalid),
        .host_rd_data(host_vec_rdata),
        .eng_en(vec_en), .eng_we(vec_we), .eng_addr(vec_addr),
        .eng_wr_data(vec_wdata), .eng_rd_valid(vec_rvalid),
        .eng_rd_data(vec_rdata)
    );

    rl_weight_cache u_wt (
        .clk(clk), .host_wr_en(host_wt_we), .host_bank(host_wt_bank),
        .host_addr(host_wt_addr), .host_wr_data(host_wt_wdata),
        .host_rd_en(1'b0), .host_rd_valid(), .host_rd_data(),
        .eng_rd_en(wt_re), .eng_rd_addr(wt_raddr),
        .eng_rd_valid(wt_rvalid), .eng_rd_data(wt_rdata)
    );

    rl_gemm_engine u_dut (
        .clk(clk), .reset_n(reset_n), .start(start),
        .src_base(9'd0), .dst_base(9'd2), .bias_base(9'd1),
        .weight_base(11'd0), .dim_k(16'd8), .dim_n(16'd8),
        .output_shift(6'd0), .busy(busy), .done(done),
        .vec_en(vec_en), .vec_we(vec_we), .vec_addr(vec_addr),
        .vec_wr_data(vec_wdata), .vec_rd_valid(vec_rvalid),
        .vec_rd_data(vec_rdata), .wt_rd_en(wt_re),
        .wt_rd_addr(wt_raddr), .wt_rd_valid(wt_rvalid),
        .wt_rd_data(wt_rdata), .wt_ready(1'b1)
    );

    task vec_write;
        input [8:0] addr;
        input [127:0] data;
        begin
            @(negedge clk);
            host_vec_waddr = addr;
            host_vec_wdata = data;
            host_vec_we = 1;
            @(negedge clk);
            host_vec_we = 0;
        end
    endtask

    task wt_write;
        input [2:0] bank;
        input [10:0] addr;
        input [127:0] data;
        begin
            @(negedge clk);
            host_wt_bank = bank;
            host_wt_addr = addr;
            host_wt_wdata = data;
            host_wt_we = 1;
            @(negedge clk);
            host_wt_we = 0;
        end
    endtask

    initial begin
        reg [127:0] word;
        repeat (3) @(posedge clk);
        reset_n <= 1;

        word = 0;
        for (lane = 0; lane < 8; lane = lane + 1)
            word[lane*16 +: 16] = lane + 1;
        vec_write(0, word);

        word = 0;
        for (lane = 0; lane < 8; lane = lane + 1)
            word[lane*16 +: 16] = lane;
        vec_write(1, word);

        for (lane = 0; lane < 8; lane = lane + 1) begin
            word = 0;
            word[0*16 +: 16] = lane + 1;
            word[1*16 +: 16] = lane + 1;
            word[2*16 +: 16] = lane + 1;
            word[3*16 +: 16] = lane + 1;
            word[4*16 +: 16] = lane + 1;
            word[5*16 +: 16] = lane + 1;
            word[6*16 +: 16] = lane + 1;
            word[7*16 +: 16] = lane + 1;
            wt_write(lane[2:0], 0, word);
        end

        @(negedge clk); start = 1;
        @(negedge clk); start = 0;
        wait(done);

        @(negedge clk);
        host_vec_raddr = 2;
        host_vec_re = 1;
        @(negedge clk);
        host_vec_re = 0;
        wait(host_vec_rvalid);
        #1;
        for (lane = 0; lane < 8; lane = lane + 1) begin
            if ($signed(host_vec_rdata[lane*16 +: 16]) !== (36*(lane+1)+lane)) begin
                $display("FAIL lane %0d got %0d expected %0d", lane,
                    $signed(host_vec_rdata[lane*16 +: 16]), 36*(lane+1)+lane);
                errors = errors + 1;
            end
        end

        if (errors == 0)
            $display("PASS tb_gemm_engine");
        else
            $fatal(1, "tb_gemm_engine failed");
        $finish;
    end
endmodule
