`timescale 1ns/1ps

module tb_concat_engine;
    reg clk = 0;
    always #5 clk = ~clk;
    reg reset_n = 0;
    reg start = 0;
    reg host_we = 0;
    reg [8:0] host_addr = 0;
    reg [127:0] host_wdata = 0;
    wire busy;
    wire done;
    wire vec_en;
    wire vec_we;
    wire [8:0] vec_addr;
    wire [127:0] vec_wdata;
    wire vec_rvalid;
    wire [127:0] vec_rdata;
    integer errors = 0;

    rl_vector_cache u_cache (
        .clk(clk), .host_wr_en(host_we), .host_wr_addr(host_addr),
        .host_wr_data(host_wdata), .host_rd_en(1'b0), .host_rd_addr(9'b0),
        .host_rd_valid(), .host_rd_data(), .eng_en(vec_en), .eng_we(vec_we),
        .eng_addr(vec_addr), .eng_wr_data(vec_wdata),
        .eng_rd_valid(vec_rvalid), .eng_rd_data(vec_rdata)
    );

    rl_concat_engine u_dut (
        .clk(clk), .reset_n(reset_n), .start(start),
        .src0_base(9'd0), .src1_base(9'd4), .dst_base(9'd8),
        .count0(16'd5), .count1(16'd6), .src0_offset(9'd0),
        .busy(busy), .done(done),
        .vec_en(vec_en), .vec_we(vec_we), .vec_addr(vec_addr),
        .vec_wr_data(vec_wdata), .vec_rd_valid(vec_rvalid),
        .vec_rd_data(vec_rdata)
    );

    task write_word;
        input [8:0] address;
        input [127:0] data;
        begin
            @(negedge clk);
            host_addr = address;
            host_wdata = data;
            host_we = 1'b1;
            @(negedge clk);
            host_we = 1'b0;
        end
    endtask

    initial begin
        repeat (3) @(posedge clk);
        reset_n <= 1'b1;
        write_word(0, {16'd0,16'd0,16'd0,16'd5,16'd4,16'd3,16'd2,16'd1});
        write_word(4, {16'd0,16'd0,16'd106,16'd105,16'd104,16'd103,16'd102,16'd101});
        @(negedge clk); start = 1'b1;
        @(negedge clk); start = 1'b0;
        wait(done);
        @(negedge clk);
        if (u_cache.mem[8] !== {16'd103,16'd102,16'd101,16'd5,16'd4,16'd3,16'd2,16'd1}) begin
            $display("FAIL concat word0 %h", u_cache.mem[8]);
            errors = errors + 1;
        end
        if (u_cache.mem[9] !== {80'd0,16'd106,16'd105,16'd104}) begin
            $display("FAIL concat word1 %h", u_cache.mem[9]);
            errors = errors + 1;
        end
        if (errors == 0)
            $display("PASS tb_concat_engine");
        else
            $fatal(1, "tb_concat_engine failed with %0d errors", errors);
        $finish;
    end
endmodule
