`timescale 1ns/1ps
`include "rl_accel_params.vh"

module tb_instruction_sequencer;
    reg clk = 0;
    always #5 clk = ~clk;
    reg reset_n = 0;

    reg host_wr_en = 0;
    reg host_rd_en = 0;
    reg [4:0] host_addr = 0;
    reg [127:0] host_wr_data = 0;
    wire host_rd_valid;
    wire [127:0] host_rd_data;
    reg start = 0;
    reg [5:0] instruction_count = 0;
    wire busy;
    wire done;
    wire error;
    wire [4:0] current_index;
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

    reg core_busy = 0;
    reg core_done = 0;
    reg core_error = 0;
    reg [2:0] core_delay = 0;
    integer launches = 0;
    integer errors = 0;

    rl_instruction_sequencer u_dut (
        .clk(clk), .reset_n(reset_n), .host_wr_en(host_wr_en),
        .host_addr(host_addr), .host_wr_data(host_wr_data),
        .host_rd_en(host_rd_en), .host_rd_valid(host_rd_valid),
        .host_rd_data(host_rd_data), .start(start),
        .instruction_count(instruction_count), .busy(busy), .done(done),
        .error(error), .current_index(current_index), .cmd_start(cmd_start),
        .cmd_opcode(cmd_opcode), .cmd_src_base(cmd_src_base),
        .cmd_dst_base(cmd_dst_base), .cmd_aux0_base(cmd_aux0_base),
        .cmd_aux1_base(cmd_aux1_base), .cmd_weight_base(cmd_weight_base),
        .cmd_dim_k(cmd_dim_k), .cmd_dim_n(cmd_dim_n), .cmd_shift(cmd_shift),
        .core_busy(core_busy), .core_done(core_done), .core_error(core_error)
    );

    always @(posedge clk) begin
        core_done <= 1'b0;
        if (!reset_n) begin
            core_busy <= 1'b0;
            core_delay <= 0;
            launches <= 0;
        end else if (cmd_start) begin
            core_busy <= 1'b1;
            core_delay <= 2;
            launches <= launches + 1;
            if (launches == 0 &&
                (cmd_opcode != `RL_OP_ELU || cmd_src_base != 5 ||
                 cmd_dst_base != 6 || cmd_dim_n != 16)) begin
                $display("FAIL first decoded descriptor");
                errors = errors + 1;
            end
            if (launches == 1 &&
                (cmd_opcode != `RL_OP_CONCAT || cmd_src_base != 6 ||
                 cmd_dst_base != 8 || cmd_aux0_base != 7 ||
                 cmd_dim_k != 16 || cmd_dim_n != 3)) begin
                $display("FAIL second decoded descriptor");
                errors = errors + 1;
            end
        end else if (core_busy) begin
            if (core_delay == 0) begin
                core_busy <= 1'b0;
                core_done <= 1'b1;
            end else begin
                core_delay <= core_delay - 1'b1;
            end
        end
    end

    function automatic [127:0] descriptor;
        input [2:0] opcode;
        input [8:0] src;
        input [8:0] dst;
        input [8:0] aux0;
        input [15:0] dim_k;
        input [15:0] dim_n;
        reg [127:0] bits;
        begin
            bits = 0;
            bits[2:0] = opcode;
            bits[11:3] = src;
            bits[20:12] = dst;
            bits[29:21] = aux0;
            bits[79:64] = dim_k;
            bits[95:80] = dim_n;
            descriptor = bits;
        end
    endfunction

    task write_instruction;
        input [4:0] address;
        input [127:0] data;
        begin
            @(negedge clk);
            host_addr = address;
            host_wr_data = data;
            host_wr_en = 1;
            @(negedge clk);
            host_wr_en = 0;
        end
    endtask

    task launch_sequence;
        input [5:0] count;
        begin
            @(negedge clk);
            instruction_count = count;
            start = 1;
            @(negedge clk);
            start = 0;
            wait(done);
            @(negedge clk);
        end
    endtask

    initial begin
        reg [127:0] value;
        repeat (4) @(posedge clk);
        reset_n <= 1;
        repeat (2) @(posedge clk);

        value = descriptor(`RL_OP_ELU, 5, 6, 0, 0, 16);
        write_instruction(0, value);
        write_instruction(1, descriptor(`RL_OP_CONCAT, 6, 8, 7, 16, 3));
        @(negedge clk);
        host_addr = 0;
        host_rd_en = 1;
        @(negedge clk);
        host_rd_en = 0;
        wait(host_rd_valid);
        if (host_rd_data !== value) begin
            $display("FAIL direct instruction readback");
            errors = errors + 1;
        end

        launch_sequence(2);
        if (error || launches != 2 || current_index != 1) begin
            $display("FAIL valid sequence error=%0d launches=%0d index=%0d",
                     error, launches, current_index);
            errors = errors + 1;
        end

        launch_sequence(0);
        if (!error) begin
            $display("FAIL zero-length sequence was accepted");
            errors = errors + 1;
        end

        write_instruction(0, descriptor(3'b111, 0, 0, 0, 1, 1));
        launch_sequence(1);
        if (!error || launches != 2) begin
            $display("FAIL invalid opcode was dispatched");
            errors = errors + 1;
        end

        if (errors == 0)
            $display("PASS tb_instruction_sequencer");
        else
            $fatal(1, "tb_instruction_sequencer failed with %0d errors", errors);
        $finish;
    end
endmodule
