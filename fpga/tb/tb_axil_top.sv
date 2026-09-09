`timescale 1ns/1ps

module tb_axil_top;
    reg clk = 0;
    always #5 clk = ~clk;
    reg reset_n = 0;

    reg [11:0] awaddr = 0;
    reg awvalid = 0;
    wire awready;
    reg [31:0] wdata = 0;
    reg [3:0] wstrb = 0;
    reg wvalid = 0;
    wire wready;
    wire [1:0] bresp;
    wire bvalid;
    reg bready = 1;
    reg [11:0] araddr = 0;
    reg arvalid = 0;
    wire arready;
    wire [31:0] rdata;
    wire [1:0] rresp;
    wire rvalid;
    reg rready = 1;
    wire irq;

    integer lane;
    integer errors = 0;

    rl_accel_axil_top u_dut (
        .s_axi_aclk(clk), .s_axi_aresetn(reset_n),
        .s_axi_awaddr(awaddr), .s_axi_awvalid(awvalid), .s_axi_awready(awready),
        .s_axi_wdata(wdata), .s_axi_wstrb(wstrb), .s_axi_wvalid(wvalid),
        .s_axi_wready(wready), .s_axi_bresp(bresp), .s_axi_bvalid(bvalid),
        .s_axi_bready(bready), .s_axi_araddr(araddr), .s_axi_arvalid(arvalid),
        .s_axi_arready(arready), .s_axi_rdata(rdata), .s_axi_rresp(rresp),
        .s_axi_rvalid(rvalid), .s_axi_rready(rready), .irq(irq),
        .m_axi_araddr(), .m_axi_arlen(), .m_axi_arsize(), .m_axi_arburst(),
        .m_axi_arvalid(), .m_axi_arready(1'b0), .m_axi_rdata(64'b0),
        .m_axi_rresp(2'b00), .m_axi_rlast(1'b0), .m_axi_rvalid(1'b0),
        .m_axi_rready()
    );

    task axi_write;
        input [11:0] address;
        input [31:0] data;
        begin
            @(negedge clk);
            awaddr = address;
            awvalid = 1;
            wdata = data;
            wstrb = 4'hf;
            wvalid = 1;
            while (!(awready && wready)) @(negedge clk);
            @(negedge clk);
            awvalid = 0;
            wvalid = 0;
            while (!bvalid) @(negedge clk);
            if (bresp != 0) begin
                $display("FAIL AXI write response at %h", address);
                errors = errors + 1;
            end
            @(negedge clk);
        end
    endtask

    task axi_read;
        input [11:0] address;
        output [31:0] data;
        begin
            @(negedge clk);
            araddr = address;
            arvalid = 1;
            while (!arready) @(negedge clk);
            @(negedge clk);
            arvalid = 0;
            while (!rvalid) @(negedge clk);
            data = rdata;
            if (rresp != 0) begin
                $display("FAIL AXI read response at %h", address);
                errors = errors + 1;
            end
            @(negedge clk);
        end
    endtask

    task vec_write;
        input [8:0] address;
        input [127:0] data;
        begin
            axi_write(12'h040, address);
            axi_write(12'h044, data[31:0]);
            axi_write(12'h048, data[63:32]);
            axi_write(12'h04c, data[95:64]);
            axi_write(12'h050, data[127:96]);
            axi_write(12'h054, 32'h1);
        end
    endtask

    task vec_read;
        input [8:0] address;
        output [127:0] data;
        reg [31:0] d0;
        reg [31:0] d1;
        reg [31:0] d2;
        reg [31:0] d3;
        begin
            axi_write(12'h040, address);
            axi_write(12'h054, 32'h2);
            repeat (3) @(posedge clk);
            axi_read(12'h058, d0);
            axi_read(12'h05c, d1);
            axi_read(12'h060, d2);
            axi_read(12'h064, d3);
            data = {d3, d2, d1, d0};
        end
    endtask

    task wt_write;
        input [2:0] bank;
        input [10:0] address;
        input [127:0] data;
        begin
            axi_write(12'h080, {13'b0, bank, 5'b0, address});
            axi_write(12'h084, data[31:0]);
            axi_write(12'h088, data[63:32]);
            axi_write(12'h08c, data[95:64]);
            axi_write(12'h090, data[127:96]);
            axi_write(12'h094, 32'h1);
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
            axi_write(12'h0b0, address);
            axi_write(12'h0b4, data[31:0]);
            axi_write(12'h0b8, data[63:32]);
            axi_write(12'h0bc, data[95:64]);
            axi_write(12'h0c0, data[127:96]);
            axi_write(12'h0c4, 32'h1);
        end
    endtask

    task instruction_read;
        input [4:0] address;
        output [127:0] data;
        reg [31:0] d0;
        reg [31:0] d1;
        reg [31:0] d2;
        reg [31:0] d3;
        begin
            axi_write(12'h0b0, address);
            axi_write(12'h0c4, 32'h2);
            repeat (3) @(posedge clk);
            axi_read(12'h0c8, d0);
            axi_read(12'h0cc, d1);
            axi_read(12'h0d0, d2);
            axi_read(12'h0d4, d3);
            data = {d3, d2, d1, d0};
        end
    endtask

    task wait_done;
        reg [31:0] status;
        integer timeout;
        begin
            status = 0;
            timeout = 0;
            while (!status[1] && timeout < 10000) begin
                axi_read(12'h004, status);
                timeout = timeout + 1;
            end
            if (!status[1] || status[2]) begin
                $display("FAIL command status=%h timeout=%0d", status, timeout);
                errors = errors + 1;
            end
        end
    endtask

    initial begin
        reg [127:0] word;
        reg [127:0] result;
        reg [31:0] value32;

        repeat (4) @(posedge clk);
        reset_n <= 1;
        repeat (2) @(posedge clk);

        axi_read(12'h0fc, value32);
        if (value32 != 32'h0001_0004) begin
            $display("FAIL version register %h", value32);
            errors = errors + 1;
        end

        // GEMM: y[j] = sum(1..8)*(j+1) + j.
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
        axi_write(12'h008, 1);
        axi_write(12'h00c, 0);
        axi_write(12'h010, 2);
        axi_write(12'h014, 1);
        axi_write(12'h01c, 0);
        axi_write(12'h020, 8);
        axi_write(12'h024, 8);
        axi_write(12'h028, 0);
        axi_write(12'h000, 1);
        wait_done();
        vec_read(2, result);
        for (lane = 0; lane < 8; lane = lane + 1)
            if ($signed(result[lane*16 +: 16]) !== 36*(lane+1)+lane) begin
                $display("FAIL top GEMM lane %0d got %0d", lane,
                    $signed(result[lane*16 +: 16]));
                errors = errors + 1;
            end

        // VECTOR AFFINE (opcode 3, sub-op 0): dst = sat((src*a + b) >> shift).
        // This replaces the retired NORM operator, whose per-feature affine is
        // a strict subset of it and which no corpus policy used.
        word = 0;
        for (lane = 0; lane < 8; lane = lane + 1)
            word[lane*16 +: 16] = 128 + lane;
        vec_write(10, word);
        word = 0;
        for (lane = 0; lane < 8; lane = lane + 1)
            word[lane*16 +: 16] = 64;              // scale, with shift 6 => x1
        vec_write(11, word);
        word = 0;
        for (lane = 0; lane < 8; lane = lane + 1)
            word[lane*16 +: 16] = 100;             // offset
        vec_write(12, word);
        axi_write(12'h008, 3);
        axi_write(12'h00c, 10);
        axi_write(12'h010, 13);
        axi_write(12'h014, 11);
        axi_write(12'h018, 12);
        axi_write(12'h024, 8);
        axi_write(12'h028, 6);
        axi_write(12'h000, 1);
        wait_done();
        vec_read(13, result);
        for (lane = 0; lane < 8; lane = lane + 1)
            if ($signed(result[lane*16 +: 16]) !== 228 + lane) begin
                $display("FAIL top VECTOR AFFINE lane %0d got %0d expected %0d", lane,
                    $signed(result[lane*16 +: 16]), 228 + lane);
                errors = errors + 1;
            end

        // ELU integration, including saturation below -8.
        word = 0;
        word[0*16 +: 16] = 16'sd256;
        word[1*16 +: 16] = -16'sd256;
        word[2*16 +: 16] = -16'sd2048;
        word[3*16 +: 16] = -16'sd4096;
        vec_write(14, word);
        axi_write(12'h008, 2);
        axi_write(12'h00c, 14);
        axi_write(12'h010, 15);
        axi_write(12'h024, 4);
        axi_write(12'h000, 1);
        wait_done();
        vec_read(15, result);
        if ($signed(result[15:0]) != 256 ||
            $signed(result[31:16]) != -162 ||
            $signed(result[47:32]) != -256 ||
            $signed(result[63:48]) != -256) begin
            $display("FAIL top ELU result=%h", result);
            errors = errors + 1;
        end

        // Program two CONCAT descriptors, read them back, and execute both
        // with one sequence start. This is the normal inference control path.
        word = 0;
        for (lane = 0; lane < 8; lane = lane + 1)
            word[lane*16 +: 16] = 100 + lane;
        vec_write(20, word);
        word = 0;
        for (lane = 0; lane < 8; lane = lane + 1)
            word[lane*16 +: 16] = 200 + lane;
        vec_write(21, word);

        word = descriptor(4, 20, 22, 21, 0, 0, 3, 2, 0);
        instruction_write(0, word);
        instruction_read(0, result);
        if (result !== word) begin
            $display("FAIL instruction readback got=%h expected=%h", result, word);
            errors = errors + 1;
        end
        word = descriptor(4, 22, 23, 20, 0, 0, 5, 1, 0);
        instruction_write(1, word);
        instruction_read(1, result);
        if (result !== word) begin
            $display("FAIL instruction readback got=%h expected=%h", result, word);
            errors = errors + 1;
        end
        axi_write(12'h0d8, 2);
        axi_write(12'h0dc, 1);
        wait_done();
        axi_read(12'h0e0, value32);
        if (value32[2:0] != 0 || value32[12:8] != 1) begin
            $display("FAIL sequence status %h", value32);
            errors = errors + 1;
        end
        vec_read(23, result);
        if ($signed(result[15:0]) != 100 ||
            $signed(result[31:16]) != 101 ||
            $signed(result[47:32]) != 102 ||
            $signed(result[63:48]) != 200 ||
            $signed(result[79:64]) != 201 ||
            $signed(result[95:80]) != 100) begin
            $display("FAIL automatic CONCAT sequence result=%h", result);
            errors = errors + 1;
        end

        if (errors == 0)
            $display("PASS tb_axil_top");
        else
            $fatal(1, "tb_axil_top failed with %0d errors", errors);
        $finish;
    end
endmodule
