`timescale 1ns/1ps

module rl_weight_cache #(
    parameter integer DEPTH = 1280,
    parameter integer ADDR_W = 11
) (
    input  wire                 clk,

    input  wire                 host_wr_en,
    input  wire [2:0]           host_bank,
    input  wire [ADDR_W-1:0]    host_addr,
    input  wire [127:0]         host_wr_data,
    input  wire                 host_rd_en,
    output reg                  host_rd_valid,
    output wire [127:0]         host_rd_data,

    input  wire                 eng_rd_en,
    input  wire [ADDR_W-1:0]    eng_rd_addr,
    output reg                  eng_rd_valid,
    output wire [1023:0]        eng_rd_data
);
    wire [127:0] eng_bank_rd [0:7];
    wire common_rd_en = host_rd_en || eng_rd_en;
    wire [ADDR_W-1:0] common_rd_addr = host_rd_en ? host_addr : eng_rd_addr;

    genvar bank;
    generate
        for (bank = 0; bank < 8; bank = bank + 1) begin : g_bank
            // A monolithic 1280x128 memory is an inefficient aspect ratio for
            // RAMB36E1.  Splitting at 1024 lets Vivado use four 1024x36 blocks
            // for the main part and two 512x72 blocks for the tail.
            (* ram_style = "block" *) reg [127:0] mem_main [0:1023];
            (* ram_style = "block" *) reg [127:0] mem_tail [0:255];
            reg [127:0] main_q;
            reg [127:0] tail_q;
            reg tail_select_q;

            always @(posedge clk) begin
                if (host_wr_en && (host_bank == bank[2:0])) begin
                    if (host_addr < 11'd1024)
                        mem_main[host_addr[9:0]] <= host_wr_data;
                    else if (host_addr < DEPTH)
                        mem_tail[host_addr[7:0]] <= host_wr_data;
                end
            end

            always @(posedge clk) begin
                if (common_rd_en) begin
                    tail_select_q <= (common_rd_addr >= 11'd1024);
                    if (common_rd_addr < 11'd1024)
                        main_q <= mem_main[common_rd_addr[9:0]];
                    else
                        tail_q <= mem_tail[common_rd_addr[7:0]];
                end
            end

            assign eng_bank_rd[bank] = tail_select_q ? tail_q : main_q;
            assign eng_rd_data[bank*128 +: 128] = eng_bank_rd[bank];
        end
    endgenerate

    assign host_rd_data = eng_bank_rd[host_bank];

    always @(posedge clk) begin
        host_rd_valid <= host_rd_en;
        eng_rd_valid <= eng_rd_en;
    end
endmodule
