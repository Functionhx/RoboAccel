`timescale 1ns/1ps

module rl_vector_cache #(
    parameter integer DEPTH = 512,
    parameter integer ADDR_W = 9
) (
    input  wire                 clk,

    input  wire                 host_wr_en,
    input  wire [ADDR_W-1:0]    host_wr_addr,
    input  wire [127:0]         host_wr_data,
    input  wire                 host_rd_en,
    input  wire [ADDR_W-1:0]    host_rd_addr,
    output reg                  host_rd_valid,
    output reg  [127:0]         host_rd_data,

    input  wire                 eng_en,
    input  wire                 eng_we,
    input  wire [ADDR_W-1:0]    eng_addr,
    input  wire [127:0]         eng_wr_data,
    output reg                  eng_rd_valid,
    output reg  [127:0]         eng_rd_data
);
    (* ram_style = "block" *) reg [127:0] mem [0:DEPTH-1];

    // Port A: PS/debug access.
    always @(posedge clk) begin
        host_rd_valid <= host_rd_en;
        if (host_wr_en)
            mem[host_wr_addr] <= host_wr_data;
        if (host_rd_en)
            host_rd_data <= mem[host_rd_addr];
    end

    // Port B: compute engines. Read-first behavior is not relied upon.
    always @(posedge clk) begin
        eng_rd_valid <= eng_en && !eng_we;
        if (eng_en) begin
            if (eng_we)
                mem[eng_addr] <= eng_wr_data;
            else
                eng_rd_data <= mem[eng_addr];
        end
    end
endmodule
