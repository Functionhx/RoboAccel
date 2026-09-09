`timescale 1ns/1ps

// RoboAccel-v2 weight streamer.
//
// The v1 accelerator can only run a policy whose entire weight set fits in the
// on-chip weight cache (1280 words/bank).  Real quadruped and humanoid
// locomotion actors need several times the whole device's block RAM, so they
// cannot be made resident at any cache size on an XC7Z010.  This module reads
// the weight stream straight out of DDR instead.
//
// The GEMM engine walks a layer's weights strictly sequentially -- out_group 0
// reads words [base, base+kg), out_group 1 reads [base+kg, base+2*kg), and so
// on -- so the whole layer is one linear scan of kg*ng words.  That makes a
// plain incrementing burst reader sufficient; no reordering or caching is
// needed.
//
// One 1024-bit weight word is exactly one 16-beat INCR burst on a 64-bit AXI
// data bus (128 bytes), which also guarantees a burst never crosses a 4 KB
// boundary as long as the base address is 128-byte aligned.
//
// Two 1024-bit buffers ping-pong: one is filled by AXI while the other is
// drained by the GEMM engine.  Two outstanding bursts are allowed so the AXI
// read channel keeps moving across DDR latency.
module rl_weight_streamer #(
    parameter integer AXI_ADDR_W = 32,
    parameter integer AXI_DATA_W = 64
) (
    input  wire                    clk,
    input  wire                    reset_n,

    // Layer control, pulsed together with the GEMM engine's start.
    input  wire                    start,
    input  wire [AXI_ADDR_W-1:0]   base_address,
    input  wire [15:0]             dim_k,
    input  wire [15:0]             dim_n,

    // Engine-facing port. Mirrors rl_weight_cache's timing: a read enable
    // produces data with eng_rd_valid one cycle later. eng_ready additionally
    // tells the GEMM engine when a word is actually available.
    input  wire                    eng_rd_en,
    output wire                    eng_ready,
    output reg                     eng_rd_valid,
    output reg  [1023:0]           eng_rd_data,

    // AXI4 read-only master.
    output reg  [AXI_ADDR_W-1:0]   m_axi_araddr,
    output wire [7:0]              m_axi_arlen,
    output wire [2:0]              m_axi_arsize,
    output wire [1:0]              m_axi_arburst,
    output reg                     m_axi_arvalid,
    input  wire                    m_axi_arready,
    input  wire [AXI_DATA_W-1:0]   m_axi_rdata,
    input  wire [1:0]              m_axi_rresp,
    input  wire                    m_axi_rlast,
    input  wire                    m_axi_rvalid,
    output wire                    m_axi_rready,

    output reg                     error
);
    localparam integer BEATS = 1024 / AXI_DATA_W;   // 16 for a 64-bit bus
    localparam integer WORD_BYTES = 128;

    assign m_axi_arlen   = BEATS - 1;
    assign m_axi_arsize  = (AXI_DATA_W == 64) ? 3'b011 : 3'b010;
    assign m_axi_arburst = 2'b01;                   // INCR
    assign m_axi_rready  = 1'b1;                    // buffers are always free

    reg [1023:0] buffer [0:1];
    reg [1:0]    buffer_full;
    reg          fill_select;
    reg          issue_select;
    reg          drain_select;
    reg [1:0]    outstanding;

    reg [25:0]   total_words;
    reg [25:0]   issued_words;
    reg [AXI_ADDR_W-1:0] next_address;
    reg          active;

    wire [15:0] k_groups = (dim_k + 16'd7) >> 3;
    wire [15:0] n_groups = (dim_n + 16'd7) >> 3;

    assign eng_ready = buffer_full[drain_select];

    wire issue_allowed = active && (issued_words < total_words) &&
                         (outstanding < 2'd2) && !buffer_full[issue_select] &&
                         !(m_axi_arvalid && !m_axi_arready);
    wire burst_end = m_axi_rvalid && m_axi_rlast;

    always @(posedge clk) begin
        if (!reset_n) begin
            buffer_full   <= 2'b00;
            fill_select   <= 1'b0;
            issue_select  <= 1'b0;
            drain_select  <= 1'b0;
            outstanding   <= 2'd0;
            total_words   <= 0;
            issued_words  <= 0;
            next_address  <= 0;
            active        <= 1'b0;
            m_axi_arvalid <= 1'b0;
            m_axi_araddr  <= 0;
            eng_rd_valid  <= 1'b0;
            eng_rd_data   <= 0;
            error         <= 1'b0;
        end else begin
            eng_rd_valid <= 1'b0;

            if (start) begin
                // A new layer restarts the scan. Any buffered word belongs to
                // the previous layer and must be discarded.
                total_words   <= k_groups * n_groups;
                issued_words  <= 0;
                next_address  <= base_address;
                buffer_full   <= 2'b00;
                fill_select   <= 1'b0;
                issue_select  <= 1'b0;
                drain_select  <= 1'b0;
                outstanding   <= 2'd0;
                m_axi_arvalid <= 1'b0;
                active        <= 1'b1;
                error         <= 1'b0;
            end else begin
                // Address channel.
                if (m_axi_arvalid && m_axi_arready)
                    m_axi_arvalid <= 1'b0;

                if (issue_allowed) begin
                    m_axi_araddr  <= next_address;
                    m_axi_arvalid <= 1'b1;
                    next_address  <= next_address + WORD_BYTES;
                    issued_words  <= issued_words + 1'b1;
                    issue_select  <= ~issue_select;
                end

                // One issue and one completion in the same cycle cancel out.
                case ({issue_allowed, burst_end})
                    2'b10:   outstanding <= outstanding + 1'b1;
                    2'b01:   outstanding <= outstanding - 1'b1;
                    default: outstanding <= outstanding;
                endcase

                // Read data channel: assemble beats least-significant first.
                if (m_axi_rvalid) begin
                    buffer[fill_select] <=
                        {m_axi_rdata, buffer[fill_select][1023:AXI_DATA_W]};
                    if (m_axi_rresp != 2'b00)
                        error <= 1'b1;
                    if (m_axi_rlast) begin
                        buffer_full[fill_select] <= 1'b1;
                        fill_select <= ~fill_select;
                    end
                end

                // Engine-facing drain.
                if (eng_rd_en && buffer_full[drain_select]) begin
                    eng_rd_data  <= buffer[drain_select];
                    eng_rd_valid <= 1'b1;
                    buffer_full[drain_select] <= 1'b0;
                    drain_select <= ~drain_select;
                end
            end
        end
    end
endmodule
