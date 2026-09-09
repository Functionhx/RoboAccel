`timescale 1ns/1ps
`include "rl_accel_params.vh"

module rl_accel_core (
    input  wire                         clk,
    input  wire                         reset_n,

    input  wire                         cmd_start,
    input  wire [2:0]                   cmd_opcode,
    input  wire [`RL_VECTOR_ADDR_W-1:0] cmd_src_base,
    input  wire [`RL_VECTOR_ADDR_W-1:0] cmd_dst_base,
    input  wire [`RL_VECTOR_ADDR_W-1:0] cmd_aux0_base,
    input  wire [`RL_VECTOR_ADDR_W-1:0] cmd_aux1_base,
    input  wire [`RL_WEIGHT_ADDR_W-1:0] cmd_weight_base,
    input  wire [15:0]                  cmd_dim_k,
    input  wire [15:0]                  cmd_dim_n,
    input  wire [5:0]                   cmd_shift,
    input  wire [5:0]                   cmd_subop,
    input  wire                         cmd_weight_stream,
    input  wire [`RL_WEIGHT_STREAM_WORD_W-1:0] cmd_weight_word,
    input  wire [31:0]                  weight_ddr_base,
    output reg                          busy,
    output reg                          done,
    output reg                          error,

    input  wire                         host_vec_wr_en,
    input  wire [`RL_VECTOR_ADDR_W-1:0] host_vec_addr,
    input  wire [127:0]                 host_vec_wr_data,
    input  wire                         host_vec_rd_en,
    output wire                         host_vec_rd_valid,
    output wire [127:0]                 host_vec_rd_data,

    input  wire                         host_wt_wr_en,
    input  wire [2:0]                   host_wt_bank,
    input  wire [`RL_WEIGHT_ADDR_W-1:0] host_wt_addr,
    input  wire [127:0]                 host_wt_wr_data,
    input  wire                         host_wt_rd_en,
    output wire                         host_wt_rd_valid,
    output wire [127:0]                 host_wt_rd_data,

    // RoboAccel-v2 DDR weight stream (AXI4 read-only master).
    output wire [31:0]                  m_axi_araddr,
    output wire [7:0]                   m_axi_arlen,
    output wire [2:0]                   m_axi_arsize,
    output wire [1:0]                   m_axi_arburst,
    output wire                         m_axi_arvalid,
    input  wire                         m_axi_arready,
    input  wire [63:0]                  m_axi_rdata,
    input  wire [1:0]                   m_axi_rresp,
    input  wire                         m_axi_rlast,
    input  wire                         m_axi_rvalid,
    output wire                         m_axi_rready
);
    reg [2:0] active_op;
    reg [`RL_VECTOR_ADDR_W-1:0] active_src_base;
    reg [`RL_VECTOR_ADDR_W-1:0] active_dst_base;
    reg [`RL_VECTOR_ADDR_W-1:0] active_aux0_base;
    reg [`RL_VECTOR_ADDR_W-1:0] active_aux1_base;
    reg [`RL_WEIGHT_ADDR_W-1:0] active_weight_base;
    reg [15:0] active_dim_k;
    reg [15:0] active_dim_n;
    reg [5:0] active_shift;
    reg [5:0] active_subop;
    reg active_weight_stream;
    reg [`RL_WEIGHT_STREAM_WORD_W-1:0] active_weight_word;
    reg gemm_start;
    reg elu_start;
    reg vector_start;
    reg sfu_start;
    reg concat_start;

    wire gemm_busy;
    wire gemm_done;
    wire elu_busy;
    wire elu_done;
    wire vector_busy;
    wire vector_done;
    wire sfu_busy;
    wire sfu_done;
    wire concat_busy;
    wire concat_done;

    wire gemm_vec_en;
    wire gemm_vec_we;
    wire [`RL_VECTOR_ADDR_W-1:0] gemm_vec_addr;
    wire [127:0] gemm_vec_wdata;
    wire gemm_wt_re;
    wire [`RL_WEIGHT_ADDR_W-1:0] gemm_wt_addr;

    wire elu_vec_en;
    wire elu_vec_we;
    wire [`RL_VECTOR_ADDR_W-1:0] elu_vec_addr;
    wire [127:0] elu_vec_wdata;

    wire vector_vec_en;
    wire vector_vec_we;
    wire [`RL_VECTOR_ADDR_W-1:0] vector_vec_addr;
    wire [127:0] vector_vec_wdata;

    wire sfu_vec_en;
    wire sfu_vec_we;
    wire [`RL_VECTOR_ADDR_W-1:0] sfu_vec_addr;
    wire [127:0] sfu_vec_wdata;

    wire concat_vec_en;
    wire concat_vec_we;
    wire [`RL_VECTOR_ADDR_W-1:0] concat_vec_addr;
    wire [127:0] concat_vec_wdata;

    reg cache_vec_en;
    reg cache_vec_we;
    reg [`RL_VECTOR_ADDR_W-1:0] cache_vec_addr;
    reg [127:0] cache_vec_wdata;
    wire cache_vec_rvalid;
    wire [127:0] cache_vec_rdata;
    wire cache_wt_rvalid;
    wire [1023:0] cache_wt_rdata;
    wire stream_wt_rvalid;
    wire [1023:0] stream_wt_rdata;
    wire stream_ready;
    wire stream_error;

    // The weight source is selected per GEMM. A descriptor with word3 == 0
    // decodes as stream = 0 and takes exactly the v1 on-chip cache path.
    wire gemm_wt_ready  = active_weight_stream ? stream_ready : 1'b1;
    wire gemm_wt_rvalid = active_weight_stream ? stream_wt_rvalid : cache_wt_rvalid;
    wire [1023:0] gemm_wt_rdata =
        active_weight_stream ? stream_wt_rdata : cache_wt_rdata;

    rl_vector_cache #(
        .DEPTH(`RL_VECTOR_DEPTH), .ADDR_W(`RL_VECTOR_ADDR_W)
    ) u_vector_cache (
        .clk(clk),
        .host_wr_en(host_vec_wr_en), .host_wr_addr(host_vec_addr),
        .host_wr_data(host_vec_wr_data), .host_rd_en(host_vec_rd_en),
        .host_rd_addr(host_vec_addr), .host_rd_valid(host_vec_rd_valid),
        .host_rd_data(host_vec_rd_data),
        .eng_en(cache_vec_en), .eng_we(cache_vec_we),
        .eng_addr(cache_vec_addr), .eng_wr_data(cache_vec_wdata),
        .eng_rd_valid(cache_vec_rvalid), .eng_rd_data(cache_vec_rdata)
    );

    rl_weight_cache #(
        .DEPTH(`RL_WEIGHT_DEPTH), .ADDR_W(`RL_WEIGHT_ADDR_W)
    ) u_weight_cache (
        .clk(clk), .host_wr_en(host_wt_wr_en), .host_bank(host_wt_bank),
        .host_addr(host_wt_addr), .host_wr_data(host_wt_wr_data),
        .host_rd_en(host_wt_rd_en), .host_rd_valid(host_wt_rd_valid),
        .host_rd_data(host_wt_rd_data),
        .eng_rd_en(gemm_wt_re && !active_weight_stream), .eng_rd_addr(gemm_wt_addr),
        .eng_rd_valid(cache_wt_rvalid), .eng_rd_data(cache_wt_rdata)
    );

    rl_weight_streamer u_weight_streamer (
        .clk(clk), .reset_n(reset_n),
        .start(gemm_start && active_weight_stream),
        .base_address(weight_ddr_base + {active_weight_word[24:0], 7'b0}),
        .dim_k(active_dim_k), .dim_n(active_dim_n),
        .eng_rd_en(gemm_wt_re && active_weight_stream),
        .eng_ready(stream_ready),
        .eng_rd_valid(stream_wt_rvalid), .eng_rd_data(stream_wt_rdata),
        .m_axi_araddr(m_axi_araddr), .m_axi_arlen(m_axi_arlen),
        .m_axi_arsize(m_axi_arsize), .m_axi_arburst(m_axi_arburst),
        .m_axi_arvalid(m_axi_arvalid), .m_axi_arready(m_axi_arready),
        .m_axi_rdata(m_axi_rdata), .m_axi_rresp(m_axi_rresp),
        .m_axi_rlast(m_axi_rlast), .m_axi_rvalid(m_axi_rvalid),
        .m_axi_rready(m_axi_rready), .error(stream_error)
    );

    rl_gemm_engine u_gemm (
        .clk(clk), .reset_n(reset_n), .start(gemm_start),
        .src_base(active_src_base), .dst_base(active_dst_base),
        .bias_base(active_aux0_base), .weight_base(active_weight_base),
        .dim_k(active_dim_k), .dim_n(active_dim_n), .output_shift(active_shift),
        .busy(gemm_busy), .done(gemm_done),
        .vec_en(gemm_vec_en), .vec_we(gemm_vec_we),
        .vec_addr(gemm_vec_addr), .vec_wr_data(gemm_vec_wdata),
        .vec_rd_valid(cache_vec_rvalid), .vec_rd_data(cache_vec_rdata),
        .wt_rd_en(gemm_wt_re), .wt_rd_addr(gemm_wt_addr),
        .wt_rd_valid(gemm_wt_rvalid), .wt_rd_data(gemm_wt_rdata),
        .wt_ready(gemm_wt_ready)
    );

    rl_elu_engine u_elu (
        .clk(clk), .reset_n(reset_n), .start(elu_start),
        .src_base(active_src_base), .dst_base(active_dst_base),
        .element_count(active_dim_n), .busy(elu_busy), .done(elu_done),
        .vec_en(elu_vec_en), .vec_we(elu_vec_we),
        .vec_addr(elu_vec_addr), .vec_wr_data(elu_vec_wdata),
        .vec_rd_valid(cache_vec_rvalid), .vec_rd_data(cache_vec_rdata)
    );

    rl_vector_engine u_vector (
        .clk(clk), .reset_n(reset_n), .start(vector_start),
        .subop(active_subop[2:0]),
        .src_base(active_src_base), .dst_base(active_dst_base),
        .aux0_base(active_aux0_base), .aux1_base(active_aux1_base),
        .element_count(active_dim_n), .shift(active_shift),
        .busy(vector_busy), .done(vector_done),
        .vec_en(vector_vec_en), .vec_we(vector_vec_we),
        .vec_addr(vector_vec_addr), .vec_wr_data(vector_vec_wdata),
        .vec_rd_valid(cache_vec_rvalid), .vec_rd_data(cache_vec_rdata)
    );

    rl_sfu_engine u_sfu (
        .clk(clk), .reset_n(reset_n), .start(sfu_start),
        .curve(active_subop[1:0]),
        .src_base(active_src_base), .dst_base(active_dst_base),
        .element_count(active_dim_n),
        .busy(sfu_busy), .done(sfu_done),
        .vec_en(sfu_vec_en), .vec_we(sfu_vec_we),
        .vec_addr(sfu_vec_addr), .vec_wr_data(sfu_vec_wdata),
        .vec_rd_valid(cache_vec_rvalid), .vec_rd_data(cache_vec_rdata)
    );

    rl_concat_engine u_concat (
        .clk(clk), .reset_n(reset_n), .start(concat_start),
        .src0_base(active_src_base), .src1_base(active_aux0_base),
        .dst_base(active_dst_base), .count0(active_dim_k),
        .count1(active_dim_n), .src0_offset(active_aux1_base),
        .busy(concat_busy), .done(concat_done),
        .vec_en(concat_vec_en), .vec_we(concat_vec_we),
        .vec_addr(concat_vec_addr), .vec_wr_data(concat_vec_wdata),
        .vec_rd_valid(cache_vec_rvalid), .vec_rd_data(cache_vec_rdata)
    );

    always @* begin
        cache_vec_en = 1'b0;
        cache_vec_we = 1'b0;
        cache_vec_addr = 0;
        cache_vec_wdata = 0;
        case (active_op)
            `RL_OP_GEMM: begin
                cache_vec_en = gemm_vec_en;
                cache_vec_we = gemm_vec_we;
                cache_vec_addr = gemm_vec_addr;
                cache_vec_wdata = gemm_vec_wdata;
            end
            `RL_OP_ELU: begin
                cache_vec_en = elu_vec_en;
                cache_vec_we = elu_vec_we;
                cache_vec_addr = elu_vec_addr;
                cache_vec_wdata = elu_vec_wdata;
            end
            `RL_OP_VECTOR: begin
                cache_vec_en = vector_vec_en;
                cache_vec_we = vector_vec_we;
                cache_vec_addr = vector_vec_addr;
                cache_vec_wdata = vector_vec_wdata;
            end
            `RL_OP_CONCAT: begin
                cache_vec_en = concat_vec_en;
                cache_vec_we = concat_vec_we;
                cache_vec_addr = concat_vec_addr;
                cache_vec_wdata = concat_vec_wdata;
            end
            `RL_OP_SFU: begin
                cache_vec_en = sfu_vec_en;
                cache_vec_we = sfu_vec_we;
                cache_vec_addr = sfu_vec_addr;
                cache_vec_wdata = sfu_vec_wdata;
            end
            default: begin end
        endcase
    end

    always @(posedge clk) begin
        if (!reset_n) begin
            active_op <= 0;
            active_src_base <= 0;
            active_dst_base <= 0;
            active_aux0_base <= 0;
            active_aux1_base <= 0;
            active_weight_base <= 0;
            active_dim_k <= 0;
            active_dim_n <= 0;
            active_shift <= 0;
            active_weight_stream <= 1'b0;
            active_weight_word <= 0;
            gemm_start <= 1'b0;
            elu_start <= 1'b0;
            vector_start <= 1'b0;
            sfu_start <= 1'b0;
            concat_start <= 1'b0;
            active_subop <= 0;
            busy <= 1'b0;
            done <= 1'b0;
            error <= 1'b0;
        end else begin
            gemm_start <= 1'b0;
            elu_start <= 1'b0;
            vector_start <= 1'b0;
            sfu_start <= 1'b0;
            concat_start <= 1'b0;
            done <= 1'b0;

            if (!busy && cmd_start) begin
                error <= 1'b0;
                active_op <= cmd_opcode;
                active_src_base <= cmd_src_base;
                active_dst_base <= cmd_dst_base;
                active_aux0_base <= cmd_aux0_base;
                active_aux1_base <= cmd_aux1_base;
                active_weight_base <= cmd_weight_base;
                active_dim_k <= cmd_dim_k;
                active_dim_n <= cmd_dim_n;
                active_shift <= cmd_shift;
                active_subop <= cmd_subop;
                active_weight_stream <= cmd_weight_stream && (cmd_opcode == `RL_OP_GEMM);
                active_weight_word <= cmd_weight_word;
                case (cmd_opcode)
                    `RL_OP_GEMM: begin
                        if ((cmd_dim_k == 0) || (cmd_dim_n == 0)) begin
                            error <= 1'b1;
                            done <= 1'b1;
                        end else begin
                            busy <= 1'b1;
                            gemm_start <= 1'b1;
                        end
                    end
                    `RL_OP_ELU: begin
                        if (cmd_dim_n == 0) begin
                            error <= 1'b1;
                            done <= 1'b1;
                        end else begin
                            busy <= 1'b1;
                            elu_start <= 1'b1;
                        end
                    end
                    `RL_OP_VECTOR: begin
                        if (cmd_dim_n == 0) begin
                            error <= 1'b1;
                            done <= 1'b1;
                        end else begin
                            busy <= 1'b1;
                            vector_start <= 1'b1;
                        end
                    end
                    `RL_OP_SFU: begin
                        if (cmd_dim_n == 0) begin
                            error <= 1'b1;
                            done <= 1'b1;
                        end else begin
                            busy <= 1'b1;
                            sfu_start <= 1'b1;
                        end
                    end
                    `RL_OP_CONCAT: begin
                        if ((cmd_dim_k == 0) && (cmd_dim_n == 0)) begin
                            error <= 1'b1;
                            done <= 1'b1;
                        end else begin
                            busy <= 1'b1;
                            concat_start <= 1'b1;
                        end
                    end
                    default: begin
                        error <= 1'b1;
                        done <= 1'b1;
                    end
                endcase
            end else if (busy) begin
                case (active_op)
                    `RL_OP_GEMM: if (gemm_done) begin
                        busy <= 1'b0;
                        done <= 1'b1;
                        if (active_weight_stream && stream_error) error <= 1'b1;
                    end
                    `RL_OP_ELU:  if (elu_done)  begin busy <= 1'b0; done <= 1'b1; end
                    `RL_OP_VECTOR: if (vector_done) begin busy <= 1'b0; done <= 1'b1; end
                    `RL_OP_SFU: if (sfu_done) begin busy <= 1'b0; done <= 1'b1; end
                    `RL_OP_CONCAT: if (concat_done) begin busy <= 1'b0; done <= 1'b1; end
                    default: begin busy <= 1'b0; error <= 1'b1; done <= 1'b1; end
                endcase
            end
        end
    end
endmodule
