`timescale 1ns/1ps
`include "rl_accel_params.vh"

module rl_accel_axil_top #(
    parameter integer AXI_ADDR_W = 12,
    parameter integer AXI_DATA_W = 32
) (
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 s_axi_aclk CLK" *)
    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME s_axi_aclk, ASSOCIATED_BUSIF S_AXI, ASSOCIATED_RESET s_axi_aresetn, FREQ_HZ 100000000" *)
    input  wire                       s_axi_aclk,
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 s_axi_aresetn RST" *)
    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME s_axi_aresetn, POLARITY ACTIVE_LOW" *)
    input  wire                       s_axi_aresetn,

    // RoboAccel-v2 DDR weight stream. Read-only AXI4 master; left unconnected
    // a v1 system still builds and behaves identically.
    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME M_AXI, PROTOCOL AXI4, DATA_WIDTH 64, ADDR_WIDTH 32, FREQ_HZ 100000000, NUM_READ_OUTSTANDING 2, NUM_WRITE_OUTSTANDING 1, HAS_BURST 1, SUPPORTS_NARROW_BURST 0, MAX_BURST_LENGTH 16, READ_WRITE_MODE READ_ONLY" *)
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI ARADDR" *)
    output wire [31:0]                m_axi_araddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI ARLEN" *)
    output wire [7:0]                 m_axi_arlen,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI ARSIZE" *)
    output wire [2:0]                 m_axi_arsize,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI ARBURST" *)
    output wire [1:0]                 m_axi_arburst,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI ARVALID" *)
    output wire                       m_axi_arvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI ARREADY" *)
    input  wire                       m_axi_arready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI RDATA" *)
    input  wire [63:0]                m_axi_rdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI RRESP" *)
    input  wire [1:0]                 m_axi_rresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI RLAST" *)
    input  wire                       m_axi_rlast,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI RVALID" *)
    input  wire                       m_axi_rvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 M_AXI RREADY" *)
    output wire                       m_axi_rready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWADDR" *)
    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME S_AXI, PROTOCOL AXI4LITE, DATA_WIDTH 32, ADDR_WIDTH 12, FREQ_HZ 100000000, MEMORY_TYPE REGISTER, HAS_BURST 0, HAS_LOCK 0, HAS_CACHE 0, HAS_PROT 0, HAS_QOS 0, HAS_REGION 0, SUPPORTS_NARROW_BURST 0" *)
    input  wire [AXI_ADDR_W-1:0]      s_axi_awaddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWVALID" *)
    input  wire                       s_axi_awvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWREADY" *)
    output wire                       s_axi_awready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WDATA" *)
    input  wire [AXI_DATA_W-1:0]      s_axi_wdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WSTRB" *)
    input  wire [AXI_DATA_W/8-1:0]    s_axi_wstrb,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WVALID" *)
    input  wire                       s_axi_wvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WREADY" *)
    output wire                       s_axi_wready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BRESP" *)
    output reg  [1:0]                 s_axi_bresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BVALID" *)
    output reg                        s_axi_bvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BREADY" *)
    input  wire                       s_axi_bready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARADDR" *)
    input  wire [AXI_ADDR_W-1:0]      s_axi_araddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARVALID" *)
    input  wire                       s_axi_arvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARREADY" *)
    output wire                       s_axi_arready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RDATA" *)
    output reg  [AXI_DATA_W-1:0]      s_axi_rdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RRESP" *)
    output reg  [1:0]                 s_axi_rresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RVALID" *)
    output reg                        s_axi_rvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RREADY" *)
    input  wire                       s_axi_rready,
    (* X_INTERFACE_INFO = "xilinx.com:signal:interrupt:1.0 irq INTERRUPT" *)
    (* X_INTERFACE_PARAMETER = "XIL_INTERFACENAME irq, SENSITIVITY LEVEL_HIGH" *)
    output wire                       irq
);
    reg aw_hold;
    reg [AXI_ADDR_W-1:0] awaddr_hold;
    reg w_hold;
    reg [31:0] wdata_hold;
    reg [3:0] wstrb_hold;

    reg cmd_start;
    reg [2:0] opcode;
    reg [`RL_VECTOR_ADDR_W-1:0] src_base;
    reg [`RL_VECTOR_ADDR_W-1:0] dst_base;
    reg [`RL_VECTOR_ADDR_W-1:0] aux0_base;
    reg [`RL_VECTOR_ADDR_W-1:0] aux1_base;
    reg [`RL_WEIGHT_ADDR_W-1:0] weight_base;
    reg [31:0] weight_ddr_base;
    reg [15:0] dim_k;
    reg [15:0] dim_n;
    reg [5:0] output_shift;
    wire core_busy;
    wire core_done;
    wire core_error;
    reg done_sticky;
    reg error_sticky;
    reg irq_enable;

    reg [`RL_VECTOR_ADDR_W-1:0] vec_host_addr;
    reg [31:0] vec_stage [0:3];
    reg vec_wr_pulse;
    reg vec_rd_pulse;
    wire vec_rd_valid;
    wire [127:0] vec_rd_data;
    reg [127:0] vec_read_latch;

    reg [2:0] wt_host_bank;
    reg [`RL_WEIGHT_ADDR_W-1:0] wt_host_addr;
    reg [31:0] wt_stage [0:3];
    reg wt_wr_pulse;
    reg wt_rd_pulse;
    wire wt_rd_valid;
    wire [127:0] wt_rd_data;
    reg [127:0] wt_read_latch;

    reg [`RL_INSTRUCTION_ADDR_W-1:0] instr_host_addr;
    reg [31:0] instr_stage [0:3];
    reg instr_wr_pulse;
    reg instr_rd_pulse;
    wire instr_rd_valid;
    wire [127:0] instr_rd_data;
    reg [127:0] instr_read_latch;
    reg [5:0] sequence_length;
    reg sequence_start;
    wire sequence_busy;
    wire sequence_done;
    wire sequence_error;
    wire [`RL_INSTRUCTION_ADDR_W-1:0] sequence_index;
    wire sequence_cmd_start;
    wire [2:0] sequence_opcode;
    wire [`RL_VECTOR_ADDR_W-1:0] sequence_src_base;
    wire [`RL_VECTOR_ADDR_W-1:0] sequence_dst_base;
    wire [`RL_VECTOR_ADDR_W-1:0] sequence_aux0_base;
    wire [`RL_VECTOR_ADDR_W-1:0] sequence_aux1_base;
    wire [`RL_WEIGHT_ADDR_W-1:0] sequence_weight_base;
    wire [5:0] sequence_subop;
    wire sequence_weight_stream;
    wire [`RL_WEIGHT_STREAM_WORD_W-1:0] sequence_weight_word;
    wire [15:0] sequence_dim_k;
    wire [15:0] sequence_dim_n;
    wire [5:0] sequence_shift;

    wire [127:0] vec_stage_word = {vec_stage[3], vec_stage[2],
                                   vec_stage[1], vec_stage[0]};
    wire [127:0] wt_stage_word = {wt_stage[3], wt_stage[2],
                                  wt_stage[1], wt_stage[0]};
    wire [127:0] instr_stage_word = {instr_stage[3], instr_stage[2],
                                     instr_stage[1], instr_stage[0]};
    wire accel_busy = core_busy || sequence_busy;

    assign s_axi_awready = !aw_hold && !s_axi_bvalid;
    assign s_axi_wready  = !w_hold && !s_axi_bvalid;
    assign s_axi_arready = !s_axi_rvalid;
    assign irq = irq_enable && done_sticky;

    function automatic [31:0] merge_wstrb;
        input [31:0] old_value;
        input [31:0] new_value;
        input [3:0] strb;
        integer byte_index;
        begin
            merge_wstrb = old_value;
            for (byte_index = 0; byte_index < 4; byte_index = byte_index + 1)
                if (strb[byte_index])
                    merge_wstrb[byte_index*8 +: 8] = new_value[byte_index*8 +: 8];
        end
    endfunction

    function automatic [31:0] read_register;
        input [AXI_ADDR_W-1:0] address;
        begin
            case (address[11:0])
                12'h004: read_register = {29'b0, error_sticky, done_sticky, accel_busy};
                12'h008: read_register = {29'b0, opcode};
                12'h00c: read_register = {{(32-`RL_VECTOR_ADDR_W){1'b0}}, src_base};
                12'h010: read_register = {{(32-`RL_VECTOR_ADDR_W){1'b0}}, dst_base};
                12'h014: read_register = {{(32-`RL_VECTOR_ADDR_W){1'b0}}, aux0_base};
                12'h018: read_register = {{(32-`RL_VECTOR_ADDR_W){1'b0}}, aux1_base};
                12'h01c: read_register = {{(32-`RL_WEIGHT_ADDR_W){1'b0}}, weight_base};
                12'h020: read_register = {16'b0, dim_k};
                12'h024: read_register = {16'b0, dim_n};
                12'h028: read_register = {26'b0, output_shift};
                12'h02c: read_register = {31'b0, irq_enable};
                12'h040: read_register = {{(32-`RL_VECTOR_ADDR_W){1'b0}}, vec_host_addr};
                12'h044: read_register = vec_stage[0];
                12'h048: read_register = vec_stage[1];
                12'h04c: read_register = vec_stage[2];
                12'h050: read_register = vec_stage[3];
                12'h058: read_register = vec_read_latch[31:0];
                12'h05c: read_register = vec_read_latch[63:32];
                12'h060: read_register = vec_read_latch[95:64];
                12'h064: read_register = vec_read_latch[127:96];
                12'h080: read_register = {13'b0, wt_host_bank, 5'b0, wt_host_addr};
                12'h084: read_register = wt_stage[0];
                12'h088: read_register = wt_stage[1];
                12'h08c: read_register = wt_stage[2];
                12'h090: read_register = wt_stage[3];
                12'h098: read_register = wt_read_latch[31:0];
                12'h09c: read_register = wt_read_latch[63:32];
                12'h0a0: read_register = wt_read_latch[95:64];
                12'h0a4: read_register = wt_read_latch[127:96];
                12'h0b0: read_register = {{(32-`RL_INSTRUCTION_ADDR_W){1'b0}}, instr_host_addr};
                12'h0b4: read_register = instr_stage[0];
                12'h0b8: read_register = instr_stage[1];
                12'h0bc: read_register = instr_stage[2];
                12'h0c0: read_register = instr_stage[3];
                12'h0c8: read_register = instr_read_latch[31:0];
                12'h0cc: read_register = instr_read_latch[63:32];
                12'h0d0: read_register = instr_read_latch[95:64];
                12'h0d4: read_register = instr_read_latch[127:96];
                12'h0d8: read_register = {26'b0, sequence_length};
                12'h0e0: read_register = {19'b0, sequence_index, 6'b0,
                                           sequence_error, sequence_busy};
                12'h0e4: read_register = weight_ddr_base;
                12'h0fc: read_register = 32'h0001_0004;
                default: read_register = 32'b0;
            endcase
        end
    endfunction

    rl_instruction_sequencer u_instruction_sequencer (
        .clk(s_axi_aclk), .reset_n(s_axi_aresetn),
        .host_wr_en(instr_wr_pulse), .host_addr(instr_host_addr),
        .host_wr_data(instr_stage_word), .host_rd_en(instr_rd_pulse),
        .host_rd_valid(instr_rd_valid), .host_rd_data(instr_rd_data),
        .start(sequence_start), .instruction_count(sequence_length),
        .busy(sequence_busy), .done(sequence_done), .error(sequence_error),
        .current_index(sequence_index), .cmd_start(sequence_cmd_start),
        .cmd_opcode(sequence_opcode), .cmd_src_base(sequence_src_base),
        .cmd_dst_base(sequence_dst_base), .cmd_aux0_base(sequence_aux0_base),
        .cmd_aux1_base(sequence_aux1_base),
        .cmd_weight_base(sequence_weight_base),
        .cmd_dim_k(sequence_dim_k), .cmd_dim_n(sequence_dim_n),
        .cmd_shift(sequence_shift),
        .cmd_subop(sequence_subop),
        .cmd_weight_stream(sequence_weight_stream),
        .cmd_weight_word(sequence_weight_word),
        .core_busy(core_busy),
        .core_done(core_done), .core_error(core_error)
    );

    rl_accel_core u_core (
        .clk(s_axi_aclk), .reset_n(s_axi_aresetn),
        .cmd_start(cmd_start || sequence_cmd_start),
        .cmd_opcode(sequence_busy ? sequence_opcode : opcode),
        .cmd_src_base(sequence_busy ? sequence_src_base : src_base),
        .cmd_dst_base(sequence_busy ? sequence_dst_base : dst_base),
        .cmd_aux0_base(sequence_busy ? sequence_aux0_base : aux0_base),
        .cmd_aux1_base(sequence_busy ? sequence_aux1_base : aux1_base),
        .cmd_weight_base(sequence_busy ? sequence_weight_base : weight_base),
        .cmd_dim_k(sequence_busy ? sequence_dim_k : dim_k),
        .cmd_dim_n(sequence_busy ? sequence_dim_n : dim_n),
        .cmd_shift(sequence_busy ? sequence_shift : output_shift),
        .cmd_subop(sequence_busy ? sequence_subop : 6'd0),
        // Streaming is driven by the descriptor program only; a directly
        // programmed descriptor always uses the on-chip weight cache.
        .cmd_weight_stream(sequence_busy ? sequence_weight_stream : 1'b0),
        .cmd_weight_word(sequence_busy ? sequence_weight_word :
                         {`RL_WEIGHT_STREAM_WORD_W{1'b0}}),
        .weight_ddr_base(weight_ddr_base),
        .busy(core_busy), .done(core_done), .error(core_error),
        .host_vec_wr_en(vec_wr_pulse), .host_vec_addr(vec_host_addr),
        .host_vec_wr_data(vec_stage_word), .host_vec_rd_en(vec_rd_pulse),
        .host_vec_rd_valid(vec_rd_valid), .host_vec_rd_data(vec_rd_data),
        .host_wt_wr_en(wt_wr_pulse), .host_wt_bank(wt_host_bank),
        .host_wt_addr(wt_host_addr), .host_wt_wr_data(wt_stage_word),
        .host_wt_rd_en(wt_rd_pulse), .host_wt_rd_valid(wt_rd_valid),
        .host_wt_rd_data(wt_rd_data),
        .m_axi_araddr(m_axi_araddr), .m_axi_arlen(m_axi_arlen),
        .m_axi_arsize(m_axi_arsize), .m_axi_arburst(m_axi_arburst),
        .m_axi_arvalid(m_axi_arvalid), .m_axi_arready(m_axi_arready),
        .m_axi_rdata(m_axi_rdata), .m_axi_rresp(m_axi_rresp),
        .m_axi_rlast(m_axi_rlast), .m_axi_rvalid(m_axi_rvalid),
        .m_axi_rready(m_axi_rready)
    );

    always @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            aw_hold <= 1'b0;
            w_hold <= 1'b0;
            s_axi_bvalid <= 1'b0;
            s_axi_bresp <= 2'b00;
            s_axi_rvalid <= 1'b0;
            s_axi_rresp <= 2'b00;
            s_axi_rdata <= 0;
            cmd_start <= 1'b0;
            weight_ddr_base <= 0;
            opcode <= 0;
            src_base <= 0;
            dst_base <= 0;
            aux0_base <= 0;
            aux1_base <= 0;
            weight_base <= 0;
            dim_k <= 0;
            dim_n <= 0;
            output_shift <= 0;
            done_sticky <= 1'b0;
            error_sticky <= 1'b0;
            irq_enable <= 1'b0;
            vec_host_addr <= 0;
            vec_stage[0] <= 0;
            vec_stage[1] <= 0;
            vec_stage[2] <= 0;
            vec_stage[3] <= 0;
            vec_wr_pulse <= 1'b0;
            vec_rd_pulse <= 1'b0;
            vec_read_latch <= 0;
            wt_host_bank <= 0;
            wt_host_addr <= 0;
            wt_stage[0] <= 0;
            wt_stage[1] <= 0;
            wt_stage[2] <= 0;
            wt_stage[3] <= 0;
            wt_wr_pulse <= 1'b0;
            wt_rd_pulse <= 1'b0;
            wt_read_latch <= 0;
            instr_host_addr <= 0;
            instr_stage[0] <= 0;
            instr_stage[1] <= 0;
            instr_stage[2] <= 0;
            instr_stage[3] <= 0;
            instr_wr_pulse <= 1'b0;
            instr_rd_pulse <= 1'b0;
            instr_read_latch <= 0;
            sequence_length <= 0;
            sequence_start <= 1'b0;
        end else begin
            cmd_start <= 1'b0;
            vec_wr_pulse <= 1'b0;
            vec_rd_pulse <= 1'b0;
            wt_wr_pulse <= 1'b0;
            wt_rd_pulse <= 1'b0;
            instr_wr_pulse <= 1'b0;
            instr_rd_pulse <= 1'b0;
            sequence_start <= 1'b0;

            if (sequence_done) begin
                done_sticky <= 1'b1;
                error_sticky <= sequence_error;
            end else if (core_done && !sequence_busy) begin
                done_sticky <= 1'b1;
                error_sticky <= core_error;
            end
            if (vec_rd_valid)
                vec_read_latch <= vec_rd_data;
            if (wt_rd_valid)
                wt_read_latch <= wt_rd_data;
            if (instr_rd_valid)
                instr_read_latch <= instr_rd_data;

            if (s_axi_awready && s_axi_awvalid) begin
                aw_hold <= 1'b1;
                awaddr_hold <= s_axi_awaddr;
            end
            if (s_axi_wready && s_axi_wvalid) begin
                w_hold <= 1'b1;
                wdata_hold <= s_axi_wdata;
                wstrb_hold <= s_axi_wstrb;
            end

            if (aw_hold && w_hold && !s_axi_bvalid) begin
                case (awaddr_hold[11:0])
                    12'h000: if (wstrb_hold[0]) begin
                        if (wdata_hold[0] && !accel_busy) begin
                            cmd_start <= 1'b1;
                            done_sticky <= 1'b0;
                            error_sticky <= 1'b0;
                        end
                        if (wdata_hold[1]) begin
                            done_sticky <= 1'b0;
                            error_sticky <= 1'b0;
                        end
                    end
                    12'h008: opcode <= merge_wstrb({29'b0, opcode}, wdata_hold, wstrb_hold);
                    12'h00c: src_base <= merge_wstrb({{(32-`RL_VECTOR_ADDR_W){1'b0}}, src_base}, wdata_hold, wstrb_hold);
                    12'h010: dst_base <= merge_wstrb({{(32-`RL_VECTOR_ADDR_W){1'b0}}, dst_base}, wdata_hold, wstrb_hold);
                    12'h014: aux0_base <= merge_wstrb({{(32-`RL_VECTOR_ADDR_W){1'b0}}, aux0_base}, wdata_hold, wstrb_hold);
                    12'h018: aux1_base <= merge_wstrb({{(32-`RL_VECTOR_ADDR_W){1'b0}}, aux1_base}, wdata_hold, wstrb_hold);
                    12'h01c: weight_base <= merge_wstrb({{(32-`RL_WEIGHT_ADDR_W){1'b0}}, weight_base}, wdata_hold, wstrb_hold);
                    12'h020: dim_k <= merge_wstrb({16'b0, dim_k}, wdata_hold, wstrb_hold);
                    12'h024: dim_n <= merge_wstrb({16'b0, dim_n}, wdata_hold, wstrb_hold);
                    12'h028: output_shift <= merge_wstrb({26'b0, output_shift}, wdata_hold, wstrb_hold);
                    12'h02c: if (wstrb_hold[0]) irq_enable <= wdata_hold[0];
                    12'h040: vec_host_addr <= merge_wstrb({{(32-`RL_VECTOR_ADDR_W){1'b0}}, vec_host_addr}, wdata_hold, wstrb_hold);
                    12'h044: vec_stage[0] <= merge_wstrb(vec_stage[0], wdata_hold, wstrb_hold);
                    12'h048: vec_stage[1] <= merge_wstrb(vec_stage[1], wdata_hold, wstrb_hold);
                    12'h04c: vec_stage[2] <= merge_wstrb(vec_stage[2], wdata_hold, wstrb_hold);
                    12'h050: vec_stage[3] <= merge_wstrb(vec_stage[3], wdata_hold, wstrb_hold);
                    12'h054: if (wstrb_hold[0]) begin
                        if (wdata_hold[0] && !accel_busy) vec_wr_pulse <= 1'b1;
                        if (wdata_hold[1] && !accel_busy) vec_rd_pulse <= 1'b1;
                    end
                    12'h080: begin
                        wt_host_addr <= wdata_hold[`RL_WEIGHT_ADDR_W-1:0];
                        wt_host_bank <= wdata_hold[18:16];
                    end
                    12'h084: wt_stage[0] <= merge_wstrb(wt_stage[0], wdata_hold, wstrb_hold);
                    12'h088: wt_stage[1] <= merge_wstrb(wt_stage[1], wdata_hold, wstrb_hold);
                    12'h08c: wt_stage[2] <= merge_wstrb(wt_stage[2], wdata_hold, wstrb_hold);
                    12'h090: wt_stage[3] <= merge_wstrb(wt_stage[3], wdata_hold, wstrb_hold);
                    12'h094: if (wstrb_hold[0]) begin
                        if (wdata_hold[0] && !accel_busy) wt_wr_pulse <= 1'b1;
                        if (wdata_hold[1] && !accel_busy) wt_rd_pulse <= 1'b1;
                    end
                    12'h0b0: instr_host_addr <=
                        merge_wstrb({{(32-`RL_INSTRUCTION_ADDR_W){1'b0}},
                                     instr_host_addr}, wdata_hold, wstrb_hold);
                    12'h0b4: instr_stage[0] <=
                        merge_wstrb(instr_stage[0], wdata_hold, wstrb_hold);
                    12'h0b8: instr_stage[1] <=
                        merge_wstrb(instr_stage[1], wdata_hold, wstrb_hold);
                    12'h0bc: instr_stage[2] <=
                        merge_wstrb(instr_stage[2], wdata_hold, wstrb_hold);
                    12'h0c0: instr_stage[3] <=
                        merge_wstrb(instr_stage[3], wdata_hold, wstrb_hold);
                    12'h0c4: if (wstrb_hold[0]) begin
                        if (wdata_hold[0] && !accel_busy) instr_wr_pulse <= 1'b1;
                        if (wdata_hold[1] && !accel_busy) instr_rd_pulse <= 1'b1;
                    end
                    12'h0d8: sequence_length <=
                        merge_wstrb({26'b0, sequence_length},
                                    wdata_hold, wstrb_hold);
                    12'h0e4: weight_ddr_base <=
                        merge_wstrb(weight_ddr_base, wdata_hold, wstrb_hold);
                    12'h0dc: if (wstrb_hold[0] && wdata_hold[0] &&
                                      !accel_busy) begin
                        sequence_start <= 1'b1;
                        done_sticky <= 1'b0;
                        error_sticky <= 1'b0;
                    end
                    default: begin end
                endcase
                aw_hold <= 1'b0;
                w_hold <= 1'b0;
                s_axi_bresp <= 2'b00;
                s_axi_bvalid <= 1'b1;
            end else if (s_axi_bvalid && s_axi_bready) begin
                s_axi_bvalid <= 1'b0;
            end

            if (s_axi_arready && s_axi_arvalid) begin
                s_axi_rdata <= read_register(s_axi_araddr);
                s_axi_rresp <= 2'b00;
                s_axi_rvalid <= 1'b1;
            end else if (s_axi_rvalid && s_axi_rready) begin
                s_axi_rvalid <= 1'b0;
            end
        end
    end
endmodule
