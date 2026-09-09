IVERILOG ?= iverilog
VVP ?= vvp
RTL := ../rtl

.PHONY: all primitives gemm concat sequencer top policy stream cycles v3 clean

all: primitives gemm concat sequencer top policy stream v3

primitives:
	$(IVERILOG) -g2012 -Wall -s tb_primitives -o primitives.vvp \
		$(RTL)/rl_round_shift_sat16.sv \
		$(RTL)/rl_mac_array_8x8.sv \
		$(RTL)/rl_elu_array8.sv \
		tb_primitives.sv
	$(VVP) primitives.vvp

gemm:
	$(IVERILOG) -g2012 -Wall -s tb_gemm_engine -o gemm.vvp \
		$(RTL)/rl_round_shift_sat16.sv \
		$(RTL)/rl_mac_array_8x8.sv \
		$(RTL)/rl_vector_cache.sv \
		$(RTL)/rl_weight_cache.sv \
		$(RTL)/rl_gemm_engine.sv tb_gemm_engine.sv
	$(VVP) gemm.vvp

concat:
	$(IVERILOG) -g2012 -Wall -s tb_concat_engine -o concat.vvp \
		$(RTL)/rl_vector_cache.sv \
		$(RTL)/rl_concat_engine.sv tb_concat_engine.sv
	$(VVP) concat.vvp

sequencer:
	$(IVERILOG) -g2012 -Wall -I $(RTL) -s tb_instruction_sequencer \
		-o sequencer.vvp $(RTL)/rl_instruction_sequencer.sv \
		tb_instruction_sequencer.sv
	$(VVP) sequencer.vvp

top:
	$(IVERILOG) -g2012 -Wall -I $(RTL) -s tb_axil_top -o top.vvp \
		$(RTL)/rl_round_shift_sat16.sv \
		$(RTL)/rl_mac_array_8x8.sv \
		$(RTL)/rl_elu_array8.sv \
		$(RTL)/rl_vector_cache.sv \
		$(RTL)/rl_weight_cache.sv \
		$(RTL)/rl_gemm_engine.sv \
		$(RTL)/rl_elu_engine.sv \
		$(RTL)/rl_concat_engine.sv \
		$(RTL)/rl_vector_engine.sv \
		$(RTL)/rl_sfu_engine.sv \
		$(RTL)/rl_weight_streamer.sv \
		$(RTL)/rl_accel_core.sv \
		$(RTL)/rl_instruction_sequencer.sv \
		$(RTL)/rl_accel_axil_top.sv tb_axil_top.sv
	$(VVP) top.vvp

policy:
	$(IVERILOG) -g2012 -Wall -I $(RTL) -s tb_policy_e2e -o policy.vvp \
		$(RTL)/rl_round_shift_sat16.sv \
		$(RTL)/rl_mac_array_8x8.sv \
		$(RTL)/rl_elu_array8.sv \
		$(RTL)/rl_vector_cache.sv \
		$(RTL)/rl_weight_cache.sv \
		$(RTL)/rl_gemm_engine.sv \
		$(RTL)/rl_elu_engine.sv \
		$(RTL)/rl_concat_engine.sv \
		$(RTL)/rl_vector_engine.sv \
		$(RTL)/rl_sfu_engine.sv \
		$(RTL)/rl_weight_streamer.sv \
		$(RTL)/rl_accel_core.sv \
		$(RTL)/rl_instruction_sequencer.sv tb_policy_e2e.sv
	$(VVP) policy.vvp

clean:
	rm -f *.vvp *.vcd

# RoboAccel-v2: a policy too large for the on-chip cache, executed from a DDR
# model through the accelerator's AXI4 master.
STREAM_RTL := $(RTL)/rl_round_shift_sat16.sv $(RTL)/rl_mac_array_8x8.sv \
	$(RTL)/rl_elu_array8.sv \
	$(RTL)/rl_vector_cache.sv $(RTL)/rl_weight_cache.sv \
	$(RTL)/rl_weight_streamer.sv $(RTL)/rl_gemm_engine.sv \
	$(RTL)/rl_elu_engine.sv \
	$(RTL)/rl_concat_engine.sv $(RTL)/rl_vector_engine.sv $(RTL)/rl_sfu_engine.sv $(RTL)/rl_accel_core.sv \
	$(RTL)/rl_instruction_sequencer.sv

stream:
	$(IVERILOG) -g2012 -Wall -I $(RTL) -s tb_weight_stream -o stream.vvp \
		$(STREAM_RTL) tb_weight_stream.sv
	$(VVP) stream.vvp +dir=../generated_go2 +ninstr=7 +inwords=6 \
		+actbase=230 +actwords=2 +nactions=12 +latency=30
	$(VVP) stream.vvp +dir=../generated_g1 +ninstr=7 +inwords=60 \
		+actbase=284 +actwords=4 +nactions=29 +latency=30

cycles:
	$(IVERILOG) -g2012 -Wall -I $(RTL) -s tb_cycle_model -o cyc.vvp \
		$(STREAM_RTL) tb_cycle_model.sv
	$(VVP) cyc.vvp

v3:
	$(IVERILOG) -g2012 -Wall -I $(RTL) -s tb_v3_primitives -o v3.vvp \
		$(STREAM_RTL) tb_v3_primitives.sv
	$(VVP) v3.vvp
