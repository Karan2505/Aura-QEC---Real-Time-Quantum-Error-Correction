# Vitis HLS flow for the RL-BP decoder kernel.
#
#   cd aura-qec/hls/rlbp
#   vitis_hls -f run_hls.tcl                       (Vitis HLS 2020.2 .. 2024.x)
#   vitis-run --mode hls --tcl run_hls.tcl         (Vitis 2024.2+ unified flow)
#
# Environment overrides (optional):
#   RLBP_MODE  = float | fixed       (float: bit-exact vs Python; fixed: ap_fixed LLRs)
#   RLBP_PART  = FPGA part           (default: Kria KV260 / K26 SOM, free license)
#   RLBP_CLK   = clock period, ns    (default 3.333 = 300 MHz, the AURA core clock)
#   RLBP_STEPS = "csim csynth cosim export"   (subset to run)
#   RLBP_LLR   = "W I"  fixed-point LLR width / integer bits (default "10 6")
#
# Each mode gets its own solution (sol_float, sol_fixed); the project is not
# reset, so running one mode keeps the other's reports.

proc env_or {name default} {
    if {[info exists ::env($name)] && $::env($name) ne ""} { return $::env($name) }
    return $default
}
set MODE  [env_or RLBP_MODE  float]
set PART  [env_or RLBP_PART  xck26-sfvc784-2LV-c]
set CLK   [env_or RLBP_CLK   3.333]
set STEPS [env_or RLBP_STEPS "csim csynth cosim"]

set CFLAGS "-std=c++14"
if {$MODE eq "fixed"} {
    append CFLAGS " -DRLBP_FIXED"
    set LLR [env_or RLBP_LLR ""]
    if {[llength $LLR] == 2} { append CFLAGS " -DRLBP_LLR_W=[lindex $LLR 0] -DRLBP_LLR_I=[lindex $LLR 1]" }
}

open_project rlbp_prj
set_top rlbp_decode
add_files rlbp_decoder.cpp -cflags $CFLAGS
add_files -tb tb_rlbp.cpp  -cflags "$CFLAGS -Wno-unknown-pragmas"
add_files -tb vectors.txt

open_solution -reset "sol_$MODE" -flow_target vivado
set_part $PART
create_clock -period $CLK -name default

if {[lsearch $STEPS csim]   >= 0} { csim_design -argv "vectors.txt" }
if {[lsearch $STEPS csynth] >= 0} { csynth_design }
if {[lsearch $STEPS cosim]  >= 0} { cosim_design -argv "vectors.txt" -trace_level none }
if {[lsearch $STEPS export] >= 0} { export_design -format ip_catalog -rtl verilog }
exit
