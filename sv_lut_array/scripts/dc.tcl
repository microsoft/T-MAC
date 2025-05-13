remove_design -all
#################################### set library ########################################
# this is the library path for the TSMC 28nm process, set yout own library path here
set search_path "$search_path ../rtl ../scripts /home/lincx/library/tsmc28/lib /home/LIBRARY/TSMC28HPC/TSMCHOME/digital/Front_End/timing_power_noise/CCS/tcbn28hpcplusbwp30p140ulvt_140a /home/ranxiangyu/siliconsmart/convert"
set target_library   "tcbn28hpcplusbwp30p140ulvttt0p9v25c_ccs.db"
#set target_library   "tcbn28_ulvt_900.db"
set synthetic_library   "dw_foundation.sldb"
set link_library     " $target_library $synthetic_library"

#################################### set location #######################################
set top_design fp_lut_array_b_cycle_stage1
# set your name here

# set top_design lut_16

set outdir ..

###################################### initial ##########################################
# analyze -format sverilog  ../rtl/Parameters.sv
analyze -format sverilog  ../rtl/${top_design}.sv

elaborate $top_design
link

################################## add constraint ######################################
source constraint.tcl
source namingrules.tcl

#################################### compile #######################################
#set_flatten true
compile
# compile_ultra -retime -area_high_effort_script

#compile_ultra -timing_high_effort_script

############################ save results ##################################

# write -format verilog -hier -out $outdir/netlists/${top_design}.nl.v
# write_sdf $outdir/reports/${top_design}.sdf
# write_sdc $outdir/reports/${top_design}.sdc
# write -format ddc -hier -o $outdir/netlists/${top_design}.ddc
report_area -hier > $outdir/reports/area_${top_design}.rpt
report_power > $outdir/reports/power_${top_design}.rpt
report_constraint -all_violators > $outdir/reports/violation_${top_design}.rpt
report_timing -delay max > $outdir/reports/timing_${top_design}.rpt
report_timing -delay min >> $outdir/reports/timing_${top_design}.rpt

#check_design

#write -format verilog -hier -out $outdir/netlists/${top_design}.v
##write_sdf $outdir/reports/${top_design}.sdf
##write_sdc $outdir/reports/${top_design}.sdc
##report_area -hier > $outdir/reports/area.rpt
##report_power > $outdir/reports/power.rpt
##report_constraint -all_violators > $outdir/reports/violation.rpt
##report_timing -delay max > $outdir/reports/timing.rpt
##report_timing -delay min >> $outdir/reports/timing.rpt
#
######################################## quit ##########################################
#
