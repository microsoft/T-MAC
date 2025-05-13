# Reset all constraints
reset_design

#---------------- SET AREA CONSTRAINT ----------------
# 1
set_max_area 0
#set_max_leakage_power 0


#---------------- SET TIMING CONSTRAINTS ----------------
# 2.1 Set clock

#set_max_delay -from [get_ports data] -to [get_ports diff] 0.5

set clk_period 1
# set clk_period 0.70922
create_clock -period $clk_period [get_ports clk]

#set_clock_uncertainty [expr 0.05*$clk_period] [get_clocks clk]
set_clock_transition [expr 0.025*$clk_period] [get_clocks clk]

# 2.2
set_input_delay -max [expr 0*$clk_period] -clock clk [remove_from_collection [all_inputs] {clk rst_n}]
set_driving_cell -lib_cell BUFFD1BWP30P140ULVT [remove_from_collection [all_inputs] {clk rst_n}]
#set_load 2 [all_outputs]

# 2.3
set_output_delay -max [expr 0*$clk_period] -clock clk [all_outputs]
set_dont_touch [get_ports clk]
set_ideal_network {clk rst_n}


