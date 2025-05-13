
module fp_lut_array_b_cycle_stage1#(
    // 使用Parameters包中的参数作为默认参数值
    parameter SIG_WIDTH = 10, 
    parameter EXP_WIDTH = 5,
    parameter IEEE_COMPLIANCE = 0,
    parameter A_LUT_BIT = SIG_WIDTH+EXP_WIDTH+1,
    parameter B_BIT = 2,
    parameter C_BIT = A_LUT_BIT,           // 
    parameter M_DIM = 2,
    parameter N_DIM = 64,
    parameter K_DIM = 4,
    parameter A_LUT_DIM = 2**(K_DIM-1)
)
(
    input logic clk,
    input logic rst_n,
    input logic signed [A_LUT_BIT-1:0] a_lut [M_DIM-1:0][A_LUT_DIM-1:0],
    input logic [B_BIT-1:0] b_tile [K_DIM-1:0][N_DIM-1:0],
    input logic signed [C_BIT-1:0] c_tile_psum [M_DIM-1:0][N_DIM-1:0],
    output logic valid,
    output logic signed [C_BIT-1:0] result [M_DIM-1:0][N_DIM-1:0] // Output of the adder
);
    logic signed [A_LUT_BIT-1:0] temp_result[M_DIM-1:0][N_DIM-1:0];
    logic [K_DIM-1:0] temp_lut_index [N_DIM-1:0];
    logic [K_DIM-2:0] temp_lut_index_transformed [N_DIM-1:0];
    logic signed [C_BIT-1:0] adder_mux [M_DIM-1:0][N_DIM-1:0];
    logic signed [C_BIT-1:0] result_w [M_DIM-1:0][N_DIM-1:0];
    logic signed [A_LUT_BIT-1:0] temp_result_before_exp_shift[M_DIM-1:0][N_DIM-1:0];
    logic signed [EXP_WIDTH-1:0] temp_result_exp_value[M_DIM-1:0][N_DIM-1:0];
    logic temp_result_sign_bit[M_DIM-1:0][N_DIM-1:0];

    logic [$clog2(B_BIT)-1:0] b_index;

    // FSM here
    always_ff @(posedge clk or negedge rst_n) begin
        if (~rst_n) begin
            b_index <= '0;
        end else begin
            if(b_index==(B_BIT-1)) begin
                b_index <= 0;
            end else begin
                b_index <= b_index+1;
            end
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (~rst_n) begin
            valid <= '0;
        end else begin
            if(b_index==(B_BIT-1)) begin
                valid <= 1;
            end else begin
                valid <= 0;
            end
        end
    end

    genvar i,j,k;

    generate
        for (j=0;j<N_DIM;j=j+1) begin:output_column

            for (k=0;k<K_DIM;k=k+1) begin:k_reduce
                // concat
               
                assign temp_lut_index[j][k] = b_tile[k][j][b_index];
                
            end

            // assign temp_lut_index_transformed[j]=temp_lut_index[j][0]?~temp_lut_index[j][K_DIM-1:1]:temp_lut_index[j][K_DIM-1:1];
            assign temp_lut_index_transformed[j]=temp_lut_index[j][K_DIM-1:1];

            
        end    
    endgenerate

    generate
        for (i=0;i<M_DIM;i=i+1) begin:output_row
            for (j=0;j<N_DIM;j=j+1) begin:output_column


                assign temp_result_before_exp_shift[i][j]=a_lut[i][ temp_lut_index_transformed[j] ];
                
                assign temp_result_sign_bit[i][j]=(temp_lut_index[j][0])?~temp_result_before_exp_shift[i][j][SIG_WIDTH+EXP_WIDTH]:temp_result_before_exp_shift[i][j][SIG_WIDTH+EXP_WIDTH];
                assign temp_result_exp_value[i][j]=temp_result_before_exp_shift[i][j][SIG_WIDTH+EXP_WIDTH-1:SIG_WIDTH]+b_index;
                
                assign temp_result[i][j]={temp_result_sign_bit[i][j],temp_result_exp_value[i][j],temp_result_before_exp_shift[i][j][SIG_WIDTH-1:0]};  

                
            end    
        end
    endgenerate


    generate
        for (i=0;i<M_DIM;i=i+1) begin
            for (j=0;j<N_DIM;j=j+1) begin

                assign adder_mux[i][j]=(b_index==0)?c_tile_psum[i][j]:result[i][j];

                DW_fp_add #(SIG_WIDTH, EXP_WIDTH, IEEE_COMPLIANCE)
	            output_psum ( .a(temp_result[i][j]), .b(adder_mux[i][j]), .rnd(3'b000), .z(result_w[i][j]), .status( ) );

                always_ff @(posedge clk or negedge rst_n) begin
                    if (~rst_n) begin
                        result[i][j] <= '0;
                    end 
                    else begin

                        result[i][j]<=result_w[i][j];
                    end

                end
            end    
        end
    endgenerate

endmodule

