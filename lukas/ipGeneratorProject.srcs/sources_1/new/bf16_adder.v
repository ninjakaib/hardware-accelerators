`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 12/01/2024 07:41:57 PM
// Design Name: 
// Module Name: bf16_adder
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module bf16_adder(
    input[15:0] A,
    input[15:0] B,
    output [15:0] S
    );
    
    wire exp_sign;
    wire[7:0] exp_diff; 
    
    //subtract exponent
    add_sub_8_bit exp_sub(A[14:7], B[14:7], exp_diff, 0, 1);
    assign exp_sign = exp_diff[7];
    
    wire[7:0] diff_not;
    
    //absolute value
    not_8_bit n1(exp_diff, diff_not);
    
    wire[7:0] abs_out;
    
    integer one = 1;
    
    add_sub_8_bit plus_1(diff_not, one, abs_out, 0, 0);
    
    wire[7:0] abs_diff;
    
    mux_8_bit abs_mux(exp_diff, abs_out, exp_sign, abs_diff);
    
    wire[6:0] greater;
    wire[6:0] lesser;
    
    mux_8_bit greater_mux(A[6:0], B[6:0], exp_sign, greater);
    mux_8_bit lesser_mux(B[6:0], A[6:0], exp_sign, lesser);
    
    wire[6:0] lesser_shifted;
    barrel_shift shifter(lesser, abs_diff, lesser_shifted);
    
    //mantissa addition
    wire sign_xor = A[15] ^ B[15];
    wire[7:0] mantissa_sum;
    
    add_sub_8_bit mantissa_adder(greater, lesser_shifted, mantissa_sum, 0, sign_xor);
    wire carry;
    assign carry = mantissa_sum[7];
    assign mantissa_sum[7] = 0;
    
    wire invert;
    assign invert = sign_xor & carry;
    
    wire[7:0] mantissa_not;
    
    not_8_bit n2(mantissa_sum, mantissa_not);
    
    wire[7:0] m_abs_out;
        
    add_sub_8_bit plus_1_2(mantissa_not, one, m_abs_out, 0, 0);
    
    wire[7:0] mantissa_abs;
    
    mux_8_bit mantissa_mux(m_abs_out, mantissa_sum, invert, mantissa_abs);
    
    //normalize
    wire[2:0] normalize_count;
    wire[6:0] shift_digital;
    
    shift_detector detector(mantissa_abs, shift_digital, normalize_count);
    
    wire[6:0] m_inv;    
    bit_7_flip f1(mantissa_abs, m_inv);
    
    wire[6:0] mantissa_shifted_inv;
    barrel_shift mantissa_shifter(m_inv, normalize_count, mantissa_shifted_inv);
    
    wire[6:0] mantissa_shifted;
    bit_7_flip f2(mantissa_shifted_inv, mantissa_shifted);
    
    wire[7:0] larger_exp;
    mux_8_bit exp_mux(A[14:7], B[14:7], exp_sign, larger_exp);
    
    wire[7:0] normalized_exp;
    wire[7:0] to_add;
    assign to_add[3] = 0;
    assign to_add[4] = 0;
    assign to_add[5] = 0;
    assign to_add[6] = 0;
    assign to_add[7] = 0;
    assign to_add[2:0] = normalize_count;
    add_sub_8_bit exp_norm(larger_exp, to_add, normalized_exp, 0, 0);
 
    assign S[15] = A[15] ^ B[15];
    assign S[14:7] = normalized_exp;
    assign S[6:0] = mantissa_shifted;
    
endmodule
