`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 11/14/2024 02:08:22 PM
// Design Name: 
// Module Name: zero_check
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


module zero_check(
    input [7:0] A_exp,
    input [7:0] B_exp,
    output zero
    );
    
    wire A = A_exp[0] | A_exp[1] | A_exp[2] | A_exp[3] | A_exp[4] | A_exp[5] | A_exp[6] | A_exp[7];
    wire B = B_exp[0] | B_exp[1] | B_exp[2] | B_exp[3] | B_exp[4] | B_exp[5] | B_exp[6] | B_exp[7];
    
    assign zero = A | B;
    
endmodule
