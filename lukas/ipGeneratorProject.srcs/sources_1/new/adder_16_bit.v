`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 11/05/2024 04:22:51 PM
// Design Name: 
// Module Name: adder_16_bit
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


module adder_16_bit(
    input[15:0] A,
    input[15:0] B,
    output[15:0] S,
    input cin,
    output cout
    );
    
    
    generate genvar i;
    wire [16:0] carry;
    assign carry[0] = cin;
    
    
    for(i=0; i<16; i = i + 1) 
        full_adder A(carry[i],A[i],B[i],S[i],carry[i + 1]);
    
    endgenerate;    
    assign cout = carry[16];
endmodule
