`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 11/05/2024 06:30:03 PM
// Design Name: 
// Module Name: subtractor_16_bit
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


module subtractor_16_bit(
    input[15:0] A,
    input[15:0] B,
    output[15:0] D,
    input cin,
    output cout
    );
    
    generate genvar i;
    wire [16:0] carry;
    assign carry[0] = cin;
    
    
    for(i=0; i<16; i = i + 1) 
        full_subtractor S(carry[i],A[i],B[i],D[i],carry[i + 1]);
    
    endgenerate;    
    assign cout = carry[16];
endmodule
