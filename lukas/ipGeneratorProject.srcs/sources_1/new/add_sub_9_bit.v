`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 12/01/2024 07:35:18 PM
// Design Name: 
// Module Name: add_sub_9_bit
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


module add_sub_9_bit(
    input[8:0] A,
    input[8:0] B,
    output[8:0] S,
    input cin,
    input sub,
    output cout
    );
    
    wire[8:0] I;
    wire c;
    
    generate genvar i;
    for (i = 0; i < 9; i = i + 1) 
        assign I[i] = B[i] ^ sub;
    endgenerate;
    wire s = cin ^ sub;
    adder_8_bit adder(A[7:0], I[7:0], S[7:0], s, c);
    full_adder bit_9(c, A[8], I[8], S[8], cout);
endmodule
