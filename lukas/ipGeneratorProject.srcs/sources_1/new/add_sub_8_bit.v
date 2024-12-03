`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 12/01/2024 05:52:25 PM
// Design Name: 
// Module Name: add_sub_8_bit
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


module add_sub_8_bit(
    input[7:0] A,
    input[7:0] B,
    output[7:0] S,
    input cin,
    input sub,
    output cout
    );
    
    wire[7:0] I;
    
    generate genvar i;
    for (i = 0; i < 8; i = i + 1) 
        assign I[i] = B[i] ^ sub;
    endgenerate;
    wire s = cin ^ sub;
    adder_8_bit adder(A, I, S, s, cout);
endmodule
