`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 11/05/2024 06:35:18 PM
// Design Name: 
// Module Name: lmul
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


module lmul (
    input [15:0] A,
    input [15:0] B,
    output [15:0] P    
    );
    
    wire s0 = A[15];
    wire s1 = B[15];
    wire g = 0;
    wire h = 1;
    
    wire [15:0] S;
    wire [15:0] I;
    integer offset = 16248;
    
    adder_16_bit Add(A, B, S, g, g);
    subtractor_16_bit Sub(S, offset, I, h, g);
    
    assign I[15] = s0 ^ s1;
    
    wire o;
    wire [7:0] A_e = A [14:7];
    wire [7:0] B_e = B [14:7];
    
    zero_check Check(A_e, B_e, o);
        
    generate genvar i;
    for(i=0; i<16; i = i + 1) 
        assign P[i] = o & I[i];
    
    endgenerate;  
endmodule
