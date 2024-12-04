`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 12/01/2024 05:50:17 PM
// Design Name: 
// Module Name: adder_8_bit
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


module adder_8_bit(
    input[7:0] A,
    input[7:0] B,
    output[7:0] S,
    input cin,
    output cout
    );
    
    
    generate genvar i;
    wire [8:0] carry;
    assign carry[0] = cin;
    
    
    for(i=0; i<8; i = i + 1) begin
        full_adder A(carry[i],A[i],B[i],S[i],carry[i + 1]);
    end
    endgenerate;    
    assign cout = carry[8];
endmodule
