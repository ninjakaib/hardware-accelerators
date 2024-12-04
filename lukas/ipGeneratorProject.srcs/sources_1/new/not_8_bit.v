`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 12/01/2024 09:39:01 PM
// Design Name: 
// Module Name: not_8_bit
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


module not_8_bit(
    input[7:0] A,
    output[7:0] O
    );
    
    generate genvar i;
    for (i = 0; i < 8; i = i + 1) 
        assign O[i] = !A[i];
    endgenerate;
endmodule
