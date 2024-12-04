`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 12/01/2024 05:33:01 PM
// Design Name: 
// Module Name: barrel_shift
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 7-bit right barrel shifter
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module barrel_shift(
    input [6:0] data,    
    input [2:0] amt,     
    output [6:0] out     
);

    wire [6:0] s0, s1, s2;
    
    assign s0 = amt[0] ? {1'b0, data[6:1]} : data;
    assign s1 = amt[1] ? {2'b00, s0[6:2]} : s0;
    assign s2 = amt[2] ? {4'b0000, s1[6:4]} : s1;
    assign out = s2;

endmodule