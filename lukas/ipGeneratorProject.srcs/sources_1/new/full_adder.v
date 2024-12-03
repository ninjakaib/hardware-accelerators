`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 11/05/2024 01:23:43 PM
// Design Name: 
// Module Name: full_adder
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


module full_adder(
    input cin,a,b,
    output sum,cout
    );
    
    wire s1,c1,c2;
    
    half_adder HA1(a,b,s1,c1);
    half_adder HA2(s1,cin,sum,c2);
    assign cout = c1 | c2;
    
endmodule
