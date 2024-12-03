`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 11/05/2024 02:43:27 PM
// Design Name: 
// Module Name: full_subtractor
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


module full_subtractor(
    input cin,a,b,
    output diff,cout
    );
    
    wire s1,c1,c2;
    
    half_subtractor HS1(a,b,s1,c1);
    half_subtractor HS2(s1,cin,diff,c2);
    assign cout = c1 | c2;
    
endmodule
