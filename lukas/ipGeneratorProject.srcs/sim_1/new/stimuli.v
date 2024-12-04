`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 11/05/2024 02:31:36 PM
// Design Name: 
// Module Name: stimuli
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


module stimuli(    );
    
    wire [15:0] P;
    reg [15:0] A;
    reg [15:0] B;
    
    lmul dut(A,B,P);
    initial
    begin
        A = 16'b0100000001000000; B = 16'b0100000001000000;
    // #10 A = 16'b1110100100000000; B = 16'b1111001000000000;
    #10 $finish;    
    end
endmodule
