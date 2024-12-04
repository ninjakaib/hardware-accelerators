`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 12/04/2024 09:03:32 AM
// Design Name: 
// Module Name: PET
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


module PET(    );
    wire [15:0] sum;
    reg [15:0] A;
    reg [15:0] B;
    reg clock;
    reg reset;

    PE uut (
        .A(A),
        .B(B),
        .S(sum),
        .clock(clock),
        .reset(reset)
        
    );

    initial begin
        $monitor("Time: %t | A = %b | B = %b | sum = %b | clock = %b | reset = %b", $time, A, B, sum, clock, reset);
        
        //A = 16'b0000000000000000; B = 16'b0000000000000000; #50; 
        //A = 16'b0000000001000000; B = 16'b0000000000000000; #50;
        A = 16'b0000000001110001; B = 16'b0000000001110001; #50;
        
        $finish;
    end
endmodule
