`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 12/03/2024 02:17:39 PM
// Design Name: 
// Module Name: PE
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


module PE(
    input[15:0] A,
    input[15:0] B,
    input clock,
    input reset,
    output reg [15:0] S,
    output reg [15:0] A_out,
    output reg [15:0] B_out
    );
    
    wire[15:0] P, t;        
    reg[15:0] T = 16'b0000000000000000;
    
    lmul mul (
        .A(A),
        .B(B),
        .P(P)
    );

    bf16_adder add (
        .A(S),
        .B(P),
        .S(t)
    );
    
    always@(posedge clock) 
    begin
    if(reset)    
      S <= 16'b0000000000000000;
    else 
        T <= t;
        #1
        S <= T;
        A_out <= A;
        B_out <=B;
    end
    
endmodule
