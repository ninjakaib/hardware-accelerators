`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 12/01/2024 07:54:39 PM
// Design Name: 
// Module Name: tadd_sub
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


module tadd_sub(    );

    wire [7:0] P;
    reg [7:0] A;
    reg [7:0] B;
    reg sub;
    reg cin = 0;
    wire cout;
    
    add_sub_8_bit dut(A,B,P,cin,sub,cout);
    initial
    begin
        A = 8'b00000011; B = 16'b00000101; sub = 0;
    #10 A = 8'b00000011; B = 16'b00000101; sub = 1;
    #10 $finish;    
    end
endmodule
