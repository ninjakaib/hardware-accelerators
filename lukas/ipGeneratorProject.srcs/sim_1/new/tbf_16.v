`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 12/03/2024 09:46:24 AM
// Design Name: 
// Module Name: tbf_16
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


module tbf_16(   );
    wire [15:0] sum;
    reg [15:0] A;
    reg [15:0] B;
    

    bf16_adder uut (
        .A(A),
        .B(B),
        .S(sum)
        
    );

    initial begin
        $monitor("Time: %t | A = %b | B = %b | sum = %b", $time, A, B, sum);
        
        A = 16'b0000000000000000; B = 16'b0000000000000000; #10; 
        
        $finish;
    end
endmodule
