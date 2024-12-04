`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 12/02/2024 08:20:59 PM
// Design Name: 
// Module Name: tshift_Deconder
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


module tshift_Deconder(    );
    
    reg [6:0] data;
    wire [6:0] D;
    wire [2:0] B;

    shift_detector uut (
        .A(data),
        .D(D),
        .B(B)
        
    );

    initial begin
        $monitor("Time: %t | data = %b | D = %b | B = %b", $time, data, D, B);
        
        data = 7'b1111111;  #10; // No shift
        data = 7'b0111111;  #10; // Shift 1 bit to the right
        data = 7'b0011111;  #10; // Shift 2 bits to the right
        data = 7'b0001111;  #10; // Shift 3 bits to the right
        data = 7'b0000111;  #10; // Shift 4 bits to the right
        data = 7'b0000011;  #10; // Shift 5 bits to the right
        data = 7'b00000001;  #10; // Shift 6 bits to the right

        //data = 7'b11110000; amt = 3'b010; #10; // Shift 2 bits to the right
        
        $finish;
    end
endmodule
