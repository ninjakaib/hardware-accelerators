`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 12/01/2024 05:38:58 PM
// Design Name: 
// Module Name: tbbarrel_shift
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


module tbbarrel_shift;

    reg [6:0] data;
    reg [2:0] amt;
    wire [6:0] out;

    barrel_shift uut (
        .data(data),
        .amt(amt),
        .out(out)
    );

    initial begin
        $monitor("Time: %t | data = %b | amt = %b | out = %b", $time, data, amt, out);
        
        data = 7'b1111111; amt = 3'b000; #10; // No shift
        data = 7'b1111111; amt = 3'b001; #10; // Shift 1 bit to the right
        data = 7'b1111111; amt = 3'b010; #10; // Shift 2 bits to the right
        data = 7'b1111111; amt = 3'b011; #10; // Shift 3 bits to the right
        data = 7'b1111111; amt = 3'b100; #10; // Shift 4 bits to the right
        data = 7'b1111111; amt = 3'b101; #10; // Shift 5 bits to the right
        data = 7'b1111111; amt = 3'b110; #10; // Shift 6 bits to the right

        data = 7'b11110000; amt = 3'b010; #10; // Shift 2 bits to the right
        
        $finish;
    end

endmodule
