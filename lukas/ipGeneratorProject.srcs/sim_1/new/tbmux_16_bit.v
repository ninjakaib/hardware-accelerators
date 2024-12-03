`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 12/01/2024 04:04:35 PM
// Design Name: 
// Module Name: tbmux_16_bit
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: My testbench for 16-bit multiplexer :)
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module tbmux_16_bit;

    reg [15:0] a;
    reg [15:0] b;
    reg s;

    wire [15:0] o;

    mux_16_bit uut (
        .a(a),
        .b(b),
        .s(s),
        .o(o)
    );

    initial begin
    
    $display("Time\t a\t\t b\t\t s\t\t o");
    $monitor("%g\t %b\t %b\t %b\t %b", $time, a, b, s, o);

    // Test Case 1
    a = 16'b1010101010101010;  
    b = 16'b0101010101010101;  
    s = 0;                    // Select 'a'
    #10;                      

    // Test Case 2
    s = 1;                    // Select 'b'
    #10;                      

    // Test Case 3
    a = 16'b1111000011110000;  
    b = 16'b0000111100001111;  
    s = 0;                     // Select 'a'
    #10;                       

    // Test Case 4
    s = 1;                     // Select 'b'
    #10;                       

    $finish;                   // End the simulation
end

endmodule
