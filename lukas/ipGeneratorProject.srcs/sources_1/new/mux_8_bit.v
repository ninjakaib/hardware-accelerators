`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 12/01/2024 09:26:04 PM
// Design Name: 
// Module Name: mux_8_bit
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


module mux_8_bit(
    input [7:0] a,  
    input [7:0] b,  
    input s,         
    output [7:0] o  
);

    generate
        genvar i;
        for (i = 0; i < 8; i = i + 1) begin : mux_loop
            mux single_bit_mux (
                .a(a[i]),
                .b(b[i]),
                .s(s),
                .o(o[i])
            );
        end
    endgenerate
endmodule
