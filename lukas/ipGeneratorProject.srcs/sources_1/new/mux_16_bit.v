`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 12/01/2024 03:047:00 PM
// Design Name: 
// Module Name: mux_16_bit
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 16-bit multiplexer that selects between two 16-bit inputs based on a control signal.
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module mux_16_bit(
    input [15:0] a,  
    input [15:0] b,  
    input s,         
    output [15:0] o  
);

    generate
        genvar i;
        for (i = 0; i < 16; i = i + 1) begin : mux_loop
            mux single_bit_mux (
                .a(a[i]),
                .b(b[i]),
                .s(s),
                .o(o[i])
            );
        end
    endgenerate

endmodule