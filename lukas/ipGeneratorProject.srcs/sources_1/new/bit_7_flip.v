`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 12/03/2024 09:30:59 AM
// Design Name: 
// Module Name: bit_7_flip
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


module bit_7_flip(
    input[6:0] in,
    output[6:0] out
    );
    
    generate genvar i;
    for (i = 0; i < 7; i = i + 1) begin
        assign out[6 - i] = in[i];
    end
    endgenerate;
endmodule
