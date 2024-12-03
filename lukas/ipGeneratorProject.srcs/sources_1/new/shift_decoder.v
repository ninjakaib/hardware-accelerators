`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 12/01/2024 05:14:25 PM
// Design Name: 
// Module Name: shift_decoder
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


module shift_decoder(
    input[2:0] count,
    output[6:0] shift
    );
    
    assign shift[0] = !count[0] & !count[1] & !count[2];
    assign shift[1] =  count[0] & !count[1] & !count[2];
    assign shift[2] = !count[0] &  count[1] & !count[2];
    assign shift[3] =  count[0] &  count[1] & !count[2];
    assign shift[4] = !count[0] & !count[1] &  count[2];
    assign shift[5] =  count[0] & !count[1] &  count[2];
    assign shift[6] = !count[0] &  count[1] &  count[2];
endmodule
