`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 12/02/2024 08:15:44 PM
// Design Name: 
// Module Name: shift_detector
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


module shift_detector(
    input[6:0] A,
    output[6:0] D,
    output[2:0] B
    );
        
    wire[6:0] zeros;
    assign zeros[6] = !A[6];
    assign D[0] = A[6];
    generate genvar i;
    for (i = 5; i >= 0; i = i - 1) begin
        assign zeros[i] = zeros[i + 1] & !A[i];
        assign D[6 - i] = zeros[i + 1] & A[i];
    end
    endgenerate;
    
    assign B[0] = D[1] | D[3] | D[5];
    assign B[1] = D[2] | D[3] | D[6];
    assign B[2] = D[4] | D[5] | D[6];
        
endmodule
