`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 12/04/2024 09:29:43 AM
// Design Name: 
// Module Name: systolic
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


module systolic(
        input[15:0] top_00, top_01, top_10, top_11, left_00, left_01, left_10, left_11,
        input clock,
        input reset,
        output p_00, p_01, p_10, p_11,
        output reg done,
        output reg [3:0] count
    );
    
    
    
    reg[15:0] top_in_0, top_in_1, left_in_0, left_in_1;
    wire[15:0] top_00to01, top_10to11, left_00to10, left_01to11;
    reg[15:0] zero = 16'b0000000000000000;
    wire[15:0] gnd;
    
    always @(posedge clock) begin
		if(reset) begin
			done <= 0;
			count <= 0;
		end
		else begin
			if(count == 3) begin
				done <= 1;
				count <= 0;
			end
			else begin
				done <= 0;
				count <= count + 1;
			end
		end
		if (count == 0) begin
            top_in_0 <= top_00;
            top_in_1 <= zero;
            left_in_0 <= left_00;
            left_in_1 <= zero;
        end else if (count == 1) begin
            top_in_0 <= top_10;
            top_in_1 <= top_01;
            left_in_0 <= left_10;
            left_in_1 <= left_01;
        end else if (count == 2) begin
            top_in_0 <= zero;
            top_in_1 <= top_11;
            left_in_0 <= zero;
            left_in_1 <= left_11;
        end	
	end
    
    
    PE P00(top_in_0, left_in_0, clock, reset, p_00, top_00to01, left_00to10);
    PE P10(top_in_1, left_00to10, clock, reset, p_10, top_10to11, gnd);
    PE P01(top_00to01, left_in_1, clock, reset, p_01, gnd, left_01to11);
    PE P11(top_10to11, left_01to11, clock, reset, p_11, gnd, gnd);
    
    
    
endmodule
