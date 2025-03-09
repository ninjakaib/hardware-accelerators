// Generated automatically via PyRTL
// As one initial test of synthesis, map to FPGA with:
//   yosys -p "synth_xilinx -top toplevel" thisfile.v

module toplevel(clk, rst, a, b, result);
    input clk;
    input rst;
    input[15:0] a;
    input[15:0] b;
    output[15:0] result;

    reg[15:0] a_reg = 16'd0;
    reg[15:0] b_reg = 16'd0;

    wire[16:0] tmp18;
    wire[15:0] tmp19;

    // Combinational
    assign result = tmp19;
    assign tmp18 = a_reg + b_reg;
    assign tmp19 = {tmp18[15], tmp18[14], tmp18[13], tmp18[12], tmp18[11], tmp18[10], tmp18[9], tmp18[8], tmp18[7], tmp18[6], tmp18[5], tmp18[4], tmp18[3], tmp18[2], tmp18[1], tmp18[0]};

    // Registers
    always @(posedge clk)
    begin
        if (rst) begin
            a_reg <= 0;
            b_reg <= 0;
        end
        else begin
            a_reg <= a;
            b_reg <= b;
        end
    end

endmodule

