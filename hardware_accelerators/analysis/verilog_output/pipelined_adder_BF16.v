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

    wire[16:0] tmp20;
    wire[15:0] tmp21;

    // Combinational
    assign result = tmp21;
    assign tmp20 = a_reg + b_reg;
    assign tmp21 = {tmp20[15], tmp20[14], tmp20[13], tmp20[12], tmp20[11], tmp20[10], tmp20[9], tmp20[8], tmp20[7], tmp20[6], tmp20[5], tmp20[4], tmp20[3], tmp20[2], tmp20[1], tmp20[0]};

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

