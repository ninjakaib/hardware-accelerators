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

    wire[31:0] tmp28;
    wire[15:0] tmp29;

    // Combinational
    assign result = tmp29;
    assign tmp28 = a_reg * b_reg;
    assign tmp29 = {tmp28[15], tmp28[14], tmp28[13], tmp28[12], tmp28[11], tmp28[10], tmp28[9], tmp28[8], tmp28[7], tmp28[6], tmp28[5], tmp28[4], tmp28[3], tmp28[2], tmp28[1], tmp28[0]};

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

