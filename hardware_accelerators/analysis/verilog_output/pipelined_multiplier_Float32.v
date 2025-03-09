// Generated automatically via PyRTL
// As one initial test of synthesis, map to FPGA with:
//   yosys -p "synth_xilinx -top toplevel" thisfile.v

module toplevel(clk, rst, a, b, result);
    input clk;
    input rst;
    input[31:0] a;
    input[31:0] b;
    output[31:0] result;

    reg[31:0] a_reg = 32'd0;
    reg[31:0] b_reg = 32'd0;

    wire[63:0] tmp30;
    wire[31:0] tmp31;

    // Combinational
    assign result = tmp31;
    assign tmp30 = a_reg * b_reg;
    assign tmp31 = {tmp30[31], tmp30[30], tmp30[29], tmp30[28], tmp30[27], tmp30[26], tmp30[25], tmp30[24], tmp30[23], tmp30[22], tmp30[21], tmp30[20], tmp30[19], tmp30[18], tmp30[17], tmp30[16], tmp30[15], tmp30[14], tmp30[13], tmp30[12], tmp30[11], tmp30[10], tmp30[9], tmp30[8], tmp30[7], tmp30[6], tmp30[5], tmp30[4], tmp30[3], tmp30[2], tmp30[1], tmp30[0]};

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

