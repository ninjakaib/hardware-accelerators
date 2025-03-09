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

    wire[32:0] tmp22;
    wire[31:0] tmp23;

    // Combinational
    assign result = tmp23;
    assign tmp22 = a_reg + b_reg;
    assign tmp23 = {tmp22[31], tmp22[30], tmp22[29], tmp22[28], tmp22[27], tmp22[26], tmp22[25], tmp22[24], tmp22[23], tmp22[22], tmp22[21], tmp22[20], tmp22[19], tmp22[18], tmp22[17], tmp22[16], tmp22[15], tmp22[14], tmp22[13], tmp22[12], tmp22[11], tmp22[10], tmp22[9], tmp22[8], tmp22[7], tmp22[6], tmp22[5], tmp22[4], tmp22[3], tmp22[2], tmp22[1], tmp22[0]};

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

