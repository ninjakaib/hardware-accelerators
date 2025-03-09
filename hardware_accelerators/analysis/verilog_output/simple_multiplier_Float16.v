// Generated automatically via PyRTL
// As one initial test of synthesis, map to FPGA with:
//   yosys -p "synth_xilinx -top toplevel" thisfile.v

module toplevel(clk, rst, a, b, result);
    input clk;
    input rst;
    input[15:0] a;
    input[15:0] b;
    output[15:0] result;

    wire[31:0] tmp10;
    wire[15:0] tmp11;

    // Combinational
    assign result = tmp11;
    assign tmp10 = a * b;
    assign tmp11 = {tmp10[15], tmp10[14], tmp10[13], tmp10[12], tmp10[11], tmp10[10], tmp10[9], tmp10[8], tmp10[7], tmp10[6], tmp10[5], tmp10[4], tmp10[3], tmp10[2], tmp10[1], tmp10[0]};

endmodule

