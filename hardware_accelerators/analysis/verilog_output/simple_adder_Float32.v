// Generated automatically via PyRTL
// As one initial test of synthesis, map to FPGA with:
//   yosys -p "synth_xilinx -top toplevel" thisfile.v

module toplevel(clk, rst, a, b, result);
    input clk;
    input rst;
    input[31:0] a;
    input[31:0] b;
    output[31:0] result;

    wire[32:0] tmp6;
    wire[31:0] tmp7;

    // Combinational
    assign result = tmp7;
    assign tmp6 = a + b;
    assign tmp7 = {tmp6[31], tmp6[30], tmp6[29], tmp6[28], tmp6[27], tmp6[26], tmp6[25], tmp6[24], tmp6[23], tmp6[22], tmp6[21], tmp6[20], tmp6[19], tmp6[18], tmp6[17], tmp6[16], tmp6[15], tmp6[14], tmp6[13], tmp6[12], tmp6[11], tmp6[10], tmp6[9], tmp6[8], tmp6[7], tmp6[6], tmp6[5], tmp6[4], tmp6[3], tmp6[2], tmp6[1], tmp6[0]};

endmodule

