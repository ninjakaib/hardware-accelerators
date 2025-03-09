// Generated automatically via PyRTL
// As one initial test of synthesis, map to FPGA with:
//   yosys -p "synth_xilinx -top toplevel" thisfile.v

module toplevel(clk, rst, a, b, result);
    input clk;
    input rst;
    input[31:0] a;
    input[31:0] b;
    output[31:0] result;

    wire[63:0] tmp14;
    wire[31:0] tmp15;

    // Combinational
    assign result = tmp15;
    assign tmp14 = a * b;
    assign tmp15 = {tmp14[31], tmp14[30], tmp14[29], tmp14[28], tmp14[27], tmp14[26], tmp14[25], tmp14[24], tmp14[23], tmp14[22], tmp14[21], tmp14[20], tmp14[19], tmp14[18], tmp14[17], tmp14[16], tmp14[15], tmp14[14], tmp14[13], tmp14[12], tmp14[11], tmp14[10], tmp14[9], tmp14[8], tmp14[7], tmp14[6], tmp14[5], tmp14[4], tmp14[3], tmp14[2], tmp14[1], tmp14[0]};

endmodule

