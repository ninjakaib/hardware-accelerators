// Generated automatically via PyRTL
// As one initial test of synthesis, map to FPGA with:
//   yosys -p "synth_xilinx -top toplevel" thisfile.v

module toplevel(clk, rst, a, b, result);
    input clk;
    input rst;
    input[15:0] a;
    input[15:0] b;
    output[15:0] result;

    wire[16:0] tmp2;
    wire[15:0] tmp3;

    // Combinational
    assign result = tmp3;
    assign tmp2 = a + b;
    assign tmp3 = {tmp2[15], tmp2[14], tmp2[13], tmp2[12], tmp2[11], tmp2[10], tmp2[9], tmp2[8], tmp2[7], tmp2[6], tmp2[5], tmp2[4], tmp2[3], tmp2[2], tmp2[1], tmp2[0]};

endmodule

