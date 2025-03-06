// Generated automatically via PyRTL
// As one initial test of synthesis, map to FPGA with:
//   yosys -p "synth_xilinx -top toplevel" thisfile.v

module toplevel(clk, rst, a, b, result);
    input clk;
    input rst;
    input[15:0] a;
    input[15:0] b;
    output[15:0] result;

    wire[16:0] tmp4;
    wire[15:0] tmp5;

    // Combinational
    assign result = tmp5;
    assign tmp4 = a + b;
    assign tmp5 = {tmp4[15], tmp4[14], tmp4[13], tmp4[12], tmp4[11], tmp4[10], tmp4[9], tmp4[8], tmp4[7], tmp4[6], tmp4[5], tmp4[4], tmp4[3], tmp4[2], tmp4[1], tmp4[0]};

endmodule

