// Generated automatically via PyRTL
// As one initial test of synthesis, map to FPGA with:
//   yosys -p "synth_xilinx -top toplevel" thisfile.v

module toplevel(clk, rst, a, b, result);
    input clk;
    input rst;
    input[7:0] a;
    input[7:0] b;
    output[7:0] result;

    wire[8:0] tmp0;
    wire[7:0] tmp1;

    // Combinational
    assign result = tmp1;
    assign tmp0 = a + b;
    assign tmp1 = {tmp0[7], tmp0[6], tmp0[5], tmp0[4], tmp0[3], tmp0[2], tmp0[1], tmp0[0]};

endmodule

