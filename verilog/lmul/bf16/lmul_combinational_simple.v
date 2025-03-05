// Generated automatically via PyRTL
// As one initial test of synthesis, map to FPGA with:
//   yosys -p "synth_xilinx -top toplevel" thisfile.v

module toplevel(clk, rst, float_a, float_b, out12109);
    input clk;
    input rst;
    input[15:0] float_a;
    input[15:0] float_b;
    output[15:0] out12109;

    wire[14:0] const_2050_16248;
    wire const_2051_0;
    wire tmp12100;
    wire tmp12101;
    wire tmp12102;
    wire[14:0] tmp12103;
    wire[14:0] tmp12104;
    wire[15:0] tmp12105;
    wire tmp12106;
    wire[15:0] tmp12107;
    wire[16:0] tmp12108;
    wire[15:0] tmp12109;
    wire[14:0] tmp12110;
    wire[15:0] tmp12111;

    // Combinational
    assign const_2050_16248 = 16248;
    assign const_2051_0 = 0;
    assign out12109 = tmp12109;
    assign tmp12100 = {float_a[15]};
    assign tmp12101 = {float_b[15]};
    assign tmp12102 = tmp12100 ^ tmp12101;
    assign tmp12103 = {float_a[14], float_a[13], float_a[12], float_a[11], float_a[10], float_a[9], float_a[8], float_a[7], float_a[6], float_a[5], float_a[4], float_a[3], float_a[2], float_a[1], float_a[0]};
    assign tmp12104 = {float_b[14], float_b[13], float_b[12], float_b[11], float_b[10], float_b[9], float_b[8], float_b[7], float_b[6], float_b[5], float_b[4], float_b[3], float_b[2], float_b[1], float_b[0]};
    assign tmp12105 = tmp12103 + tmp12104;
    assign tmp12106 = {const_2051_0};
    assign tmp12107 = {tmp12106, const_2050_16248};
    assign tmp12108 = tmp12105 - tmp12107;
    assign tmp12109 = tmp12111;
    assign tmp12110 = {tmp12108[14], tmp12108[13], tmp12108[12], tmp12108[11], tmp12108[10], tmp12108[9], tmp12108[8], tmp12108[7], tmp12108[6], tmp12108[5], tmp12108[4], tmp12108[3], tmp12108[2], tmp12108[1], tmp12108[0]};
    assign tmp12111 = {tmp12102, tmp12110};

endmodule

