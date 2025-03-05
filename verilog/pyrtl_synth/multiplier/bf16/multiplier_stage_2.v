// Generated automatically via PyRTL
// As one initial test of synthesis, map to FPGA with:
//   yosys -p "synth_xilinx -top toplevel" thisfile.v

module toplevel(clk, rst, float_a, float_b, out7712, out7714, out7715);
    input clk;
    input rst;
    input[15:0] float_a;
    input[15:0] float_b;
    output out7712;
    output[8:0] out7714;
    output[15:0] out7715;

    wire const_1534_1;
    wire const_1535_1;
    wire tmp7698;
    wire tmp7699;
    wire tmp7700;
    wire tmp7701;
    wire[7:0] tmp7702;
    wire[7:0] tmp7703;
    wire[7:0] tmp7704;
    wire[7:0] tmp7705;
    wire[7:0] tmp7706;
    wire[7:0] tmp7707;
    wire[6:0] tmp7708;
    wire[7:0] tmp7709;
    wire[6:0] tmp7710;
    wire[7:0] tmp7711;
    wire tmp7712;
    wire tmp7713;
    wire[8:0] tmp7714;
    wire[15:0] tmp7715;
    wire[8:0] tmp7716;
    wire[15:0] tmp7717;

    // Combinational
    assign const_1534_1 = 1;
    assign const_1535_1 = 1;
    assign out7712 = tmp7712;
    assign out7714 = tmp7714;
    assign out7715 = tmp7715;
    assign tmp7698 = tmp7700;
    assign tmp7699 = tmp7701;
    assign tmp7700 = {float_a[15]};
    assign tmp7701 = {float_b[15]};
    assign tmp7702 = tmp7704;
    assign tmp7703 = tmp7705;
    assign tmp7704 = {float_a[14], float_a[13], float_a[12], float_a[11], float_a[10], float_a[9], float_a[8], float_a[7]};
    assign tmp7705 = {float_b[14], float_b[13], float_b[12], float_b[11], float_b[10], float_b[9], float_b[8], float_b[7]};
    assign tmp7706 = tmp7709;
    assign tmp7707 = tmp7711;
    assign tmp7708 = {float_a[6], float_a[5], float_a[4], float_a[3], float_a[2], float_a[1], float_a[0]};
    assign tmp7709 = {const_1534_1, tmp7708};
    assign tmp7710 = {float_b[6], float_b[5], float_b[4], float_b[3], float_b[2], float_b[1], float_b[0]};
    assign tmp7711 = {const_1535_1, tmp7710};
    assign tmp7712 = tmp7713;
    assign tmp7713 = tmp7698 ^ tmp7699;
    assign tmp7714 = tmp7716;
    assign tmp7715 = tmp7717;
    assign tmp7716 = tmp7702 + tmp7703;
    assign tmp7717 = tmp7706 * tmp7707;

endmodule

