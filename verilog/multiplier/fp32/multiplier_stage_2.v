// Generated automatically via PyRTL
// As one initial test of synthesis, map to FPGA with:
//   yosys -p "synth_xilinx -top toplevel" thisfile.v

module toplevel(clk, rst, float_a, float_b, out20850, out20852, out20853);
    input clk;
    input rst;
    input[31:0] float_a;
    input[31:0] float_b;
    output out20850;
    output[8:0] out20852;
    output[47:0] out20853;

    wire const_4175_1;
    wire const_4176_1;
    wire tmp20836;
    wire tmp20837;
    wire tmp20838;
    wire tmp20839;
    wire[7:0] tmp20840;
    wire[7:0] tmp20841;
    wire[7:0] tmp20842;
    wire[7:0] tmp20843;
    wire[23:0] tmp20844;
    wire[23:0] tmp20845;
    wire[22:0] tmp20846;
    wire[23:0] tmp20847;
    wire[22:0] tmp20848;
    wire[23:0] tmp20849;
    wire tmp20850;
    wire tmp20851;
    wire[8:0] tmp20852;
    wire[47:0] tmp20853;
    wire[8:0] tmp20854;
    wire[47:0] tmp20855;

    // Combinational
    assign const_4175_1 = 1;
    assign const_4176_1 = 1;
    assign out20850 = tmp20850;
    assign out20852 = tmp20852;
    assign out20853 = tmp20853;
    assign tmp20836 = tmp20838;
    assign tmp20837 = tmp20839;
    assign tmp20838 = {float_a[31]};
    assign tmp20839 = {float_b[31]};
    assign tmp20840 = tmp20842;
    assign tmp20841 = tmp20843;
    assign tmp20842 = {float_a[30], float_a[29], float_a[28], float_a[27], float_a[26], float_a[25], float_a[24], float_a[23]};
    assign tmp20843 = {float_b[30], float_b[29], float_b[28], float_b[27], float_b[26], float_b[25], float_b[24], float_b[23]};
    assign tmp20844 = tmp20847;
    assign tmp20845 = tmp20849;
    assign tmp20846 = {float_a[22], float_a[21], float_a[20], float_a[19], float_a[18], float_a[17], float_a[16], float_a[15], float_a[14], float_a[13], float_a[12], float_a[11], float_a[10], float_a[9], float_a[8], float_a[7], float_a[6], float_a[5], float_a[4], float_a[3], float_a[2], float_a[1], float_a[0]};
    assign tmp20847 = {const_4175_1, tmp20846};
    assign tmp20848 = {float_b[22], float_b[21], float_b[20], float_b[19], float_b[18], float_b[17], float_b[16], float_b[15], float_b[14], float_b[13], float_b[12], float_b[11], float_b[10], float_b[9], float_b[8], float_b[7], float_b[6], float_b[5], float_b[4], float_b[3], float_b[2], float_b[1], float_b[0]};
    assign tmp20849 = {const_4176_1, tmp20848};
    assign tmp20850 = tmp20851;
    assign tmp20851 = tmp20836 ^ tmp20837;
    assign tmp20852 = tmp20854;
    assign tmp20853 = tmp20855;
    assign tmp20854 = tmp20840 + tmp20841;
    assign tmp20855 = tmp20844 * tmp20845;

endmodule

