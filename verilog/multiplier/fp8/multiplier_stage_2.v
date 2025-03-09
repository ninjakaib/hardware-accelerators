// Generated automatically via PyRTL
// As one initial test of synthesis, map to FPGA with:
//   yosys -p "synth_xilinx -top toplevel" thisfile.v

module toplevel(clk, rst, float_a, float_b, out2224, out2226, out2227);
    input clk;
    input rst;
    input[7:0] float_a;
    input[7:0] float_b;
    output out2224;
    output[4:0] out2226;
    output[7:0] out2227;

    wire const_466_1;
    wire const_467_1;
    wire tmp2210;
    wire tmp2211;
    wire tmp2212;
    wire tmp2213;
    wire[3:0] tmp2214;
    wire[3:0] tmp2215;
    wire[3:0] tmp2216;
    wire[3:0] tmp2217;
    wire[3:0] tmp2218;
    wire[3:0] tmp2219;
    wire[2:0] tmp2220;
    wire[3:0] tmp2221;
    wire[2:0] tmp2222;
    wire[3:0] tmp2223;
    wire tmp2224;
    wire tmp2225;
    wire[4:0] tmp2226;
    wire[7:0] tmp2227;
    wire[4:0] tmp2228;
    wire[7:0] tmp2229;

    // Combinational
    assign const_466_1 = 1;
    assign const_467_1 = 1;
    assign out2224 = tmp2224;
    assign out2226 = tmp2226;
    assign out2227 = tmp2227;
    assign tmp2210 = tmp2212;
    assign tmp2211 = tmp2213;
    assign tmp2212 = {float_a[7]};
    assign tmp2213 = {float_b[7]};
    assign tmp2214 = tmp2216;
    assign tmp2215 = tmp2217;
    assign tmp2216 = {float_a[6], float_a[5], float_a[4], float_a[3]};
    assign tmp2217 = {float_b[6], float_b[5], float_b[4], float_b[3]};
    assign tmp2218 = tmp2221;
    assign tmp2219 = tmp2223;
    assign tmp2220 = {float_a[2], float_a[1], float_a[0]};
    assign tmp2221 = {const_466_1, tmp2220};
    assign tmp2222 = {float_b[2], float_b[1], float_b[0]};
    assign tmp2223 = {const_467_1, tmp2222};
    assign tmp2224 = tmp2225;
    assign tmp2225 = tmp2210 ^ tmp2211;
    assign tmp2226 = tmp2228;
    assign tmp2227 = tmp2229;
    assign tmp2228 = tmp2214 + tmp2215;
    assign tmp2229 = tmp2218 * tmp2219;

endmodule

