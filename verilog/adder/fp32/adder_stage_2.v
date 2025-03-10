// Generated automatically via PyRTL
// As one initial test of synthesis, map to FPGA with:
//   yosys -p "synth_xilinx -top toplevel" thisfile.v

module toplevel(clk, rst, float_a, float_b, out16545, out16547, out16548, out16549, out16550);
    input clk;
    input rst;
    input[31:0] float_a;
    input[31:0] float_b;
    output out16545;
    output[8:0] out16547;
    output[7:0] out16548;
    output[23:0] out16549;
    output[23:0] out16550;

    wire const_2878_1;
    wire const_2879_1;
    wire const_2880_1;
    wire const_2881_0;
    wire const_2882_0;
    wire tmp16531;
    wire tmp16532;
    wire tmp16533;
    wire tmp16534;
    wire[7:0] tmp16535;
    wire[7:0] tmp16536;
    wire[7:0] tmp16537;
    wire[7:0] tmp16538;
    wire[23:0] tmp16539;
    wire[23:0] tmp16540;
    wire[22:0] tmp16541;
    wire[23:0] tmp16542;
    wire[22:0] tmp16543;
    wire[23:0] tmp16544;
    wire tmp16545;
    wire[8:0] tmp16546;
    wire[8:0] tmp16547;
    wire[7:0] tmp16548;
    wire[23:0] tmp16549;
    wire[23:0] tmp16550;
    wire tmp16551;
    wire[8:0] tmp16552;
    wire tmp16553;
    wire tmp16554;
    wire[7:0] tmp16555;
    wire[7:0] tmp16556;
    wire tmp16557;
    wire[7:0] tmp16558;
    wire[7:0] tmp16559;
    wire[6:0] tmp16560;
    wire[7:0] tmp16561;
    wire[8:0] tmp16562;
    wire[7:0] tmp16563;
    wire[8:0] tmp16564;
    wire tmp16565;
    wire[8:0] tmp16566;
    wire[8:0] tmp16567;
    wire tmp16568;
    wire[23:0] tmp16569;
    wire tmp16570;
    wire[23:0] tmp16571;

    // Combinational
    assign const_2878_1 = 1;
    assign const_2879_1 = 1;
    assign const_2880_1 = 1;
    assign const_2881_0 = 0;
    assign const_2882_0 = 0;
    assign out16545 = tmp16545;
    assign out16547 = tmp16547;
    assign out16548 = tmp16548;
    assign out16549 = tmp16549;
    assign out16550 = tmp16550;
    assign tmp16531 = tmp16533;
    assign tmp16532 = tmp16534;
    assign tmp16533 = {float_a[31]};
    assign tmp16534 = {float_b[31]};
    assign tmp16535 = tmp16537;
    assign tmp16536 = tmp16538;
    assign tmp16537 = {float_a[30], float_a[29], float_a[28], float_a[27], float_a[26], float_a[25], float_a[24], float_a[23]};
    assign tmp16538 = {float_b[30], float_b[29], float_b[28], float_b[27], float_b[26], float_b[25], float_b[24], float_b[23]};
    assign tmp16539 = tmp16542;
    assign tmp16540 = tmp16544;
    assign tmp16541 = {float_a[22], float_a[21], float_a[20], float_a[19], float_a[18], float_a[17], float_a[16], float_a[15], float_a[14], float_a[13], float_a[12], float_a[11], float_a[10], float_a[9], float_a[8], float_a[7], float_a[6], float_a[5], float_a[4], float_a[3], float_a[2], float_a[1], float_a[0]};
    assign tmp16542 = {const_2878_1, tmp16541};
    assign tmp16543 = {float_b[22], float_b[21], float_b[20], float_b[19], float_b[18], float_b[17], float_b[16], float_b[15], float_b[14], float_b[13], float_b[12], float_b[11], float_b[10], float_b[9], float_b[8], float_b[7], float_b[6], float_b[5], float_b[4], float_b[3], float_b[2], float_b[1], float_b[0]};
    assign tmp16544 = {const_2879_1, tmp16543};
    assign tmp16545 = tmp16551;
    assign tmp16546 = tmp16552;
    assign tmp16547 = tmp16567;
    assign tmp16548 = tmp16555;
    assign tmp16549 = tmp16569;
    assign tmp16550 = tmp16571;
    assign tmp16551 = tmp16531 ^ tmp16532;
    assign tmp16552 = tmp16535 - tmp16536;
    assign tmp16553 = {tmp16546[8]};
    assign tmp16554 = {tmp16546[8]};
    assign tmp16555 = tmp16554 ? tmp16536 : tmp16535;
    assign tmp16556 = {tmp16546[7], tmp16546[6], tmp16546[5], tmp16546[4], tmp16546[3], tmp16546[2], tmp16546[1], tmp16546[0]};
    assign tmp16557 = {tmp16546[8]};
    assign tmp16558 = {tmp16546[7], tmp16546[6], tmp16546[5], tmp16546[4], tmp16546[3], tmp16546[2], tmp16546[1], tmp16546[0]};
    assign tmp16559 = ~tmp16558;
    assign tmp16560 = {const_2881_0, const_2881_0, const_2881_0, const_2881_0, const_2881_0, const_2881_0, const_2881_0};
    assign tmp16561 = {tmp16560, const_2880_1};
    assign tmp16562 = tmp16559 + tmp16561;
    assign tmp16563 = {tmp16562[7], tmp16562[6], tmp16562[5], tmp16562[4], tmp16562[3], tmp16562[2], tmp16562[1], tmp16562[0]};
    assign tmp16564 = {tmp16557, tmp16563};
    assign tmp16565 = {const_2882_0};
    assign tmp16566 = {tmp16565, tmp16556};
    assign tmp16567 = tmp16553 ? tmp16564 : tmp16566;
    assign tmp16568 = {tmp16546[8]};
    assign tmp16569 = tmp16568 ? tmp16539 : tmp16540;
    assign tmp16570 = {tmp16546[8]};
    assign tmp16571 = tmp16570 ? tmp16540 : tmp16539;

endmodule

