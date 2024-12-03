// Generated automatically via PyRTL
// As one initial test of synthesis, map to FPGA with:
//   yosys -p "synth_xilinx -top toplevel" thisfile.v

module toplevel(clk, rst, a_in, b_in, clear, write_en, out);
    input clk;
    input rst;
    input[7:0] a_in;
    input[7:0] b_in;
    input clear;
    input write_en;
    output[15:0] out;

    reg[7:0] a_reg = 8'd0;
    reg[15:0] accumulator = 16'd0;
    reg[7:0] b_reg = 8'd0;
    reg[15:0] product_reg = 16'd0;

    wire[15:0] tmp6;
    wire[16:0] tmp7;
    wire[15:0] tmp8;
    wire tmp9;
    wire[15:0] tmp10;
    wire[15:0] tmp11;

    // Combinational
    assign out = accumulator;
    assign tmp6 = a_reg * b_reg;
    assign tmp7 = product_reg + accumulator;
    assign tmp8 = {tmp7[15], tmp7[14], tmp7[13], tmp7[12], tmp7[11], tmp7[10], tmp7[9], tmp7[8], tmp7[7], tmp7[6], tmp7[5], tmp7[4], tmp7[3], tmp7[2], tmp7[1], tmp7[0]};
    assign tmp9 = ~write_en;
    assign tmp10 = write_en ? tmp8 : accumulator;
    assign tmp11 = tmp9 ? accumulator : tmp10;

    // Registers
    always @(posedge clk)
    begin
        if (rst) begin
            a_reg <= 0;
            accumulator <= 0;
            b_reg <= 0;
            product_reg <= 0;
        end
        else begin
            a_reg <= a_in;
            accumulator <= tmp11;
            b_reg <= b_in;
            product_reg <= tmp6;
        end
    end

endmodule

