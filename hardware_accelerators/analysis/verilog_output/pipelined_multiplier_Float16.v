// Generated automatically via PyRTL
// As one initial test of synthesis, map to FPGA with:
//   yosys -p "synth_xilinx -top toplevel" thisfile.v

module toplevel(clk, rst, a, b, result);
    input clk;
    input rst;
    input[15:0] a;
    input[15:0] b;
    output[15:0] result;

    reg[15:0] a_reg = 16'd0;
    reg[15:0] b_reg = 16'd0;

    wire[31:0] tmp26;
    wire[15:0] tmp27;

    // Combinational
    assign result = tmp27;
    assign tmp26 = a_reg * b_reg;
    assign tmp27 = {tmp26[15], tmp26[14], tmp26[13], tmp26[12], tmp26[11], tmp26[10], tmp26[9], tmp26[8], tmp26[7], tmp26[6], tmp26[5], tmp26[4], tmp26[3], tmp26[2], tmp26[1], tmp26[0]};

    // Registers
    always @(posedge clk)
    begin
        if (rst) begin
            a_reg <= 0;
            b_reg <= 0;
        end
        else begin
            a_reg <= a;
            b_reg <= b;
        end
    end

endmodule

