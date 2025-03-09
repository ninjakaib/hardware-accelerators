// Generated automatically via PyRTL
// As one initial test of synthesis, map to FPGA with:
//   yosys -p "synth_xilinx -top toplevel" thisfile.v

module toplevel(clk, rst, a, b, result);
    input clk;
    input rst;
    input[7:0] a;
    input[7:0] b;
    output[7:0] result;

    reg[7:0] a_reg = 8'd0;
    reg[7:0] b_reg = 8'd0;

    wire[15:0] tmp24;
    wire[7:0] tmp25;

    // Combinational
    assign result = tmp25;
    assign tmp24 = a_reg * b_reg;
    assign tmp25 = {tmp24[7], tmp24[6], tmp24[5], tmp24[4], tmp24[3], tmp24[2], tmp24[1], tmp24[0]};

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

