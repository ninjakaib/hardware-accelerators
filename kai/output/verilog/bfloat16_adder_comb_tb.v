`include "bfloat16_adder_combinatorial.v"

module tb();
    reg clk;
    reg rst;
    reg[15:0] float_a;
    reg[15:0] float_b;
    reg[15:0] gt;
    wire[15:0] float_out;

    toplevel block(.clk(clk), .rst(rst), .float_a(float_a), .float_b(float_b), .gt(gt), .float_out(float_out));

    always
        #5 clk = ~clk;

    initial begin
        $dumpfile ("bf16_adder_comb.vcd");
        $dumpvars;

        clk = 0;
        rst = 0;
        float_a = 16'd16002;
        float_b = 16'd16198;
        gt = 16'd16260;

        #10
        $display("%d", float_out);
        float_a = 16'd16222;
        float_b = 16'd16152;
        gt = 16'd16315;

        #10
        $display("%d", float_out);
        float_a = 16'd16212;
        float_b = 16'd16133;
        gt = 16'd16300;

        #10
        $display("%d", float_out);
        float_a = 16'd16224;
        float_b = 16'd15984;
        gt = 16'd16270;

        #10
        $display("%d", float_out);
        float_a = 16'd16224;
        float_b = 16'd16024;
        gt = 16'd16278;

        #10
        $display("%d", float_out);
        float_a = 16'd15520;
        float_b = 16'd16131;
        gt = 16'd16136;

        #10
        $display("%d", float_out);
        float_a = 16'd16082;
        float_b = 16'd16232;
        gt = 16'd16296;

        #10
        $display("%d", float_out);
        float_a = 16'd16048;
        float_b = 16'd16090;
        gt = 16'd16197;

        #10
        $display("%d", float_out);
        float_a = 16'd15912;
        float_b = 16'd15488;
        gt = 16'd15928;

        #10
        $display("%d", float_out);
        float_a = 16'd16128;
        float_b = 16'd16139;
        gt = 16'd16262;

        #10
        $display("%d", float_out);
        float_a = 16'd16020;
        float_b = 16'd16212;
        gt = 16'd16271;

        #10
        $display("%d", float_out);
        float_a = 16'd15744;
        float_b = 16'd15824;
        gt = 16'd15912;

        #10
        $display("%d", float_out);
        float_a = 16'd15232;
        float_b = 16'd16234;
        gt = 16'd16235;

        #10
        $display("%d", float_out);
        float_a = 16'd16210;
        float_b = 16'd16192;
        gt = 16'd16329;

        #10
        $display("%d", float_out);
        float_a = 16'd15880;
        float_b = 16'd16176;
        gt = 16'd16210;

        #10
        $display("%d", float_out);
        float_a = 16'd16242;
        float_b = 16'd15916;
        gt = 16'd16270;

        #10
        $display("%d", float_out);
        float_a = 16'd15964;
        float_b = 16'd16068;
        gt = 16'd16153;

        #10
        $display("%d", float_out);
        float_a = 16'd15840;
        float_b = 16'd15956;
        gt = 16'd16034;

        #10
        $display("%d", float_out);
        float_a = 16'd16252;
        float_b = 16'd16137;
        gt = 16'd16322;

        #10
        $display("%d", float_out);
        float_a = 16'd15232;
        float_b = 16'd16094;
        gt = 16'd16096;

        #10
        $display("%d", float_out);
        float_a = 16'd16222;
        float_b = 16'd16170;
        gt = 16'd16324;

        #10
        $display("%d", float_out);
        float_a = 16'd16153;
        float_b = 16'd16140;
        gt = 16'd16274;

        #10
        $display("%d", float_out);
        float_a = 16'd16151;
        float_b = 16'd16235;
        gt = 16'd16321;

        #10
        $display("%d", float_out);
        float_a = 16'd16245;
        float_b = 16'd15972;
        gt = 16'd16279;

        #10
        $display("%d", float_out);
        float_a = 16'd15712;
        float_b = 16'd16133;
        gt = 16'd16147;

        #10
        $display("%d", float_out);
        float_a = 16'd15664;
        float_b = 16'd15876;
        gt = 16'd15920;

        #10
        $display("%d", float_out);
        float_a = 16'd16208;
        float_b = 16'd16194;
        gt = 16'd16329;

        #10
        $display("%d", float_out);
        float_a = 16'd16014;
        float_b = 16'd16192;
        gt = 16'd16260;

        #10
        $display("%d", float_out);
        float_a = 16'd16202;
        float_b = 16'd15952;
        gt = 16'd16254;

        #10
        $display("%d", float_out);
        float_a = 16'd15960;
        float_b = 16'd16175;
        gt = 16'd16229;

        #10
        $display("%d", float_out);
        float_a = 16'd15752;
        float_b = 16'd15948;
        gt = 16'd16008;

        #10
        $display("%d", float_out);
        float_a = 16'd16038;
        float_b = 16'd16237;
        gt = 16'd16288;

        #10
        $display("%d", float_out);
        float_a = 16'd16000;
        float_b = 16'd16253;
        gt = 16'd16286;

        #10
        $display("%d", float_out);
        float_a = 16'd16068;
        float_b = 16'd16230;
        gt = 16'd16292;

        #10
        $display("%d", float_out);
        float_a = 16'd15696;
        float_b = 16'd15888;
        gt = 16'd15940;

        #10
        $display("%d", float_out);
        float_a = 16'd16102;
        float_b = 16'd16182;
        gt = 16'd16276;

        #10
        $display("%d", float_out);
        float_a = 16'd16246;
        float_b = 16'd16143;
        gt = 16'd16322;

        #10
        $display("%d", float_out);
        float_a = 16'd16024;
        float_b = 16'd16170;
        gt = 16'd16246;

        #10
        $display("%d", float_out);
        float_a = 16'd16192;
        float_b = 16'd15912;
        gt = 16'd16234;

        #10
        $display("%d", float_out);
        float_a = 16'd16220;
        float_b = 16'd16156;
        gt = 16'd16316;

        #10
        $display("%d", float_out);
        float_a = 16'd16006;
        float_b = 16'd16224;
        gt = 16'd16274;

        #10
        $display("%d", float_out);
        float_a = 16'd16194;
        float_b = 16'd16251;
        gt = 16'd16350;

        #10
        $display("%d", float_out);
        float_a = 16'd16216;
        float_b = 16'd15912;
        gt = 16'd16257;

        #10
        $display("%d", float_out);
        float_a = 16'd15616;
        float_b = 16'd16135;
        gt = 16'd16143;

        #10
        $display("%d", float_out);
        float_a = 16'd16118;
        float_b = 16'd16166;
        gt = 16'd16272;

        #10
        $display("%d", float_out);
        float_a = 16'd15744;
        float_b = 16'd16112;
        gt = 16'd16136;

        #10
        $display("%d", float_out);
        float_a = 16'd16024;
        float_b = 16'd16218;
        gt = 16'd16275;

        #10
        $display("%d", float_out);
        float_a = 16'd16135;
        float_b = 16'd15768;
        gt = 16'd16154;

        #10
        $display("%d", float_out);
        float_a = 16'd16104;
        float_b = 16'd16044;
        gt = 16'd16202;

        #10
        $display("%d", float_out);
        float_a = 16'd16162;
        float_b = 16'd15884;
        gt = 16'd16197;

        #10
        $display("%d", float_out);
        float_a = 16'd16036;
        float_b = 16'd15728;
        gt = 16'd16066;

        #10
        $display("%d", float_out);
        float_a = 16'd16126;
        float_b = 16'd16034;
        gt = 16'd16208;

        #10
        $display("%d", float_out);
        float_a = 16'd16124;
        float_b = 16'd16244;
        gt = 16'd16313;

        #10
        $display("%d", float_out);
        float_a = 16'd16194;
        float_b = 16'd15884;
        gt = 16'd16229;

        #10
        $display("%d", float_out);
        float_a = 16'd16142;
        float_b = 16'd16068;
        gt = 16'd16240;

        #10
        $display("%d", float_out);
        float_a = 16'd15972;
        float_b = 16'd16020;
        gt = 16'd16131;

        #10
        $display("%d", float_out);
        float_a = 16'd16074;
        float_b = 16'd16002;
        gt = 16'd16166;

        #10
        $display("%d", float_out);
        float_a = 16'd0;
        float_b = 16'd16214;
        gt = 16'd16214;

        #10
        $display("%d", float_out);
        float_a = 16'd16138;
        float_b = 16'd16194;
        gt = 16'd16294;

        #10
        $display("%d", float_out);
        float_a = 16'd16143;
        float_b = 16'd16188;
        gt = 16'd16294;

        #10
        $display("%d", float_out);
        float_a = 16'd16205;
        float_b = 16'd16147;
        gt = 16'd16304;

        #10
        $display("%d", float_out);
        float_a = 16'd16206;
        float_b = 16'd16139;
        gt = 16'd16300;

        #10
        $display("%d", float_out);
        float_a = 16'd15876;
        float_b = 16'd16084;
        gt = 16'd16139;

        #10
        $display("%d", float_out);
        float_a = 16'd15884;
        float_b = 16'd16206;
        gt = 16'd16241;

        #10
        $display("%d", float_out);
        float_a = 16'd16192;
        float_b = 16'd16130;
        gt = 16'd16289;

        #10
        $display("%d", float_out);
        float_a = 16'd16012;
        float_b = 16'd16142;
        gt = 16'd16212;

        #10
        $display("%d", float_out);
        float_a = 16'd15752;
        float_b = 16'd16180;
        gt = 16'd16197;

        #10
        $display("%d", float_out);
        float_a = 16'd16050;
        float_b = 16'd16076;
        gt = 16'd16191;

        #10
        $display("%d", float_out);
        float_a = 16'd16032;
        float_b = 16'd16020;
        gt = 16'd16154;

        #10
        $display("%d", float_out);
        float_a = 16'd15864;
        float_b = 16'd16171;
        gt = 16'd16202;

        #10
        $display("%d", float_out);
        float_a = 16'd16092;
        float_b = 16'd16236;
        gt = 16'd16301;

        #10
        $display("%d", float_out);
        float_a = 16'd16084;
        float_b = 16'd16235;
        gt = 16'd16298;

        #10
        $display("%d", float_out);
        float_a = 16'd16034;
        float_b = 16'd16052;
        gt = 16'd16171;

        #10
        $display("%d", float_out);
        float_a = 16'd16210;
        float_b = 16'd16032;
        gt = 16'd16273;

        #10
        $display("%d", float_out);
        float_a = 16'd16026;
        float_b = 16'd16090;
        gt = 16'd16186;

        #10
        $display("%d", float_out);
        float_a = 16'd15864;
        float_b = 16'd16147;
        gt = 16'd16178;

        #10
        $display("%d", float_out);
        float_a = 16'd16140;
        float_b = 16'd16243;
        gt = 16'd16320;

        #10
        $display("%d", float_out);
        float_a = 16'd15752;
        float_b = 16'd16090;
        gt = 16'd16124;

        #10
        $display("%d", float_out);
        float_a = 16'd16118;
        float_b = 16'd16252;
        gt = 16'd16316;

        #10
        $display("%d", float_out);
        float_a = 16'd15920;
        float_b = 16'd16062;
        gt = 16'd16139;

        #10
        $display("%d", float_out);
        float_a = 16'd16106;
        float_b = 16'd16167;
        gt = 16'd16270;

        #10
        $display("%d", float_out);
        float_a = 16'd16139;
        float_b = 16'd16016;
        gt = 16'd16211;

        #10
        $display("%d", float_out);
        float_a = 16'd16092;
        float_b = 16'd15944;
        gt = 16'd16160;

        #10
        $display("%d", float_out);
        float_a = 16'd16074;
        float_b = 16'd16032;
        gt = 16'd16181;

        #10
        $display("%d", float_out);
        float_a = 16'd16134;
        float_b = 16'd16032;
        gt = 16'd16214;

        #10
        $display("%d", float_out);
        float_a = 16'd15920;
        float_b = 16'd16112;
        gt = 16'd16164;

        #10
        $display("%d", float_out);
        float_a = 16'd16237;
        float_b = 16'd16233;
        gt = 16'd16363;

        #10
        $display("%d", float_out);
        float_a = 16'd15920;
        float_b = 16'd16216;
        gt = 16'd16258;

        #10
        $display("%d", float_out);
        float_a = 16'd15848;
        float_b = 16'd16230;
        gt = 16'd16258;

        #10
        $display("%d", float_out);
        float_a = 16'd15776;
        float_b = 16'd16094;
        gt = 16'd16131;

        #10
        $display("%d", float_out);
        float_a = 16'd15872;
        float_b = 16'd16108;
        gt = 16'd16150;

        #10
        $display("%d", float_out);
        float_a = 16'd16184;
        float_b = 16'd16235;
        gt = 16'd16338;

        #10
        $display("%d", float_out);
        float_a = 16'd16048;
        float_b = 16'd16246;
        gt = 16'd16295;

        #10
        $display("%d", float_out);
        float_a = 16'd16198;
        float_b = 16'd16182;
        gt = 16'd16318;

        #10
        $display("%d", float_out);
        float_a = 16'd16133;
        float_b = 16'd16227;
        gt = 16'd16308;

        #10
        $display("%d", float_out);
        float_a = 16'd16110;
        float_b = 16'd16189;
        gt = 16'd16282;

        #10
        $display("%d", float_out);
        float_a = 16'd16132;
        float_b = 16'd16004;
        gt = 16'd16198;

        #10
        $display("%d", float_out);
        float_a = 16'd16114;
        float_b = 16'd15520;
        gt = 16'd16124;

        #10
        $display("%d", float_out);
        float_a = 16'd16231;
        float_b = 16'd16204;
        gt = 16'd16346;

        #10
        $display("%d", float_out);
        float_a = 16'd16152;
        float_b = 16'd15424;
        gt = 16'd16155;

        #10
        $display("%d", float_out);
        $finish;
    end
endmodule
