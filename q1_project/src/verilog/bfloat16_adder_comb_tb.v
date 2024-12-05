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
        float_a = 16'd15968;
        float_b = 16'd16150;
        gt = 16'd16206;

        #10
        $display("%d", float_out);
        float_a = 16'd16253;
        float_b = 16'd16137;
        gt = 16'd16323;

        #10
        $display("%d", float_out);
        float_a = 16'd15964;
        float_b = 16'd15728;
        gt = 16'd16012;

        #10
        $display("%d", float_out);
        float_a = 16'd16198;
        float_b = 16'd16227;
        gt = 16'd16340;

        #10
        $display("%d", float_out);
        float_a = 16'd16198;
        float_b = 16'd16171;
        gt = 16'd16312;

        #10
        $display("%d", float_out);
        float_a = 16'd16204;
        float_b = 16'd16052;
        gt = 16'd16275;

        #10
        $display("%d", float_out);
        float_a = 16'd16183;
        float_b = 16'd16199;
        gt = 16'd16319;

        #10
        $display("%d", float_out);
        float_a = 16'd15960;
        float_b = 16'd16226;
        gt = 16'd16268;

        #10
        $display("%d", float_out);
        float_a = 16'd16203;
        float_b = 16'd16008;
        gt = 16'd16264;

        #10
        $display("%d", float_out);
        float_a = 16'd15992;
        float_b = 16'd15904;
        gt = 16'd16076;

        #10
        $display("%d", float_out);
        float_a = 16'd16245;
        float_b = 16'd16155;
        gt = 16'd16328;

        #10
        $display("%d", float_out);
        float_a = 16'd16197;
        float_b = 16'd15840;
        gt = 16'd16225;

        #10
        $display("%d", float_out);
        float_a = 16'd16222;
        float_b = 16'd16040;
        gt = 16'd16281;

        #10
        $display("%d", float_out);
        float_a = 16'd16196;
        float_b = 16'd15360;
        gt = 16'd16198;

        #10
        $display("%d", float_out);
        float_a = 16'd15908;
        float_b = 16'd16116;
        gt = 16'd16163;

        #10
        $display("%d", float_out);
        float_a = 16'd16249;
        float_b = 16'd15968;
        gt = 16'd16280;

        #10
        $display("%d", float_out);
        float_a = 16'd15848;
        float_b = 16'd16062;
        gt = 16'd16120;

        #10
        $display("%d", float_out);
        float_a = 16'd16124;
        float_b = 16'd15816;
        gt = 16'd16151;

        #10
        $display("%d", float_out);
        float_a = 16'd16199;
        float_b = 16'd16140;
        gt = 16'd16298;

        #10
        $display("%d", float_out);
        float_a = 16'd16050;
        float_b = 16'd16234;
        gt = 16'd16290;

        #10
        $display("%d", float_out);
        float_a = 16'd15940;
        float_b = 16'd16217;
        gt = 16'd16261;

        #10
        $display("%d", float_out);
        float_a = 16'd15936;
        float_b = 16'd15696;
        gt = 16'd15988;

        #10
        $display("%d", float_out);
        float_a = 16'd15920;
        float_b = 16'd16231;
        gt = 16'd16266;

        #10
        $display("%d", float_out);
        float_a = 16'd16161;
        float_b = 16'd16178;
        gt = 16'd16298;

        #10
        $display("%d", float_out);
        float_a = 16'd16141;
        float_b = 16'd16227;
        gt = 16'd16312;

        #10
        $display("%d", float_out);
        float_a = 16'd16239;
        float_b = 16'd15964;
        gt = 16'd16275;

        #10
        $display("%d", float_out);
        float_a = 16'd16246;
        float_b = 16'd15948;
        gt = 16'd16276;

        #10
        $display("%d", float_out);
        float_a = 16'd16116;
        float_b = 16'd15976;
        gt = 16'd16180;

        #10
        $display("%d", float_out);
        float_a = 16'd16253;
        float_b = 16'd16202;
        gt = 16'd16356;

        #10
        $display("%d", float_out);
        float_a = 16'd16163;
        float_b = 16'd16187;
        gt = 16'd16303;

        #10
        $display("%d", float_out);
        float_a = 16'd16008;
        float_b = 16'd16213;
        gt = 16'd16268;

        #10
        $display("%d", float_out);
        float_a = 16'd16207;
        float_b = 16'd16136;
        gt = 16'd16300;

        #10
        $display("%d", float_out);
        float_a = 16'd16205;
        float_b = 16'd16058;
        gt = 16'd16277;

        #10
        $display("%d", float_out);
        float_a = 16'd16196;
        float_b = 16'd16016;
        gt = 16'd16262;

        #10
        $display("%d", float_out);
        float_a = 16'd16078;
        float_b = 16'd16248;
        gt = 16'd16304;

        #10
        $display("%d", float_out);
        float_a = 16'd16208;
        float_b = 16'd16248;
        gt = 16'd16356;

        #10
        $display("%d", float_out);
        float_a = 16'd15808;
        float_b = 16'd16152;
        gt = 16'd16176;

        #10
        $display("%d", float_out);
        float_a = 16'd16074;
        float_b = 16'd15988;
        gt = 16'd16162;

        #10
        $display("%d", float_out);
        float_a = 16'd16196;
        float_b = 16'd16225;
        gt = 16'd16338;

        #10
        $display("%d", float_out);
        float_a = 16'd16110;
        float_b = 16'd16138;
        gt = 16'd16256;

        #10
        $display("%d", float_out);
        float_a = 16'd15876;
        float_b = 16'd16244;
        gt = 16'd16266;

        #10
        $display("%d", float_out);
        float_a = 16'd16191;
        float_b = 16'd15768;
        gt = 16'd16210;

        #10
        $display("%d", float_out);
        float_a = 16'd16072;
        float_b = 16'd16161;
        gt = 16'd16258;

        #10
        $display("%d", float_out);
        float_a = 16'd15988;
        float_b = 16'd16168;
        gt = 16'd16229;

        #10
        $display("%d", float_out);
        float_a = 16'd15896;
        float_b = 16'd15616;
        gt = 16'd15928;

        #10
        $display("%d", float_out);
        float_a = 16'd15916;
        float_b = 16'd16203;
        gt = 16'd16246;

        #10
        $display("%d", float_out);
        float_a = 16'd15900;
        float_b = 16'd16221;
        gt = 16'd16258;

        #10
        $display("%d", float_out);
        float_a = 16'd15920;
        float_b = 16'd15424;
        gt = 16'd15932;

        #10
        $display("%d", float_out);
        float_a = 16'd16249;
        float_b = 16'd16210;
        gt = 16'd16358;

        #10
        $display("%d", float_out);
        float_a = 16'd15960;
        float_b = 16'd15824;
        gt = 16'd16032;

        #10
        $display("%d", float_out);
        float_a = 16'd16223;
        float_b = 16'd16225;
        gt = 16'd16352;

        #10
        $display("%d", float_out);
        float_a = 16'd16211;
        float_b = 16'd16038;
        gt = 16'd16275;

        #10
        $display("%d", float_out);
        float_a = 16'd15904;
        float_b = 16'd16204;
        gt = 16'd16244;

        #10
        $display("%d", float_out);
        float_a = 16'd16046;
        float_b = 16'd16064;
        gt = 16'd16183;

        #10
        $display("%d", float_out);
        float_a = 16'd16154;
        float_b = 16'd16241;
        gt = 16'd16326;

        #10
        $display("%d", float_out);
        float_a = 16'd16104;
        float_b = 16'd16174;
        gt = 16'd16273;

        #10
        $display("%d", float_out);
        float_a = 16'd16006;
        float_b = 16'd15896;
        gt = 16'd16082;

        #10
        $display("%d", float_out);
        float_a = 16'd16126;
        float_b = 16'd15792;
        gt = 16'd16149;

        #10
        $display("%d", float_out);
        float_a = 16'd15664;
        float_b = 16'd16064;
        gt = 16'd16086;

        #10
        $display("%d", float_out);
        float_a = 16'd16214;
        float_b = 16'd16131;
        gt = 16'd16300;

        #10
        $display("%d", float_out);
        float_a = 16'd16074;
        float_b = 16'd16076;
        gt = 16'd16203;

        #10
        $display("%d", float_out);
        float_a = 16'd16096;
        float_b = 16'd16245;
        gt = 16'd16306;

        #10
        $display("%d", float_out);
        float_a = 16'd16153;
        float_b = 16'd16038;
        gt = 16'd16236;

        #10
        $display("%d", float_out);
        float_a = 16'd16200;
        float_b = 16'd16203;
        gt = 16'd16330;

        #10
        $display("%d", float_out);
        float_a = 16'd15800;
        float_b = 16'd16221;
        gt = 16'd16244;

        #10
        $display("%d", float_out);
        float_a = 16'd16046;
        float_b = 16'd16219;
        gt = 16'd16281;

        #10
        $display("%d", float_out);
        float_a = 16'd16102;
        float_b = 16'd15872;
        gt = 16'd16147;

        #10
        $display("%d", float_out);
        float_a = 16'd16202;
        float_b = 16'd16173;
        gt = 16'd16316;

        #10
        $display("%d", float_out);
        float_a = 16'd16223;
        float_b = 16'd16222;
        gt = 16'd16350;

        #10
        $display("%d", float_out);
        float_a = 16'd15948;
        float_b = 16'd16056;
        gt = 16'd16143;

        #10
        $display("%d", float_out);
        float_a = 16'd16180;
        float_b = 16'd16146;
        gt = 16'd16291;

        #10
        $display("%d", float_out);
        float_a = 16'd16251;
        float_b = 16'd16168;
        gt = 16'd16338;

        #10
        $display("%d", float_out);
        float_a = 16'd16128;
        float_b = 16'd16225;
        gt = 16'd16304;

        #10
        $display("%d", float_out);
        float_a = 16'd15864;
        float_b = 16'd16068;
        gt = 16'd16129;

        #10
        $display("%d", float_out);
        float_a = 16'd16137;
        float_b = 16'd16210;
        gt = 16'd16302;

        #10
        $display("%d", float_out);
        float_a = 16'd16038;
        float_b = 16'd15520;
        gt = 16'd16048;

        #10
        $display("%d", float_out);
        float_a = 16'd15816;
        float_b = 16'd15976;
        gt = 16'd16038;

        #10
        $display("%d", float_out);
        float_a = 16'd15888;
        float_b = 16'd15752;
        gt = 16'd15956;

        #10
        $display("%d", float_out);
        float_a = 16'd16174;
        float_b = 16'd16014;
        gt = 16'd16245;

        #10
        $display("%d", float_out);
        float_a = 16'd16154;
        float_b = 16'd16237;
        gt = 16'd16324;

        #10
        $display("%d", float_out);
        float_a = 16'd16229;
        float_b = 16'd16193;
        gt = 16'd16339;

        #10
        $display("%d", float_out);
        float_a = 16'd16155;
        float_b = 16'd15908;
        gt = 16'd16196;

        #10
        $display("%d", float_out);
        float_a = 16'd16044;
        float_b = 16'd16248;
        gt = 16'd16295;

        #10
        $display("%d", float_out);
        float_a = 16'd16034;
        float_b = 16'd16189;
        gt = 16'd16263;

        #10
        $display("%d", float_out);
        float_a = 16'd15824;
        float_b = 16'd16148;
        gt = 16'd16174;

        #10
        $display("%d", float_out);
        float_a = 16'd16072;
        float_b = 16'd15944;
        gt = 16'd16150;

        #10
        $display("%d", float_out);
        float_a = 16'd15944;
        float_b = 16'd16224;
        gt = 16'd16265;

        #10
        $display("%d", float_out);
        float_a = 16'd16186;
        float_b = 16'd16234;
        gt = 16'd16338;

        #10
        $display("%d", float_out);
        float_a = 16'd16088;
        float_b = 16'd15680;
        gt = 16'd16112;

        #10
        $display("%d", float_out);
        float_a = 16'd15232;
        float_b = 16'd16222;
        gt = 16'd16223;

        #10
        $display("%d", float_out);
        float_a = 16'd15888;
        float_b = 16'd16170;
        gt = 16'd16206;

        #10
        $display("%d", float_out);
        float_a = 16'd15988;
        float_b = 16'd15904;
        gt = 16'd16074;

        #10
        $display("%d", float_out);
        float_a = 16'd16076;
        float_b = 16'd16044;
        gt = 16'd16188;

        #10
        $display("%d", float_out);
        float_a = 16'd16205;
        float_b = 16'd16142;
        gt = 16'd16302;

        #10
        $display("%d", float_out);
        float_a = 16'd15712;
        float_b = 16'd16022;
        gt = 16'd16050;

        #10
        $display("%d", float_out);
        float_a = 16'd16194;
        float_b = 16'd16102;
        gt = 16'd16282;

        #10
        $display("%d", float_out);
        float_a = 16'd15980;
        float_b = 16'd16134;
        gt = 16'd16193;

        #10
        $display("%d", float_out);
        float_a = 16'd16034;
        float_b = 16'd16188;
        gt = 16'd16262;

        #10
        $display("%d", float_out);
        float_a = 16'd15840;
        float_b = 16'd15940;
        gt = 16'd16026;

        #10
        $display("%d", float_out);
        float_a = 16'd15864;
        float_b = 16'd16243;
        gt = 16'd16265;

        #10
        $display("%d", float_out);
        $finish;
    end
endmodule
