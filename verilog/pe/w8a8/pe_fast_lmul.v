// Generated automatically via PyRTL
// As one initial test of synthesis, map to FPGA with:
//   yosys -p "synth_xilinx -top toplevel" thisfile.v

module toplevel(clk, rst, accum_in, adder_en, data_en, data_in, mul_en, weight_en, weight_in, out48385, out48386, out48387);
    input clk;
    input rst;
    input[7:0] accum_in;
    input adder_en;
    input data_en;
    input[7:0] data_in;
    input mul_en;
    input weight_en;
    input[7:0] weight_in;
    output[7:0] out48385;
    output[7:0] out48386;
    output[7:0] out48387;

    reg[7:0] tmp48385;
    reg[7:0] tmp48386;
    reg[7:0] tmp48387;
    reg[7:0] tmp48472;
    reg[4:0] tmp48665;
    reg tmp48666;

    wire[6:0] const_6807_73;
    wire const_6808_0;
    wire const_6809_0;
    wire const_6810_0;
    wire[6:0] const_6811_127;
    wire[6:0] const_6812_127;
    wire[6:0] const_6813_0;
    wire const_6814_1;
    wire const_6815_1;
    wire const_6816_1;
    wire const_6817_0;
    wire const_6818_0;
    wire const_6819_0;
    wire const_6820_0;
    wire const_6821_1;
    wire const_6822_0;
    wire const_6823_0;
    wire[3:0] const_6824_8;
    wire[3:0] const_6825_8;
    wire[3:0] const_6826_0;
    wire const_6827_0;
    wire const_6828_0;
    wire const_6829_0;
    wire const_6830_1;
    wire const_6831_0;
    wire const_6832_0;
    wire const_6833_0;
    wire const_6834_0;
    wire const_6835_0;
    wire const_6836_1;
    wire const_6837_0;
    wire const_6838_0;
    wire const_6839_0;
    wire const_6840_0;
    wire[1:0] const_6841_2;
    wire const_6842_1;
    wire const_6843_0;
    wire[1:0] const_6844_1;
    wire[1:0] const_6845_0;
    wire const_6846_0;
    wire const_6847_0;
    wire const_6848_0;
    wire const_6849_0;
    wire[1:0] const_6850_2;
    wire const_6851_1;
    wire const_6852_0;
    wire[1:0] const_6853_1;
    wire[1:0] const_6854_0;
    wire const_6855_0;
    wire const_6856_0;
    wire[2:0] const_6857_4;
    wire[1:0] const_6858_1;
    wire const_6859_0;
    wire const_6860_0;
    wire const_6861_0;
    wire const_6862_0;
    wire const_6863_1;
    wire const_6864_0;
    wire const_6865_0;
    wire const_6866_0;
    wire const_6867_0;
    wire const_6868_0;
    wire const_6869_1;
    wire const_6870_0;
    wire const_6871_0;
    wire const_6872_0;
    wire const_6873_0;
    wire const_6874_0;
    wire[7:0] tmp48379;
    wire[7:0] tmp48380;
    wire[7:0] tmp48381;
    wire tmp48382;
    wire tmp48383;
    wire tmp48384;
    wire tmp48388;
    wire tmp48389;
    wire[6:0] tmp48390;
    wire[6:0] tmp48391;
    wire[7:0] tmp48392;
    wire tmp48393;
    wire[6:0] tmp48394;
    wire[6:0] tmp48395;
    wire[6:0] tmp48396;
    wire[6:0] tmp48397;
    wire[6:0] tmp48398;
    wire[6:0] tmp48399;
    wire[6:0] tmp48400;
    wire[5:0] tmp48401;
    wire tmp48402;
    wire[6:0] tmp48403;
    wire[6:0] tmp48404;
    wire tmp48405;
    wire tmp48406;
    wire tmp48407;
    wire tmp48408;
    wire tmp48409;
    wire tmp48410;
    wire tmp48411;
    wire[6:0] tmp48412;
    wire tmp48413;
    wire tmp48414;
    wire tmp48415;
    wire tmp48416;
    wire tmp48417;
    wire tmp48418;
    wire tmp48419;
    wire tmp48420;
    wire tmp48421;
    wire tmp48422;
    wire tmp48423;
    wire tmp48424;
    wire tmp48425;
    wire tmp48426;
    wire tmp48427;
    wire tmp48428;
    wire tmp48429;
    wire tmp48430;
    wire tmp48431;
    wire tmp48432;
    wire tmp48433;
    wire tmp48434;
    wire tmp48435;
    wire tmp48436;
    wire tmp48437;
    wire tmp48438;
    wire tmp48439;
    wire tmp48440;
    wire tmp48441;
    wire tmp48442;
    wire tmp48443;
    wire tmp48444;
    wire tmp48445;
    wire tmp48446;
    wire tmp48447;
    wire tmp48448;
    wire tmp48449;
    wire tmp48450;
    wire tmp48451;
    wire tmp48452;
    wire tmp48453;
    wire tmp48454;
    wire tmp48455;
    wire[7:0] tmp48456;
    wire tmp48457;
    wire[7:0] tmp48458;
    wire[7:0] tmp48459;
    wire tmp48460;
    wire[8:0] tmp48461;
    wire[1:0] tmp48462;
    wire[6:0] tmp48463;
    wire tmp48464;
    wire tmp48465;
    wire[6:0] tmp48466;
    wire tmp48467;
    wire[6:0] tmp48468;
    wire[6:0] tmp48469;
    wire[7:0] tmp48470;
    wire tmp48471;
    wire[7:0] tmp48473;
    wire tmp48474;
    wire tmp48475;
    wire tmp48476;
    wire tmp48477;
    wire[3:0] tmp48478;
    wire[3:0] tmp48479;
    wire[3:0] tmp48480;
    wire[3:0] tmp48481;
    wire[3:0] tmp48482;
    wire[3:0] tmp48483;
    wire[2:0] tmp48484;
    wire[3:0] tmp48485;
    wire[2:0] tmp48486;
    wire[3:0] tmp48487;
    wire tmp48488;
    wire[4:0] tmp48489;
    wire[4:0] tmp48490;
    wire[3:0] tmp48491;
    wire[3:0] tmp48492;
    wire[3:0] tmp48493;
    wire tmp48494;
    wire[3:0] tmp48495;
    wire[2:0] tmp48496;
    wire[3:0] tmp48497;
    wire[3:0] tmp48498;
    wire[3:0] tmp48499;
    wire[3:0] tmp48500;
    wire[3:0] tmp48501;
    wire[3:0] tmp48502;
    wire[3:0] tmp48503;
    wire[3:0] tmp48504;
    wire[2:0] tmp48505;
    wire tmp48506;
    wire[3:0] tmp48507;
    wire[3:0] tmp48508;
    wire tmp48509;
    wire tmp48510;
    wire tmp48511;
    wire tmp48512;
    wire[3:0] tmp48513;
    wire tmp48514;
    wire tmp48515;
    wire tmp48516;
    wire tmp48517;
    wire tmp48518;
    wire tmp48519;
    wire tmp48520;
    wire tmp48521;
    wire tmp48522;
    wire tmp48523;
    wire tmp48524;
    wire tmp48525;
    wire tmp48526;
    wire tmp48527;
    wire tmp48528;
    wire tmp48529;
    wire[4:0] tmp48530;
    wire tmp48531;
    wire[4:0] tmp48532;
    wire[4:0] tmp48533;
    wire tmp48534;
    wire[5:0] tmp48535;
    wire[4:0] tmp48536;
    wire tmp48537;
    wire tmp48538;
    wire tmp48539;
    wire[3:0] tmp48540;
    wire[3:0] tmp48541;
    wire tmp48542;
    wire[3:0] tmp48543;
    wire[3:0] tmp48544;
    wire[2:0] tmp48545;
    wire[3:0] tmp48546;
    wire[4:0] tmp48547;
    wire[3:0] tmp48548;
    wire[4:0] tmp48549;
    wire tmp48550;
    wire[4:0] tmp48551;
    wire[4:0] tmp48552;
    wire tmp48553;
    wire[3:0] tmp48554;
    wire tmp48555;
    wire[3:0] tmp48556;
    wire[3:0] tmp48557;
    wire[3:0] tmp48558;
    wire[3:0] tmp48559;
    wire tmp48560;
    wire[3:0] tmp48561;
    wire[7:0] tmp48562;
    wire[7:0] tmp48563;
    wire[6:0] tmp48564;
    wire[7:0] tmp48565;
    wire[6:0] tmp48566;
    wire[7:0] tmp48567;
    wire[7:0] tmp48568;
    wire tmp48569;
    wire[7:0] tmp48570;
    wire[1:0] tmp48571;
    wire[1:0] tmp48572;
    wire[5:0] tmp48573;
    wire[7:0] tmp48574;
    wire[5:0] tmp48575;
    wire[7:0] tmp48576;
    wire[7:0] tmp48577;
    wire tmp48578;
    wire[7:0] tmp48579;
    wire[3:0] tmp48580;
    wire[3:0] tmp48581;
    wire[3:0] tmp48582;
    wire[7:0] tmp48583;
    wire[3:0] tmp48584;
    wire[7:0] tmp48585;
    wire[7:0] tmp48586;
    wire tmp48587;
    wire[7:0] tmp48588;
    wire[7:0] tmp48589;
    wire[7:0] tmp48590;
    wire tmp48591;
    wire[7:0] tmp48592;
    wire[3:0] tmp48593;
    wire[3:0] tmp48594;
    wire tmp48595;
    wire tmp48596;
    wire tmp48597;
    wire tmp48598;
    wire tmp48599;
    wire[1:0] tmp48600;
    wire tmp48601;
    wire tmp48602;
    wire tmp48603;
    wire tmp48604;
    wire tmp48605;
    wire[5:0] tmp48606;
    wire tmp48607;
    wire[4:0] tmp48608;
    wire[4:0] tmp48609;
    wire[3:0] tmp48610;
    wire[2:0] tmp48611;
    wire[3:0] tmp48612;
    wire[4:0] tmp48613;
    wire tmp48614;
    wire[3:0] tmp48615;
    wire[4:0] tmp48616;
    wire[4:0] tmp48617;
    wire[4:0] tmp48618;
    wire tmp48619;
    wire tmp48620;
    wire tmp48621;
    wire tmp48622;
    wire tmp48623;
    wire[4:0] tmp48624;
    wire tmp48625;
    wire tmp48626;
    wire tmp48627;
    wire tmp48628;
    wire tmp48629;
    wire tmp48630;
    wire tmp48631;
    wire tmp48632;
    wire tmp48633;
    wire tmp48634;
    wire tmp48635;
    wire tmp48636;
    wire tmp48637;
    wire tmp48638;
    wire tmp48639;
    wire tmp48640;
    wire tmp48641;
    wire tmp48642;
    wire tmp48643;
    wire tmp48644;
    wire tmp48645;
    wire tmp48646;
    wire tmp48647;
    wire tmp48648;
    wire tmp48649;
    wire[5:0] tmp48650;
    wire tmp48651;
    wire[5:0] tmp48652;
    wire[5:0] tmp48653;
    wire tmp48654;
    wire tmp48655;
    wire[4:0] tmp48656;
    wire[5:0] tmp48657;
    wire[4:0] tmp48658;
    wire[5:0] tmp48659;
    wire[6:0] tmp48660;
    wire tmp48661;
    wire[6:0] tmp48662;
    wire[6:0] tmp48663;
    wire[4:0] tmp48664;
    wire tmp48667;
    wire[3:0] tmp48668;
    wire[3:0] tmp48669;
    wire[1:0] tmp48670;
    wire[1:0] tmp48671;
    wire[1:0] tmp48672;
    wire tmp48673;
    wire[1:0] tmp48674;
    wire tmp48675;
    wire tmp48676;
    wire[1:0] tmp48677;
    wire tmp48678;
    wire tmp48679;
    wire tmp48680;
    wire tmp48681;
    wire tmp48682;
    wire tmp48683;
    wire tmp48684;
    wire[1:0] tmp48685;
    wire[1:0] tmp48686;
    wire[1:0] tmp48687;
    wire[1:0] tmp48688;
    wire[1:0] tmp48689;
    wire tmp48690;
    wire[1:0] tmp48691;
    wire tmp48692;
    wire tmp48693;
    wire[1:0] tmp48694;
    wire tmp48695;
    wire tmp48696;
    wire tmp48697;
    wire tmp48698;
    wire tmp48699;
    wire tmp48700;
    wire tmp48701;
    wire[1:0] tmp48702;
    wire[1:0] tmp48703;
    wire[1:0] tmp48704;
    wire[1:0] tmp48705;
    wire[3:0] tmp48706;
    wire[2:0] tmp48707;
    wire tmp48708;
    wire tmp48709;
    wire tmp48710;
    wire tmp48711;
    wire[2:0] tmp48712;
    wire tmp48713;
    wire tmp48714;
    wire[2:0] tmp48715;
    wire tmp48716;
    wire tmp48717;
    wire tmp48718;
    wire[1:0] tmp48719;
    wire[2:0] tmp48720;
    wire[2:0] tmp48721;
    wire[2:0] tmp48722;
    wire[2:0] tmp48723;
    wire tmp48724;
    wire[3:0] tmp48725;
    wire[2:0] tmp48726;
    wire[3:0] tmp48727;
    wire[4:0] tmp48728;
    wire[3:0] tmp48729;
    wire[4:0] tmp48730;
    wire[4:0] tmp48731;
    wire[3:0] tmp48732;
    wire[4:0] tmp48733;
    wire tmp48734;
    wire[4:0] tmp48735;
    wire[3:0] tmp48736;
    wire[4:0] tmp48737;
    wire[3:0] tmp48738;
    wire[4:0] tmp48739;
    wire[4:0] tmp48740;
    wire tmp48741;
    wire[4:0] tmp48742;
    wire[1:0] tmp48743;
    wire[1:0] tmp48744;
    wire[2:0] tmp48745;
    wire[4:0] tmp48746;
    wire[2:0] tmp48747;
    wire[4:0] tmp48748;
    wire[4:0] tmp48749;
    wire tmp48750;
    wire[4:0] tmp48751;
    wire[3:0] tmp48752;
    wire[3:0] tmp48753;
    wire tmp48754;
    wire[4:0] tmp48755;
    wire tmp48756;
    wire[4:0] tmp48757;
    wire[4:0] tmp48758;
    wire tmp48759;
    wire[4:0] tmp48760;
    wire[7:0] tmp48761;
    wire[4:0] tmp48762;
    wire tmp48763;
    wire[4:0] tmp48764;
    wire tmp48765;
    wire[4:0] tmp48766;
    wire tmp48767;
    wire tmp48768;
    wire tmp48769;
    wire tmp48770;
    wire tmp48771;
    wire tmp48772;
    wire tmp48773;
    wire tmp48774;
    wire tmp48775;
    wire tmp48776;
    wire[4:0] tmp48777;
    wire[3:0] tmp48778;
    wire[4:0] tmp48779;
    wire[5:0] tmp48780;
    wire[4:0] tmp48781;
    wire tmp48782;
    wire tmp48783;
    wire[2:0] tmp48784;
    wire[2:0] tmp48785;
    wire[2:0] tmp48786;
    wire[2:0] tmp48787;
    wire[3:0] tmp48788;
    wire[4:0] tmp48789;
    wire[3:0] tmp48790;
    wire[3:0] tmp48791;
    wire[2:0] tmp48792;
    wire[3:0] tmp48793;
    wire[4:0] tmp48794;
    wire[3:0] tmp48795;
    wire tmp48796;
    wire tmp48797;
    wire tmp48798;
    wire[3:0] tmp48799;
    wire[2:0] tmp48800;
    wire[3:0] tmp48801;
    wire tmp48802;
    wire tmp48803;
    wire tmp48804;
    wire tmp48805;
    wire tmp48806;
    wire tmp48807;
    wire tmp48808;
    wire tmp48809;
    wire tmp48810;
    wire tmp48811;
    wire tmp48812;
    wire tmp48813;
    wire tmp48814;
    wire tmp48815;
    wire tmp48816;
    wire tmp48817;
    wire tmp48818;
    wire tmp48819;
    wire[7:0] tmp48820;
    wire[7:0] tmp48821;
    wire[7:0] tmp48822;
    wire[7:0] tmp48823;
    wire[7:0] tmp48824;

    // Combinational
    assign const_6807_73 = 73;
    assign const_6808_0 = 0;
    assign const_6809_0 = 0;
    assign const_6810_0 = 0;
    assign const_6811_127 = 127;
    assign const_6812_127 = 127;
    assign const_6813_0 = 0;
    assign const_6814_1 = 1;
    assign const_6815_1 = 1;
    assign const_6816_1 = 1;
    assign const_6817_0 = 0;
    assign const_6818_0 = 0;
    assign const_6819_0 = 0;
    assign const_6820_0 = 0;
    assign const_6821_1 = 1;
    assign const_6822_0 = 0;
    assign const_6823_0 = 0;
    assign const_6824_8 = 8;
    assign const_6825_8 = 8;
    assign const_6826_0 = 0;
    assign const_6827_0 = 0;
    assign const_6828_0 = 0;
    assign const_6829_0 = 0;
    assign const_6830_1 = 1;
    assign const_6831_0 = 0;
    assign const_6832_0 = 0;
    assign const_6833_0 = 0;
    assign const_6834_0 = 0;
    assign const_6835_0 = 0;
    assign const_6836_1 = 1;
    assign const_6837_0 = 0;
    assign const_6838_0 = 0;
    assign const_6839_0 = 0;
    assign const_6840_0 = 0;
    assign const_6841_2 = 2;
    assign const_6842_1 = 1;
    assign const_6843_0 = 0;
    assign const_6844_1 = 1;
    assign const_6845_0 = 0;
    assign const_6846_0 = 0;
    assign const_6847_0 = 0;
    assign const_6848_0 = 0;
    assign const_6849_0 = 0;
    assign const_6850_2 = 2;
    assign const_6851_1 = 1;
    assign const_6852_0 = 0;
    assign const_6853_1 = 1;
    assign const_6854_0 = 0;
    assign const_6855_0 = 0;
    assign const_6856_0 = 0;
    assign const_6857_4 = 4;
    assign const_6858_1 = 1;
    assign const_6859_0 = 0;
    assign const_6860_0 = 0;
    assign const_6861_0 = 0;
    assign const_6862_0 = 0;
    assign const_6863_1 = 1;
    assign const_6864_0 = 0;
    assign const_6865_0 = 0;
    assign const_6866_0 = 0;
    assign const_6867_0 = 0;
    assign const_6868_0 = 0;
    assign const_6869_1 = 1;
    assign const_6870_0 = 0;
    assign const_6871_0 = 0;
    assign const_6872_0 = 0;
    assign const_6873_0 = 0;
    assign const_6874_0 = 0;
    assign out48385 = tmp48385;
    assign out48386 = tmp48386;
    assign out48387 = tmp48387;
    assign tmp48379 = data_in;
    assign tmp48380 = weight_in;
    assign tmp48381 = accum_in;
    assign tmp48382 = weight_en;
    assign tmp48383 = data_en;
    assign tmp48384 = adder_en;
    assign tmp48388 = {tmp48385[7]};
    assign tmp48389 = {tmp48386[7]};
    assign tmp48390 = {tmp48385[6], tmp48385[5], tmp48385[4], tmp48385[3], tmp48385[2], tmp48385[1], tmp48385[0]};
    assign tmp48391 = {tmp48386[6], tmp48386[5], tmp48386[4], tmp48386[3], tmp48386[2], tmp48386[1], tmp48386[0]};
    assign tmp48392 = tmp48470;
    assign tmp48393 = tmp48388 ^ tmp48389;
    assign tmp48394 = tmp48390 ^ tmp48391;
    assign tmp48395 = tmp48394 ^ const_6807_73;
    assign tmp48396 = tmp48390 | tmp48391;
    assign tmp48397 = tmp48390 | const_6807_73;
    assign tmp48398 = tmp48396 & tmp48397;
    assign tmp48399 = tmp48391 | const_6807_73;
    assign tmp48400 = tmp48398 & tmp48399;
    assign tmp48401 = {tmp48395[6], tmp48395[5], tmp48395[4], tmp48395[3], tmp48395[2], tmp48395[1]};
    assign tmp48402 = {const_6808_0};
    assign tmp48403 = {tmp48402, tmp48401};
    assign tmp48404 = tmp48403 ^ tmp48400;
    assign tmp48405 = {tmp48404[0]};
    assign tmp48406 = {tmp48404[1]};
    assign tmp48407 = {tmp48404[2]};
    assign tmp48408 = {tmp48404[3]};
    assign tmp48409 = {tmp48404[4]};
    assign tmp48410 = {tmp48404[5]};
    assign tmp48411 = {tmp48404[6]};
    assign tmp48412 = tmp48403 & tmp48400;
    assign tmp48413 = {tmp48412[0]};
    assign tmp48414 = {tmp48412[1]};
    assign tmp48415 = {tmp48412[2]};
    assign tmp48416 = {tmp48412[3]};
    assign tmp48417 = {tmp48412[4]};
    assign tmp48418 = {tmp48412[5]};
    assign tmp48419 = {tmp48412[6]};
    assign tmp48420 = tmp48411 & tmp48418;
    assign tmp48421 = tmp48419 | tmp48420;
    assign tmp48422 = tmp48411 & tmp48410;
    assign tmp48423 = tmp48410 & tmp48417;
    assign tmp48424 = tmp48418 | tmp48423;
    assign tmp48425 = tmp48410 & tmp48409;
    assign tmp48426 = tmp48409 & tmp48416;
    assign tmp48427 = tmp48417 | tmp48426;
    assign tmp48428 = tmp48409 & tmp48408;
    assign tmp48429 = tmp48408 & tmp48415;
    assign tmp48430 = tmp48416 | tmp48429;
    assign tmp48431 = tmp48408 & tmp48407;
    assign tmp48432 = tmp48407 & tmp48414;
    assign tmp48433 = tmp48415 | tmp48432;
    assign tmp48434 = tmp48407 & tmp48406;
    assign tmp48435 = tmp48406 & tmp48413;
    assign tmp48436 = tmp48414 | tmp48435;
    assign tmp48437 = tmp48422 & tmp48427;
    assign tmp48438 = tmp48421 | tmp48437;
    assign tmp48439 = tmp48422 & tmp48428;
    assign tmp48440 = tmp48425 & tmp48430;
    assign tmp48441 = tmp48424 | tmp48440;
    assign tmp48442 = tmp48425 & tmp48431;
    assign tmp48443 = tmp48428 & tmp48433;
    assign tmp48444 = tmp48427 | tmp48443;
    assign tmp48445 = tmp48428 & tmp48434;
    assign tmp48446 = tmp48431 & tmp48436;
    assign tmp48447 = tmp48430 | tmp48446;
    assign tmp48448 = tmp48434 & tmp48413;
    assign tmp48449 = tmp48433 | tmp48448;
    assign tmp48450 = tmp48439 & tmp48449;
    assign tmp48451 = tmp48438 | tmp48450;
    assign tmp48452 = tmp48442 & tmp48436;
    assign tmp48453 = tmp48441 | tmp48452;
    assign tmp48454 = tmp48445 & tmp48413;
    assign tmp48455 = tmp48444 | tmp48454;
    assign tmp48456 = {tmp48451, tmp48453, tmp48455, tmp48447, tmp48449, tmp48436, tmp48413, const_6809_0};
    assign tmp48457 = {const_6810_0};
    assign tmp48458 = {tmp48457, tmp48404};
    assign tmp48459 = tmp48456 ^ tmp48458;
    assign tmp48460 = {tmp48395[0]};
    assign tmp48461 = {tmp48459, tmp48460};
    assign tmp48462 = {tmp48461[8], tmp48461[7]};
    assign tmp48463 = {tmp48461[6], tmp48461[5], tmp48461[4], tmp48461[3], tmp48461[2], tmp48461[1], tmp48461[0]};
    assign tmp48464 = {tmp48462[1]};
    assign tmp48465 = {tmp48462[0]};
    assign tmp48466 = tmp48465 ? tmp48463 : const_6813_0;
    assign tmp48467 = {tmp48462[0]};
    assign tmp48468 = tmp48467 ? const_6812_127 : const_6812_127;
    assign tmp48469 = tmp48464 ? tmp48468 : tmp48466;
    assign tmp48470 = {tmp48393, tmp48469};
    assign tmp48471 = mul_en;
    assign tmp48473 = tmp48471 ? tmp48392 : tmp48472;
    assign tmp48474 = tmp48476;
    assign tmp48475 = tmp48477;
    assign tmp48476 = {tmp48472[7]};
    assign tmp48477 = {tmp48381[7]};
    assign tmp48478 = tmp48480;
    assign tmp48479 = tmp48481;
    assign tmp48480 = {tmp48472[6], tmp48472[5], tmp48472[4], tmp48472[3]};
    assign tmp48481 = {tmp48381[6], tmp48381[5], tmp48381[4], tmp48381[3]};
    assign tmp48482 = tmp48485;
    assign tmp48483 = tmp48487;
    assign tmp48484 = {tmp48472[2], tmp48472[1], tmp48472[0]};
    assign tmp48485 = {const_6814_1, tmp48484};
    assign tmp48486 = {tmp48381[2], tmp48381[1], tmp48381[0]};
    assign tmp48487 = {const_6815_1, tmp48486};
    assign tmp48488 = tmp48494;
    assign tmp48489 = tmp48536;
    assign tmp48490 = tmp48552;
    assign tmp48491 = tmp48540;
    assign tmp48492 = tmp48554;
    assign tmp48493 = tmp48556;
    assign tmp48494 = tmp48474 ^ tmp48475;
    assign tmp48495 = ~tmp48479;
    assign tmp48496 = {const_6817_0, const_6817_0, const_6817_0};
    assign tmp48497 = {tmp48496, const_6816_1};
    assign tmp48498 = tmp48478 ^ tmp48495;
    assign tmp48499 = tmp48498 ^ tmp48497;
    assign tmp48500 = tmp48478 | tmp48495;
    assign tmp48501 = tmp48478 | tmp48497;
    assign tmp48502 = tmp48500 & tmp48501;
    assign tmp48503 = tmp48495 | tmp48497;
    assign tmp48504 = tmp48502 & tmp48503;
    assign tmp48505 = {tmp48499[3], tmp48499[2], tmp48499[1]};
    assign tmp48506 = {const_6818_0};
    assign tmp48507 = {tmp48506, tmp48505};
    assign tmp48508 = tmp48507 ^ tmp48504;
    assign tmp48509 = {tmp48508[0]};
    assign tmp48510 = {tmp48508[1]};
    assign tmp48511 = {tmp48508[2]};
    assign tmp48512 = {tmp48508[3]};
    assign tmp48513 = tmp48507 & tmp48504;
    assign tmp48514 = {tmp48513[0]};
    assign tmp48515 = {tmp48513[1]};
    assign tmp48516 = {tmp48513[2]};
    assign tmp48517 = {tmp48513[3]};
    assign tmp48518 = tmp48512 & tmp48516;
    assign tmp48519 = tmp48517 | tmp48518;
    assign tmp48520 = tmp48512 & tmp48511;
    assign tmp48521 = tmp48511 & tmp48515;
    assign tmp48522 = tmp48516 | tmp48521;
    assign tmp48523 = tmp48511 & tmp48510;
    assign tmp48524 = tmp48510 & tmp48514;
    assign tmp48525 = tmp48515 | tmp48524;
    assign tmp48526 = tmp48520 & tmp48525;
    assign tmp48527 = tmp48519 | tmp48526;
    assign tmp48528 = tmp48523 & tmp48514;
    assign tmp48529 = tmp48522 | tmp48528;
    assign tmp48530 = {tmp48527, tmp48529, tmp48525, tmp48514, const_6819_0};
    assign tmp48531 = {const_6820_0};
    assign tmp48532 = {tmp48531, tmp48508};
    assign tmp48533 = tmp48530 ^ tmp48532;
    assign tmp48534 = {tmp48499[0]};
    assign tmp48535 = {tmp48533, tmp48534};
    assign tmp48536 = {tmp48535[4], tmp48535[3], tmp48535[2], tmp48535[1], tmp48535[0]};
    assign tmp48537 = {tmp48489[4]};
    assign tmp48538 = ~tmp48537;
    assign tmp48539 = {tmp48489[4]};
    assign tmp48540 = tmp48539 ? tmp48479 : tmp48478;
    assign tmp48541 = {tmp48489[3], tmp48489[2], tmp48489[1], tmp48489[0]};
    assign tmp48542 = {tmp48489[4]};
    assign tmp48543 = {tmp48489[3], tmp48489[2], tmp48489[1], tmp48489[0]};
    assign tmp48544 = ~tmp48543;
    assign tmp48545 = {const_6822_0, const_6822_0, const_6822_0};
    assign tmp48546 = {tmp48545, const_6821_1};
    assign tmp48547 = tmp48544 + tmp48546;
    assign tmp48548 = {tmp48547[3], tmp48547[2], tmp48547[1], tmp48547[0]};
    assign tmp48549 = {tmp48542, tmp48548};
    assign tmp48550 = {const_6823_0};
    assign tmp48551 = {tmp48550, tmp48541};
    assign tmp48552 = tmp48538 ? tmp48549 : tmp48551;
    assign tmp48553 = {tmp48489[4]};
    assign tmp48554 = tmp48553 ? tmp48482 : tmp48483;
    assign tmp48555 = {tmp48489[4]};
    assign tmp48556 = tmp48555 ? tmp48483 : tmp48482;
    assign tmp48557 = tmp48558;
    assign tmp48558 = {tmp48490[3], tmp48490[2], tmp48490[1], tmp48490[0]};
    assign tmp48559 = tmp48561;
    assign tmp48560 = tmp48557 > const_6824_8;
    assign tmp48561 = tmp48560 ? const_6825_8 : tmp48557;
    assign tmp48562 = {tmp48492, const_6826_0};
    assign tmp48563 = tmp48592;
    assign tmp48564 = {tmp48562[6], tmp48562[5], tmp48562[4], tmp48562[3], tmp48562[2], tmp48562[1], tmp48562[0]};
    assign tmp48565 = {tmp48564, const_6827_0};
    assign tmp48566 = {tmp48562[7], tmp48562[6], tmp48562[5], tmp48562[4], tmp48562[3], tmp48562[2], tmp48562[1]};
    assign tmp48567 = {const_6827_0, tmp48566};
    assign tmp48568 = const_6828_0 ? tmp48565 : tmp48567;
    assign tmp48569 = {tmp48559[0]};
    assign tmp48570 = tmp48569 ? tmp48568 : tmp48562;
    assign tmp48571 = {const_6827_0, const_6827_0};
    assign tmp48572 = {tmp48571[1], tmp48571[0]};
    assign tmp48573 = {tmp48570[5], tmp48570[4], tmp48570[3], tmp48570[2], tmp48570[1], tmp48570[0]};
    assign tmp48574 = {tmp48573, tmp48572};
    assign tmp48575 = {tmp48570[7], tmp48570[6], tmp48570[5], tmp48570[4], tmp48570[3], tmp48570[2]};
    assign tmp48576 = {tmp48572, tmp48575};
    assign tmp48577 = const_6828_0 ? tmp48574 : tmp48576;
    assign tmp48578 = {tmp48559[1]};
    assign tmp48579 = tmp48578 ? tmp48577 : tmp48570;
    assign tmp48580 = {tmp48572, tmp48572};
    assign tmp48581 = {tmp48580[3], tmp48580[2], tmp48580[1], tmp48580[0]};
    assign tmp48582 = {tmp48579[3], tmp48579[2], tmp48579[1], tmp48579[0]};
    assign tmp48583 = {tmp48582, tmp48581};
    assign tmp48584 = {tmp48579[7], tmp48579[6], tmp48579[5], tmp48579[4]};
    assign tmp48585 = {tmp48581, tmp48584};
    assign tmp48586 = const_6828_0 ? tmp48583 : tmp48585;
    assign tmp48587 = {tmp48559[2]};
    assign tmp48588 = tmp48587 ? tmp48586 : tmp48579;
    assign tmp48589 = {tmp48581, tmp48581};
    assign tmp48590 = {tmp48589[7], tmp48589[6], tmp48589[5], tmp48589[4], tmp48589[3], tmp48589[2], tmp48589[1], tmp48589[0]};
    assign tmp48591 = {tmp48559[3]};
    assign tmp48592 = tmp48591 ? tmp48590 : tmp48588;
    assign tmp48593 = {tmp48563[7], tmp48563[6], tmp48563[5], tmp48563[4]};
    assign tmp48594 = {tmp48563[3], tmp48563[2], tmp48563[1], tmp48563[0]};
    assign tmp48595 = tmp48598;
    assign tmp48596 = tmp48599;
    assign tmp48597 = tmp48605;
    assign tmp48598 = {tmp48594[3]};
    assign tmp48599 = {tmp48594[2]};
    assign tmp48600 = {tmp48594[1], tmp48594[0]};
    assign tmp48601 = {tmp48600[0]};
    assign tmp48602 = {tmp48601};
    assign tmp48603 = {tmp48600[1]};
    assign tmp48604 = {tmp48603};
    assign tmp48605 = tmp48602 | tmp48604;
    assign tmp48606 = tmp48653;
    assign tmp48607 = {const_6829_0};
    assign tmp48608 = {tmp48607, tmp48493};
    assign tmp48609 = tmp48617;
    assign tmp48610 = ~tmp48593;
    assign tmp48611 = {const_6831_0, const_6831_0, const_6831_0};
    assign tmp48612 = {tmp48611, const_6830_1};
    assign tmp48613 = tmp48610 + tmp48612;
    assign tmp48614 = {tmp48613[4]};
    assign tmp48615 = {const_6833_0, const_6833_0, const_6833_0, const_6833_0};
    assign tmp48616 = {tmp48615, const_6832_0};
    assign tmp48617 = tmp48488 ? tmp48613 : tmp48616;
    assign tmp48618 = tmp48608 ^ tmp48609;
    assign tmp48619 = {tmp48618[0]};
    assign tmp48620 = {tmp48618[1]};
    assign tmp48621 = {tmp48618[2]};
    assign tmp48622 = {tmp48618[3]};
    assign tmp48623 = {tmp48618[4]};
    assign tmp48624 = tmp48608 & tmp48609;
    assign tmp48625 = {tmp48624[0]};
    assign tmp48626 = {tmp48624[1]};
    assign tmp48627 = {tmp48624[2]};
    assign tmp48628 = {tmp48624[3]};
    assign tmp48629 = {tmp48624[4]};
    assign tmp48630 = tmp48623 & tmp48628;
    assign tmp48631 = tmp48629 | tmp48630;
    assign tmp48632 = tmp48623 & tmp48622;
    assign tmp48633 = tmp48622 & tmp48627;
    assign tmp48634 = tmp48628 | tmp48633;
    assign tmp48635 = tmp48622 & tmp48621;
    assign tmp48636 = tmp48621 & tmp48626;
    assign tmp48637 = tmp48627 | tmp48636;
    assign tmp48638 = tmp48621 & tmp48620;
    assign tmp48639 = tmp48620 & tmp48625;
    assign tmp48640 = tmp48626 | tmp48639;
    assign tmp48641 = tmp48632 & tmp48637;
    assign tmp48642 = tmp48631 | tmp48641;
    assign tmp48643 = tmp48632 & tmp48638;
    assign tmp48644 = tmp48635 & tmp48640;
    assign tmp48645 = tmp48634 | tmp48644;
    assign tmp48646 = tmp48638 & tmp48625;
    assign tmp48647 = tmp48637 | tmp48646;
    assign tmp48648 = tmp48643 & tmp48625;
    assign tmp48649 = tmp48642 | tmp48648;
    assign tmp48650 = {tmp48649, tmp48645, tmp48647, tmp48640, tmp48625, const_6834_0};
    assign tmp48651 = {const_6835_0};
    assign tmp48652 = {tmp48651, tmp48618};
    assign tmp48653 = tmp48650 ^ tmp48652;
    assign tmp48654 = tmp48655;
    assign tmp48655 = {tmp48606[5]};
    assign tmp48656 = tmp48664;
    assign tmp48657 = ~tmp48606;
    assign tmp48658 = {const_6837_0, const_6837_0, const_6837_0, const_6837_0, const_6837_0};
    assign tmp48659 = {tmp48658, const_6836_1};
    assign tmp48660 = tmp48657 + tmp48659;
    assign tmp48661 = {const_6838_0};
    assign tmp48662 = {tmp48661, tmp48606};
    assign tmp48663 = tmp48654 ? tmp48660 : tmp48662;
    assign tmp48664 = {tmp48663[4], tmp48663[3], tmp48663[2], tmp48663[1], tmp48663[0]};
    assign tmp48667 = {tmp48665[4]};
    assign tmp48668 = tmp48732;
    assign tmp48669 = {tmp48665[3], tmp48665[2], tmp48665[1], tmp48665[0]};
    assign tmp48670 = {tmp48669[3], tmp48669[2]};
    assign tmp48671 = {tmp48669[1], tmp48669[0]};
    assign tmp48672 = tmp48688;
    assign tmp48673 = {const_6840_0};
    assign tmp48674 = {tmp48673, const_6839_0};
    assign tmp48675 = tmp48670 == tmp48674;
    assign tmp48676 = {const_6843_0};
    assign tmp48677 = {tmp48676, const_6842_1};
    assign tmp48678 = tmp48670 == tmp48677;
    assign tmp48679 = ~tmp48675;
    assign tmp48680 = tmp48679 & tmp48678;
    assign tmp48681 = ~tmp48675;
    assign tmp48682 = ~tmp48678;
    assign tmp48683 = tmp48681 & tmp48682;
    assign tmp48684 = {const_6847_0};
    assign tmp48685 = {tmp48684, const_6846_0};
    assign tmp48686 = tmp48675 ? const_6841_2 : tmp48685;
    assign tmp48687 = tmp48680 ? const_6844_1 : tmp48686;
    assign tmp48688 = tmp48683 ? const_6845_0 : tmp48687;
    assign tmp48689 = tmp48705;
    assign tmp48690 = {const_6849_0};
    assign tmp48691 = {tmp48690, const_6848_0};
    assign tmp48692 = tmp48671 == tmp48691;
    assign tmp48693 = {const_6852_0};
    assign tmp48694 = {tmp48693, const_6851_1};
    assign tmp48695 = tmp48671 == tmp48694;
    assign tmp48696 = ~tmp48692;
    assign tmp48697 = tmp48696 & tmp48695;
    assign tmp48698 = ~tmp48692;
    assign tmp48699 = ~tmp48695;
    assign tmp48700 = tmp48698 & tmp48699;
    assign tmp48701 = {const_6856_0};
    assign tmp48702 = {tmp48701, const_6855_0};
    assign tmp48703 = tmp48692 ? const_6850_2 : tmp48702;
    assign tmp48704 = tmp48697 ? const_6853_1 : tmp48703;
    assign tmp48705 = tmp48700 ? const_6854_0 : tmp48704;
    assign tmp48706 = tmp48725;
    assign tmp48707 = tmp48723;
    assign tmp48708 = {tmp48672[1]};
    assign tmp48709 = {tmp48689[1]};
    assign tmp48710 = tmp48708 & tmp48709;
    assign tmp48711 = {tmp48689[0]};
    assign tmp48712 = {const_6858_1, tmp48711};
    assign tmp48713 = ~tmp48710;
    assign tmp48714 = tmp48713 & tmp48708;
    assign tmp48715 = {const_6859_0, tmp48672};
    assign tmp48716 = ~tmp48710;
    assign tmp48717 = ~tmp48708;
    assign tmp48718 = tmp48716 & tmp48717;
    assign tmp48719 = {const_6861_0, const_6861_0};
    assign tmp48720 = {tmp48719, const_6860_0};
    assign tmp48721 = tmp48710 ? const_6857_4 : tmp48720;
    assign tmp48722 = tmp48714 ? tmp48712 : tmp48721;
    assign tmp48723 = tmp48718 ? tmp48715 : tmp48722;
    assign tmp48724 = {const_6862_0};
    assign tmp48725 = {tmp48724, tmp48707};
    assign tmp48726 = {const_6864_0, const_6864_0, const_6864_0};
    assign tmp48727 = {tmp48726, const_6863_1};
    assign tmp48728 = tmp48706 + tmp48727;
    assign tmp48729 = {const_6866_0, const_6866_0, const_6866_0, const_6866_0};
    assign tmp48730 = {tmp48729, const_6865_0};
    assign tmp48731 = tmp48667 ? tmp48730 : tmp48728;
    assign tmp48732 = {tmp48731[3], tmp48731[2], tmp48731[1], tmp48731[0]};
    assign tmp48733 = tmp48766;
    assign tmp48734 = {const_6867_0};
    assign tmp48735 = {tmp48734, tmp48668};
    assign tmp48736 = {tmp48665[3], tmp48665[2], tmp48665[1], tmp48665[0]};
    assign tmp48737 = {tmp48736, const_6868_0};
    assign tmp48738 = {tmp48665[4], tmp48665[3], tmp48665[2], tmp48665[1]};
    assign tmp48739 = {const_6868_0, tmp48738};
    assign tmp48740 = const_6869_1 ? tmp48737 : tmp48739;
    assign tmp48741 = {tmp48735[0]};
    assign tmp48742 = tmp48741 ? tmp48740 : tmp48665;
    assign tmp48743 = {const_6868_0, const_6868_0};
    assign tmp48744 = {tmp48743[1], tmp48743[0]};
    assign tmp48745 = {tmp48742[2], tmp48742[1], tmp48742[0]};
    assign tmp48746 = {tmp48745, tmp48744};
    assign tmp48747 = {tmp48742[4], tmp48742[3], tmp48742[2]};
    assign tmp48748 = {tmp48744, tmp48747};
    assign tmp48749 = const_6869_1 ? tmp48746 : tmp48748;
    assign tmp48750 = {tmp48735[1]};
    assign tmp48751 = tmp48750 ? tmp48749 : tmp48742;
    assign tmp48752 = {tmp48744, tmp48744};
    assign tmp48753 = {tmp48752[3], tmp48752[2], tmp48752[1], tmp48752[0]};
    assign tmp48754 = {tmp48751[0]};
    assign tmp48755 = {tmp48754, tmp48753};
    assign tmp48756 = {tmp48751[4]};
    assign tmp48757 = {tmp48753, tmp48756};
    assign tmp48758 = const_6869_1 ? tmp48755 : tmp48757;
    assign tmp48759 = {tmp48735[2]};
    assign tmp48760 = tmp48759 ? tmp48758 : tmp48751;
    assign tmp48761 = {tmp48753, tmp48753};
    assign tmp48762 = {tmp48761[4], tmp48761[3], tmp48761[2], tmp48761[1], tmp48761[0]};
    assign tmp48763 = {tmp48735[3]};
    assign tmp48764 = tmp48763 ? tmp48762 : tmp48760;
    assign tmp48765 = {tmp48735[4]};
    assign tmp48766 = tmp48765 ? tmp48762 : tmp48764;
    assign tmp48767 = tmp48776;
    assign tmp48768 = {tmp48733[1]};
    assign tmp48769 = tmp48596 | tmp48597;
    assign tmp48770 = tmp48595 & tmp48769;
    assign tmp48771 = ~tmp48596;
    assign tmp48772 = tmp48595 & tmp48771;
    assign tmp48773 = ~tmp48597;
    assign tmp48774 = tmp48772 & tmp48773;
    assign tmp48775 = tmp48774 & tmp48768;
    assign tmp48776 = tmp48770 | tmp48775;
    assign tmp48777 = tmp48781;
    assign tmp48778 = {const_6870_0, const_6870_0, const_6870_0, const_6870_0};
    assign tmp48779 = {tmp48778, tmp48767};
    assign tmp48780 = tmp48733 + tmp48779;
    assign tmp48781 = {tmp48780[4], tmp48780[3], tmp48780[2], tmp48780[1], tmp48780[0]};
    assign tmp48782 = tmp48783;
    assign tmp48783 = {tmp48777[4]};
    assign tmp48784 = tmp48787;
    assign tmp48785 = {tmp48777[3], tmp48777[2], tmp48777[1]};
    assign tmp48786 = {tmp48777[2], tmp48777[1], tmp48777[0]};
    assign tmp48787 = tmp48782 ? tmp48785 : tmp48786;
    assign tmp48788 = tmp48790;
    assign tmp48789 = tmp48491 - tmp48668;
    assign tmp48790 = {tmp48789[3], tmp48789[2], tmp48789[1], tmp48789[0]};
    assign tmp48791 = tmp48795;
    assign tmp48792 = {const_6871_0, const_6871_0, const_6871_0};
    assign tmp48793 = {tmp48792, tmp48782};
    assign tmp48794 = tmp48788 + tmp48793;
    assign tmp48795 = {tmp48794[3], tmp48794[2], tmp48794[1], tmp48794[0]};
    assign tmp48796 = tmp48819;
    assign tmp48797 = tmp48474 ^ tmp48475;
    assign tmp48798 = ~tmp48797;
    assign tmp48799 = {tmp48490[3], tmp48490[2], tmp48490[1], tmp48490[0]};
    assign tmp48800 = {const_6873_0, const_6873_0, const_6873_0};
    assign tmp48801 = {tmp48800, const_6872_0};
    assign tmp48802 = tmp48799 == tmp48801;
    assign tmp48803 = tmp48666 ^ tmp48474;
    assign tmp48804 = ~tmp48798;
    assign tmp48805 = tmp48804 & tmp48802;
    assign tmp48806 = {tmp48490[4]};
    assign tmp48807 = ~tmp48798;
    assign tmp48808 = ~tmp48802;
    assign tmp48809 = tmp48807 & tmp48808;
    assign tmp48810 = tmp48809 & tmp48806;
    assign tmp48811 = ~tmp48798;
    assign tmp48812 = ~tmp48802;
    assign tmp48813 = tmp48811 & tmp48812;
    assign tmp48814 = ~tmp48806;
    assign tmp48815 = tmp48813 & tmp48814;
    assign tmp48816 = tmp48798 ? tmp48474 : const_6874_0;
    assign tmp48817 = tmp48805 ? tmp48803 : tmp48816;
    assign tmp48818 = tmp48810 ? tmp48475 : tmp48817;
    assign tmp48819 = tmp48815 ? tmp48474 : tmp48818;
    assign tmp48820 = tmp48821;
    assign tmp48821 = {tmp48796, tmp48791, tmp48784};
    assign tmp48822 = tmp48383 ? tmp48379 : tmp48385;
    assign tmp48823 = tmp48382 ? tmp48380 : tmp48386;
    assign tmp48824 = tmp48384 ? tmp48820 : tmp48387;

    // Registers
    always @(posedge clk)
    begin
        if (rst) begin
            tmp48385 <= 0;
            tmp48386 <= 0;
            tmp48387 <= 0;
            tmp48472 <= 0;
            tmp48665 <= 0;
            tmp48666 <= 0;
        end
        else begin
            tmp48385 <= tmp48822;
            tmp48386 <= tmp48823;
            tmp48387 <= tmp48824;
            tmp48472 <= tmp48473;
            tmp48665 <= tmp48656;
            tmp48666 <= tmp48654;
        end
    end

endmodule

